"""Step 01 v2: Spatial embedding experiments with harder conditions.

Changes from v1:
- FashionMNIST instead of MNIST (harder task, ~89% accuracy ceiling with MLP)
- 4-layer MLP (784→128→128→128→128→10) — deeper, narrower, harder optimization
- 50 epochs (not 10) — more time for spatial dynamics to differentiate
- 50 batches for gradient correlation (not 5) — more stable estimates
- 5000 subsample for MDS (not 500) — better representation
- 1M max pairs for quality measurement (not 100K)

Note: Originally planned CIFAR-10 but the Toronto download server returned 503.
FashionMNIST is a suitable alternative: same image size, harder than MNIST,
reliable download, and the MLP architecture achieves ~89% (not saturated).

Usage:
    .venv/bin/python steps/01-spatial-embedding-v2/run_v2.py
"""

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add v1 code to path (reuse embedding and spatial modules)
sys.path.insert(0, str(Path(__file__).parent.parent / "01-spatial-embedding"))

from code.embeddings import (
    AdversarialEmbedding,
    CorrelationEmbedding,
    DevelopmentalEmbedding,
    DifferentiableEmbedding,
    LayeredClusteredEmbedding,
    LinearEmbedding,
    RandomEmbedding,
    SpectralEmbedding,
)
from code.experiment.reproducibility import (
    get_git_hash,
    get_hardware_info,
    get_library_versions,
    set_seeds,
)
from code.spatial.coherence import SpatialCoherence
from code.spatial.knn_graph import KNNGraph
from code.spatial.lr_coupling import SpatialLRCoupling
from code.spatial.quality import QualityMeasurement
from code.visualization.plots import (
    plot_boundary_regression,
    plot_quality_vs_performance,
    plot_three_point_curve,
    plot_spatial_coherence_comparison,
)

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path(__file__).parent / "results"
N_EPOCHS = 50
N_SEEDS = 3
SEEDS = [42, 123, 456]
BATCH_SIZE = 128
LR = 1e-3
GRADIENT_BATCHES = 50
MDS_SUBSAMPLE = 5000
QUALITY_MAX_PAIRS = 1_000_000
COUPLING_K = 10
COUPLING_ALPHA = 0.5


# ============================================================
# Model: Deeper, narrower MLP for CIFAR-10
# ============================================================


class DeeperMLP(nn.Module):
    """4-hidden-layer MLP for FashionMNIST.

    Architecture: 784 → 128 → 128 → 128 → 128 → 10
    Deeper and narrower than v1 — harder optimization landscape.
    Expected accuracy: ~89% (not saturated like MNIST at 97.9%).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    @property
    def weight_layers(self) -> list[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def get_weight_count(self) -> int:
        return sum(layer.weight.numel() for layer in self.weight_layers)

    def get_weight_metadata(self):
        from code.model import WeightInfo
        metadata = []
        flat_idx = 0
        for layer_idx, layer in enumerate(self.weight_layers):
            out_features, in_features = layer.weight.shape
            for target in range(out_features):
                for source in range(in_features):
                    metadata.append(WeightInfo(
                        layer_idx=layer_idx,
                        source_neuron=source,
                        target_neuron=target,
                        flat_idx=flat_idx,
                    ))
                    flat_idx += 1
        return metadata

    def get_layer_info(self) -> list[tuple[int, int, int]]:
        return [
            (i, layer.in_features, layer.out_features)
            for i, layer in enumerate(self.weight_layers)
        ]

    def get_flat_weights(self) -> torch.Tensor:
        return torch.cat(
            [layer.weight.detach().flatten() for layer in self.weight_layers]
        )

    def get_flat_gradients(self) -> torch.Tensor:
        grads = []
        for layer in self.weight_layers:
            if layer.weight.grad is None:
                raise RuntimeError(f"No gradient for {layer}. Call backward() first.")
            grads.append(layer.weight.grad.detach().flatten())
        return torch.cat(grads)


# ============================================================
# Data: CIFAR-10
# ============================================================


def get_cifar10_loaders() -> tuple[DataLoader, DataLoader]:
    """Load FashionMNIST (harder than MNIST, reliable download)."""
    data_dir = Path(__file__).parent / "data" / "fashionmnist"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# ============================================================
# Device
# ============================================================


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Experiment runner
# ============================================================


@dataclass
class ConditionResult:
    condition_name: str
    seed: int
    final_test_accuracy: float
    steps_to_95pct: int | None
    quality_score: float
    coherence_score: float
    wall_clock_seconds: float


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


def run_condition(
    condition_name: str,
    embedding,
    coupling_enabled: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
) -> ConditionResult:
    """Run a single experimental condition."""
    set_seeds(seed)
    device = get_device()
    start = time.time()

    model = DeeperMLP().to(device)

    # Compute embedding positions (on CPU for numpy ops)
    positions = None
    coupling = None
    if embedding is not None:
        model_cpu = model.cpu()
        positions = embedding.embed(model_cpu, data_loader=train_loader)
        model = model_cpu.to(device)

        if coupling_enabled:
            knn_graph = KNNGraph(positions, k=COUPLING_K)
            coupling = SpatialLRCoupling(knn_graph, alpha=COUPLING_ALPHA)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    epoch_accuracies = []
    for epoch in range(N_EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if coupling is not None:
                coupling.apply_to_optimizer(optimizer)
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        epoch_accuracies.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"    [{condition_name}] seed={seed} epoch={epoch+1} acc={acc:.4f}")

    final_acc = epoch_accuracies[-1]

    # Steps to 95% of final
    target_acc = 0.95 * final_acc
    steps_to_95 = None
    steps_per_epoch = len(train_loader)
    for i, acc in enumerate(epoch_accuracies):
        if acc >= target_acc:
            steps_to_95 = (i + 1) * steps_per_epoch
            break

    # Quality score
    quality_score = 0.0
    coherence_score = 0.0
    if positions is not None:
        try:
            model_cpu = model.cpu()
            qm = QualityMeasurement(positions, max_pairs=QUALITY_MAX_PAIRS)
            qr = qm.compute_quality_score(model_cpu, train_loader, n_batches=GRADIENT_BATCHES)
            quality_score = qr.score
            model = model_cpu.to(device)
        except Exception as e:
            print(f"    Quality measurement failed: {e}")

        try:
            weights = model.cpu().get_flat_weights().numpy()
            sc = SpatialCoherence(n_components=10)
            coherence_score = sc.compute_coherence(weights, positions)
            model = model.to(device)
        except Exception as e:
            print(f"    Coherence measurement failed: {e}")

    elapsed = time.time() - start

    return ConditionResult(
        condition_name=condition_name,
        seed=seed,
        final_test_accuracy=final_acc,
        steps_to_95pct=steps_to_95,
        quality_score=quality_score,
        coherence_score=coherence_score,
        wall_clock_seconds=elapsed,
    )


# ============================================================
# Main experiment
# ============================================================


def get_conditions():
    """Define all experimental conditions with v2 parameters."""
    return [
        ("uncoupled_baseline", None, False),
        ("linear_coupled", LinearEmbedding(), True),
        ("random_coupled", RandomEmbedding(seed=42), True),
        ("spectral_coupled", SpectralEmbedding(), True),
        ("layered_clustered_coupled", LayeredClusteredEmbedding(), True),
        ("correlation_coupled", CorrelationEmbedding(n_batches=GRADIENT_BATCHES, subsample_size=MDS_SUBSAMPLE), True),
        ("developmental_coupled", DevelopmentalEmbedding(n_steps=500, subsample_pairs=50000, n_correlation_batches=GRADIENT_BATCHES), True),
        ("adversarial_coupled", AdversarialEmbedding(n_correlation_batches=GRADIENT_BATCHES, subsample_size=MDS_SUBSAMPLE), True),
        ("differentiable", DifferentiableEmbedding(lambda_spatial=0.01, subsample_pairs=50000), False),
        ("differentiable_coupled", DifferentiableEmbedding(lambda_spatial=0.01, subsample_pairs=50000), True),
    ]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 01 v2: SPATIAL EMBEDDING (HARDER CONDITIONS)")
    print("=" * 70)
    print(f"Architecture: 784→128→128→128→128→10 (DeeperMLP)")
    print(f"Task: FashionMNIST")
    print(f"Epochs: {N_EPOCHS}, Seeds: {SEEDS}")
    print(f"Gradient batches: {GRADIENT_BATCHES}, MDS subsample: {MDS_SUBSAMPLE}")
    print(f"Quality max pairs: {QUALITY_MAX_PAIRS}")
    print(f"Device: {get_device()}")
    print(f"Weight count: {DeeperMLP().get_weight_count()}")
    print()

    # Load data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders()
    print(f"Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")
    print()

    # Log metadata
    import json
    metadata = {
        "experiment": "step01_v2_harder_conditions",
        "architecture": "784→128→128→128→128→10",
        "task": "FashionMNIST",
        "n_epochs": N_EPOCHS,
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "gradient_batches": GRADIENT_BATCHES,
        "mds_subsample": MDS_SUBSAMPLE,
        "quality_max_pairs": QUALITY_MAX_PAIRS,
        "coupling_k": COUPLING_K,
        "coupling_alpha": COUPLING_ALPHA,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "hardware": get_hardware_info(),
        "library_versions": get_library_versions(),
        "git_hash": get_git_hash(),
    }
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Run all conditions
    conditions = get_conditions()
    all_results: list[ConditionResult] = []

    total_start = time.time()

    for cond_name, embedding, coupling_enabled in conditions:
        print(f"\n{'─' * 50}")
        print(f"Condition: {cond_name}")
        print(f"{'─' * 50}")

        for seed in SEEDS:
            result = run_condition(
                cond_name, embedding, coupling_enabled,
                train_loader, test_loader, seed
            )
            all_results.append(result)
            print(f"  seed={seed}: acc={result.final_test_accuracy:.4f}, "
                  f"quality={result.quality_score:.4f}, "
                  f"coherence={result.coherence_score:.4f}, "
                  f"time={result.wall_clock_seconds:.1f}s")

    total_time = time.time() - total_start

    # Save results CSV
    csv_path = RESULTS_DIR / "comparison_results.csv"
    fieldnames = [
        "condition", "seed", "final_test_accuracy", "steps_to_95pct",
        "quality_score", "coherence_score", "wall_clock_seconds"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "condition": r.condition_name,
                "seed": r.seed,
                "final_test_accuracy": r.final_test_accuracy,
                "steps_to_95pct": r.steps_to_95pct,
                "quality_score": r.quality_score,
                "coherence_score": r.coherence_score,
                "wall_clock_seconds": r.wall_clock_seconds,
            })

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 70}")

    baseline_acc = np.mean([r.final_test_accuracy for r in all_results if r.condition_name == "uncoupled_baseline"])

    condition_names = list(dict.fromkeys(r.condition_name for r in all_results))
    quality_scores = []
    performance_deltas = []
    labels = []

    for cond in condition_names:
        cond_results = [r for r in all_results if r.condition_name == cond]
        mean_acc = np.mean([r.final_test_accuracy for r in cond_results])
        std_acc = np.std([r.final_test_accuracy for r in cond_results])
        mean_quality = np.mean([r.quality_score for r in cond_results])
        mean_coherence = np.mean([r.coherence_score for r in cond_results])
        delta = mean_acc - baseline_acc

        print(f"  {cond:30s} acc={mean_acc:.4f}±{std_acc:.4f} "
              f"quality={mean_quality:.4f} coherence={mean_coherence:.4f} "
              f"delta={delta:+.4f}")

        if cond != "uncoupled_baseline":
            quality_scores.append(mean_quality)
            performance_deltas.append(delta)
            labels.append(cond)

    # Generate plots
    plot_quality_vs_performance(
        quality_scores, performance_deltas, labels,
        output_path=RESULTS_DIR / "embedding_vs_performance.png"
    )

    # Boundary regression
    from scipy.stats import pearsonr, linregress
    if len(quality_scores) >= 3:
        r_val, p_val = pearsonr(quality_scores, performance_deltas)
        print(f"\n  Boundary condition: r={r_val:.4f}, p={p_val:.4f}")

        plot_boundary_regression(
            quality_scores, performance_deltas,
            output_path=RESULTS_DIR / "boundary_regression.png"
        )

        # Save boundary CSV
        with open(RESULTS_DIR / "boundary_condition.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "quality_score", "performance_delta"])
            writer.writeheader()
            for label, qs, pd in zip(labels, quality_scores, performance_deltas):
                writer.writerow({"method": label, "quality_score": qs, "performance_delta": pd})

    # Three-point validation
    adv_delta = np.mean([r.final_test_accuracy for r in all_results if r.condition_name == "adversarial_coupled"]) - baseline_acc
    rand_delta = np.mean([r.final_test_accuracy for r in all_results if r.condition_name == "random_coupled"]) - baseline_acc
    best_delta = max(performance_deltas) if performance_deltas else 0.0

    monotonic = adv_delta < rand_delta < best_delta
    print(f"\n  Three-point: adversarial={adv_delta:+.4f}, random={rand_delta:+.4f}, best={best_delta:+.4f}")
    print(f"  Monotonic: {monotonic}")

    plot_three_point_curve(
        adv_delta, rand_delta, best_delta,
        output_path=RESULTS_DIR / "three_point_curve.png"
    )

    # Spatial coherence comparison
    coupled_coherences = [r.coherence_score for r in all_results
                         if r.condition_name == "spectral_coupled"]
    uncoupled_coherences = [r.coherence_score for r in all_results
                           if r.condition_name == "uncoupled_baseline"]
    if coupled_coherences and uncoupled_coherences:
        coupled_mean = np.mean(coupled_coherences)
        uncoupled_mean = np.mean(uncoupled_coherences)
        print(f"\n  Spatial coherence: coupled={coupled_mean:.4f}, uncoupled={uncoupled_mean:.4f}")
        plot_spatial_coherence_comparison(
            coupled_mean, uncoupled_mean,
            output_path=RESULTS_DIR / "spatial_coherence_comparison.png"
        )

    print(f"\n{'=' * 70}")
    print(f"COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
