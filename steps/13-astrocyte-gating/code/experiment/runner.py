"""Experiment runner for Step 13 astrocyte gating experiments.

Orchestrates training runs across conditions and seeds.
Prints UTC timestamps before/after each condition.
Logs metadata and saves per-epoch metrics to CSV.
"""

import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig
from code.experiment.config import GateConfig, ExperimentCondition
from code.experiment.metrics import (
    EpochResult,
    ConditionResult,
    save_epoch_results_csv,
    save_metadata_json,
)
from code.experiment.training import (
    create_gate,
    train_epoch,
    evaluate,
)
from code.step12_imports import (
    ThreeFactorRule,
    LocalMLP,
    get_fashion_mnist_loaders,
    RandomNoiseThirdFactor,
    GlobalRewardThirdFactor,
)


def set_all_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_hardware_info() -> dict[str, str]:
    """Collect hardware and software version info."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
    }
    if torch.cuda.is_available():
        info["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
    return info


def compute_weight_norms(model: LocalMLP) -> float:
    """Compute total Frobenius norm of all layer weights."""
    total = 0.0
    for layer in model.layers:
        total += layer.weight.data.norm().item() ** 2
    return total ** 0.5


def train_backprop(
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cpu",
    verbose: bool = True,
) -> list[EpochResult]:
    """Train with standard backpropagation (Adam optimizer).

    Args:
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for Adam.
        device: Torch device.
        verbose: Print progress.

    Returns:
        List of EpochResult per epoch.
    """
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    model = LocalMLP(
        input_size=784,
        hidden_size=128,
        n_classes=10,
    ).to(device)

    # Use standard PyTorch training with Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_results = []
    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for data, target in train_loader:
            data = data.to(device).view(data.size(0), -1)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data, detach=False)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device).view(data.size(0), -1)
                target = target.to(device)
                output = model(data, detach=False)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                test_batches += 1

        elapsed = time.time() - epoch_start
        acc = correct / max(total, 1)
        w_norm = compute_weight_norms(model)

        has_nan = any(
            torch.isnan(layer.weight.data).any() or torch.isinf(layer.weight.data).any()
            for layer in model.layers
        )

        result = EpochResult(
            epoch=epoch,
            train_loss=total_loss / max(n_batches, 1),
            test_accuracy=acc,
            test_loss=test_loss / max(test_batches, 1),
            gate_fraction_open=1.0,  # N/A for backprop
            weight_norm=w_norm,
            has_nan=has_nan,
            wall_clock_seconds=elapsed,
        )
        epoch_results.append(result)

        if verbose:
            print(f"    Epoch {epoch}: loss={result.train_loss:.4f}, "
                  f"acc={acc:.4f}, norm={w_norm:.2f}")

    return epoch_results


def run_condition(
    condition: ExperimentCondition,
    seed: int,
    n_epochs: int = 50,
    batch_size: int = 128,
    device: str = "cpu",
    checkpoint_interval: int = 10,
    checkpoint_dir: str | None = None,
    verbose: bool = True,
) -> ConditionResult:
    """Run a single condition × seed combination.

    Args:
        condition: Experiment condition specification.
        seed: Random seed.
        n_epochs: Number of epochs.
        batch_size: Batch size.
        device: Torch device.
        checkpoint_interval: Save state every N epochs.
        checkpoint_dir: Directory for checkpoints.
        verbose: Print progress.

    Returns:
        ConditionResult with all epoch metrics.
    """
    set_all_seeds(seed)

    # Special case: backprop baseline
    if condition.name == "backprop":
        epoch_results = train_backprop(
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        final_acc = epoch_results[-1].test_accuracy if epoch_results else 0.0
        best_acc = max(r.test_accuracy for r in epoch_results) if epoch_results else 0.0
        any_nan = any(r.has_nan for r in epoch_results)

        return ConditionResult(
            condition_name=condition.name,
            seed=seed,
            n_epochs=n_epochs,
            final_accuracy=final_acc,
            best_accuracy=best_acc,
            any_nan=any_nan,
            epoch_results=epoch_results,
        )

    # Three-factor rule conditions (with or without gate)
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    model = LocalMLP(
        input_size=784,
        hidden_size=128,
        n_classes=10,
    ).to(device)

    # Create domain assignment
    layer_sizes = []
    prev_size = 784
    for layer in model.layers:
        out_size = layer.linear.out_features
        layer_sizes.append((prev_size, out_size))
        prev_size = out_size

    domain_assignment = DomainAssignment(
        layer_sizes=layer_sizes,
        config=condition.domain_config,
    )

    # Create gate / third factor
    gate = None
    third_factor = None

    if condition.name == "three_factor_random":
        # Step 12 baseline: random noise third factor
        third_factor = RandomNoiseThirdFactor()
    elif condition.name == "three_factor_reward":
        # Step 12 baseline: global reward third factor
        third_factor = GlobalRewardThirdFactor()
    elif condition.gate_config is not None:
        gate = create_gate(
            gate_config=condition.gate_config,
            domain_assignment=domain_assignment,
            calcium_config=condition.calcium_config,
            device=device,
        )
        third_factor = gate

    # Create three-factor rule
    rule = ThreeFactorRule(
        lr=condition.learning_rate,
        tau=condition.tau,
        third_factor=third_factor,
    )

    epoch_results = []
    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model=model,
            rule=rule,
            gate=gate,
            train_loader=train_loader,
            condition=condition,
            device=device,
        )

        # Evaluate
        eval_metrics = evaluate(model, test_loader, device=device)

        # Weight norms
        w_norm = compute_weight_norms(model)

        # NaN check
        has_nan = any(
            torch.isnan(layer.weight.data).any() or torch.isinf(layer.weight.data).any()
            for layer in model.layers
        )

        elapsed = time.time() - epoch_start

        result = EpochResult(
            epoch=epoch,
            train_loss=train_metrics["train_loss"],
            test_accuracy=eval_metrics["test_accuracy"],
            test_loss=eval_metrics["test_loss"],
            gate_fraction_open=train_metrics["gate_fraction_open"],
            weight_norm=w_norm,
            has_nan=has_nan,
            wall_clock_seconds=elapsed,
        )
        epoch_results.append(result)

        if verbose:
            print(f"    Epoch {epoch}: loss={result.train_loss:.4f}, "
                  f"acc={result.test_accuracy:.4f}, "
                  f"gate_open={result.gate_fraction_open:.3f}, "
                  f"norm={w_norm:.2f}")

        # Checkpoint
        if checkpoint_dir and gate and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"{condition.name}_seed{seed}_epoch{epoch+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gate.state_dict(), ckpt_path)

        # Reset eligibility traces between epochs
        rule.reset()

    final_acc = epoch_results[-1].test_accuracy if epoch_results else 0.0
    best_acc = max(r.test_accuracy for r in epoch_results) if epoch_results else 0.0
    any_nan = any(r.has_nan for r in epoch_results)

    return ConditionResult(
        condition_name=condition.name,
        seed=seed,
        n_epochs=n_epochs,
        final_accuracy=final_acc,
        best_accuracy=best_acc,
        any_nan=any_nan,
        epoch_results=epoch_results,
    )


class ExperimentRunner:
    """Orchestrates running multiple conditions across seeds.

    Handles:
    - Seed management
    - UTC timestamp logging
    - Metadata collection
    - CSV output per condition/seed
    - Checkpoint management
    """

    def __init__(
        self,
        conditions: list[ExperimentCondition],
        seeds: list[int] = None,
        n_epochs: int = 50,
        batch_size: int = 128,
        device: str = "cpu",
        output_dir: str = "results",
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 10,
        verbose: bool = True,
    ):
        self.conditions = conditions
        self.seeds = seeds or [42, 123, 456]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose

        # Timestamp for this experiment run
        self.run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def run_all(self) -> list[ConditionResult]:
        """Run all conditions × seeds.

        Returns:
            List of ConditionResult for each condition/seed combo.
        """
        all_results = []

        # Save metadata
        metadata = {
            "run_timestamp": self.run_timestamp,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "seeds": self.seeds,
            "conditions": [c.name for c in self.conditions],
            "device": self.device,
            "hardware": get_hardware_info(),
        }
        save_metadata_json(metadata, self.output_dir, self.run_timestamp)

        for condition in self.conditions:
            for seed in self.seeds:
                result = self.run_single(condition, seed)
                all_results.append(result)

        return all_results

    def run_single(
        self,
        condition: ExperimentCondition,
        seed: int,
    ) -> ConditionResult:
        """Run a single condition × seed with timestamps.

        Args:
            condition: Experiment condition.
            seed: Random seed.

        Returns:
            ConditionResult.
        """
        # UTC timestamp before
        start_time = datetime.now(timezone.utc)
        print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
              f"Starting: {condition.name} seed={seed}")

        t0 = time.time()

        result = run_condition(
            condition=condition,
            seed=seed,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            device=self.device,
            checkpoint_interval=self.checkpoint_interval,
            checkpoint_dir=self.checkpoint_dir,
            verbose=self.verbose,
        )

        elapsed = time.time() - t0

        # UTC timestamp after
        end_time = datetime.now(timezone.utc)
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
              f"Completed: {condition.name} seed={seed} ({elapsed:.1f}s) "
              f"acc={result.final_accuracy:.4f}")

        # Save CSV
        save_epoch_results_csv(
            results=result.epoch_results,
            condition_name=condition.name,
            seed=seed,
            output_dir=self.output_dir,
            timestamp=self.run_timestamp,
        )

        return result

    def run_condition_all_seeds(
        self,
        condition: ExperimentCondition,
    ) -> list[ConditionResult]:
        """Run a single condition across all seeds.

        Args:
            condition: Experiment condition.

        Returns:
            List of ConditionResult (one per seed).
        """
        results = []
        for seed in self.seeds:
            result = self.run_single(condition, seed)
            results.append(result)
        return results
