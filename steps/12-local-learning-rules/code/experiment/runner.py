"""Experiment runner for local learning rules.

Provides training loops for all rule types:
- Backprop baseline (standard gradient descent)
- Hebbian/Oja/Three-factor (compute_update interface)
- Forward-forward (per-layer optimizer with goodness)
- Predictive coding (inference iterations + weight update)

Also provides ExperimentRunner class for orchestrating full experiments
with metadata logging, checkpointing, and multi-seed execution.

Adapted from steps/01-spatial-embedding experiment runner patterns.
"""

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from code.network.local_mlp import LocalMLP
from code.data.fashion_mnist import (
    get_fashion_mnist_loaders,
    ForwardForwardDataAdapter,
)
from code.experiment.metrics import (
    PerformanceMetrics,
    compute_weight_norms,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_accuracy(
    model: LocalMLP,
    test_loader: DataLoader,
    device: torch.device,
    detach: bool = True,
) -> float:
    """Evaluate model accuracy on a test set.

    Args:
        model: The LocalMLP model.
        test_loader: Test data loader.
        device: Device to run on.
        detach: Whether to use detached forward (local mode).

    Returns:
        Test accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, detach=detach)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


def train_backprop(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train with standard backpropagation (baseline).

    Uses LocalMLP with detach=False for full gradient flow,
    Adam optimizer, and CrossEntropyLoss.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for Adam.
        seed: Random seed.
        device: Device to train on.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    model = LocalMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "epoch": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward with full gradient flow (no detach)
            logits = model(images, detach=False)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        test_acc = evaluate_accuracy(model, test_loader, device, detach=False)

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(test_acc)
        history["epoch"].append(epoch)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"  Backprop epoch {epoch+1}/{epochs}: "
                f"loss={avg_loss:.4f}, test_acc={test_acc:.4f}"
            )

    return {
        "model": model,
        "history": history,
        "final_accuracy": history["test_accuracy"][-1],
        "rule_name": "backprop",
        "seed": seed,
    }


def train_local_rule(
    rule,
    epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train with a local learning rule (Hebbian, Oja, ThreeFactor).

    Uses the compute_update interface: forward pass collects LayerState,
    then rule computes weight deltas applied directly.

    Args:
        rule: A local learning rule with compute_update method.
        epochs: Number of training epochs.
        batch_size: Batch size.
        seed: Random seed.
        device: Device to train on.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    model = LocalMLP().to(device)
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "epoch": [],
    }

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            x = images.view(images.size(0), -1)

            # Forward pass collecting states
            with torch.no_grad():
                logits = model(x, detach=True)
                loss = criterion(logits, labels)

            # Collect states with loss info
            states = model.forward_with_states(x, labels=labels, global_loss=loss.item())

            # Apply local rule to each layer
            with torch.no_grad():
                for state in states:
                    delta = rule.compute_update(state)
                    layer = model.layers[state.layer_index]
                    layer.linear.weight.data += delta

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        test_acc = evaluate_accuracy(model, test_loader, device, detach=True)

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(test_acc)
        history["epoch"].append(epoch)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"  {rule.name} epoch {epoch+1}/{epochs}: "
                f"loss={avg_loss:.4f}, test_acc={test_acc:.4f}"
            )

        # Reset rule state between epochs if needed
        rule.reset()

    return {
        "model": model,
        "history": history,
        "final_accuracy": history["test_accuracy"][-1],
        "rule_name": rule.name,
        "seed": seed,
    }


def train_forward_forward(
    rule,
    epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train with the forward-forward algorithm.

    Uses per-layer optimizers and goodness-based loss.

    Args:
        rule: A ForwardForwardRule instance.
        epochs: Number of training epochs.
        batch_size: Batch size.
        seed: Random seed.
        device: Device to train on.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    model = LocalMLP().to(device)
    rule.setup_optimizers(model)

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)
    ff_adapter = ForwardForwardDataAdapter(train_loader)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "epoch": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_pos, x_neg, labels in ff_adapter:
            x_pos, x_neg = x_pos.to(device), x_neg.to(device)
            losses = rule.train_step(model, x_pos, x_neg)
            epoch_loss += sum(losses) / len(losses)
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate using FF classification
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device).view(images.size(0), -1)
                labels = labels.to(device)
                predictions = rule.classify(model, images)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        model.train()

        test_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(avg_loss)
        history["test_accuracy"].append(test_acc)
        history["epoch"].append(epoch)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"  forward_forward epoch {epoch+1}/{epochs}: "
                f"loss={avg_loss:.4f}, test_acc={test_acc:.4f}"
            )

    return {
        "model": model,
        "history": history,
        "final_accuracy": history["test_accuracy"][-1],
        "rule_name": "forward_forward",
        "seed": seed,
    }


def train_predictive_coding(
    rule,
    epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train with predictive coding.

    Uses inference iterations followed by local weight updates.

    Args:
        rule: A PredictiveCodingRule instance.
        epochs: Number of training epochs.
        batch_size: Batch size.
        seed: Random seed.
        device: Device to train on.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    model = LocalMLP().to(device)
    rule.setup_predictions(model)

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "epoch": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_error = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            x = images.view(images.size(0), -1)

            error = rule.train_step(model, x, labels)
            epoch_error += error
            n_batches += 1

        avg_error = epoch_error / n_batches
        test_acc = evaluate_accuracy(model, test_loader, device, detach=True)

        history["train_loss"].append(avg_error)
        history["test_accuracy"].append(test_acc)
        history["epoch"].append(epoch)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"  predictive_coding epoch {epoch+1}/{epochs}: "
                f"error={avg_error:.4f}, test_acc={test_acc:.4f}"
            )

    return {
        "model": model,
        "history": history,
        "final_accuracy": history["test_accuracy"][-1],
        "rule_name": "predictive_coding",
        "seed": seed,
    }


def _get_metadata() -> dict[str, Any]:
    """Collect experiment metadata (hardware, library versions, etc.)."""
    import numpy as np

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    return metadata


def _save_checkpoint(
    model: LocalMLP,
    rule_name: str,
    seed: int,
    epoch: int,
    data_dir: Path,
) -> None:
    """Save model checkpoint to data directory.

    Args:
        model: The model to checkpoint.
        rule_name: Name of the learning rule.
        seed: Random seed used.
        epoch: Current epoch number.
        data_dir: Directory to save checkpoints.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = data_dir / f"checkpoint_{rule_name}_seed{seed}_epoch{epoch}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "rule_name": rule_name,
            "seed": seed,
            "epoch": epoch,
        },
        checkpoint_path,
    )


class ExperimentRunner:
    """Orchestrates local learning rule experiments.

    Handles seed management, metadata logging, periodic checkpointing,
    and multi-seed execution for all rule types.
    """

    def __init__(
        self,
        results_dir: Path | None = None,
        data_dir: Path | None = None,
        checkpoint_interval: int = 10,
    ):
        """Initialize the experiment runner.

        Args:
            results_dir: Directory for results output.
            data_dir: Directory for checkpoints and cached data.
            checkpoint_interval: Save checkpoint every N epochs.
        """
        base = Path(__file__).parent.parent.parent
        self.results_dir = results_dir or base / "results"
        self.data_dir = data_dir or base / "data"
        self.checkpoint_interval = checkpoint_interval
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run_rule(
        self,
        rule_name: str,
        train_fn: Callable[..., dict],
        n_epochs: int = 50,
        seeds: list[int] | None = None,
        batch_size: int = 128,
        verbose: bool = True,
        **train_kwargs,
    ) -> list[dict]:
        """Run a learning rule across multiple seeds.

        Args:
            rule_name: Name of the rule for logging.
            train_fn: Training function (train_backprop, train_local_rule, etc.)
            n_epochs: Number of epochs.
            seeds: List of random seeds.
            batch_size: Batch size.
            verbose: Whether to print progress.
            **train_kwargs: Additional kwargs passed to train_fn.

        Returns:
            List of result dicts, one per seed.
        """
        if seeds is None:
            seeds = [42, 123, 456]

        device = get_device()
        results = []

        for seed in seeds:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running {rule_name} with seed={seed}")
                print(f"{'='*60}")

            start_time = time.time()

            result = train_fn(
                epochs=n_epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                verbose=verbose,
                **train_kwargs,
            )

            elapsed = time.time() - start_time
            result["wall_clock_seconds"] = elapsed
            result["seed"] = seed

            # Save final checkpoint
            if "model" in result:
                _save_checkpoint(
                    result["model"],
                    rule_name,
                    seed,
                    n_epochs - 1,
                    self.data_dir / "checkpoints",
                )

            results.append(result)

            if verbose:
                print(
                    f"  Completed in {elapsed:.1f}s, "
                    f"final accuracy: {result.get('final_accuracy', 0):.4f}"
                )

        return results

    def save_metadata(
        self,
        config: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save experiment metadata to JSON.

        Args:
            config: Experiment configuration dict.
            extra: Additional metadata to include.

        Returns:
            Path to the saved metadata file.
        """
        metadata = _get_metadata()
        metadata["config"] = config
        if extra:
            metadata.update(extra)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"metadata_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return path

    def collect_metrics(
        self,
        result: dict,
        rule_name: str,
        model: LocalMLP | None = None,
        train_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> PerformanceMetrics:
        """Collect PerformanceMetrics from a training result dict.

        Args:
            result: Result dict from a training function.
            rule_name: Name of the rule.
            model: Trained model (for weight norms).
            train_loader: Training data loader (for linear probe).
            test_loader: Test data loader (for linear probe).
            device: Device.

        Returns:
            PerformanceMetrics object with all recorded data.
        """
        seed = result.get("seed", 0)
        metrics = PerformanceMetrics(rule_name=rule_name, seed=seed)
        history = result.get("history", {})

        epochs = history.get("epoch", [])
        train_losses = history.get("train_loss", [])
        test_accs = history.get("test_accuracy", [])

        for i, epoch in enumerate(epochs):
            train_loss = train_losses[i] if i < len(train_losses) else 0.0
            test_acc = test_accs[i] if i < len(test_accs) else 0.0

            # Weight norms from model at final state (approximate)
            weight_norms = []
            if model is not None and i == len(epochs) - 1:
                weight_norms = compute_weight_norms(model)

            metrics.record_epoch(
                epoch=epoch,
                train_accuracy=0.0,  # Not tracked in current training loops
                test_accuracy=test_acc,
                train_loss=train_loss,
                test_loss=0.0,  # Not separately tracked
                weight_norms=weight_norms,
            )

        return metrics
