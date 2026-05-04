"""Performance metrics for local learning rule experiments.

Records per-epoch metrics, computes convergence speed, stability,
and linear probe accuracy on frozen hidden layer activations.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    """Metrics recorded at each epoch."""

    epoch: int
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    weight_norms: list[float] = field(default_factory=list)
    representation_quality: list[float] = field(default_factory=list)


def compute_convergence_epoch(accuracy_history: list[float]) -> int | None:
    """Find the first epoch reaching 90% of final accuracy.

    Args:
        accuracy_history: List of accuracy values per epoch.

    Returns:
        First epoch index where accuracy >= 0.9 * max(accuracy),
        or None if threshold is never reached.
    """
    if not accuracy_history:
        return None

    max_acc = max(accuracy_history)
    threshold = 0.9 * max_acc

    for i, acc in enumerate(accuracy_history):
        if acc >= threshold:
            return i

    return None


def compute_stability(accuracy_history: list[float], window: int = 10) -> float:
    """Compute stability as std of test accuracy over final epochs.

    Args:
        accuracy_history: List of accuracy values per epoch.
        window: Number of final epochs to consider.

    Returns:
        Standard deviation of accuracy over the final `window` epochs.
    """
    if len(accuracy_history) < window:
        return float(np.std(accuracy_history)) if accuracy_history else 0.0
    return float(np.std(accuracy_history[-window:]))


def compute_weight_norms(model: nn.Module) -> list[float]:
    """Compute L2 norm of weights for each layer.

    Args:
        model: The LocalMLP model.

    Returns:
        List of L2 norms, one per layer.
    """
    norms = []
    for layer in model.layers:
        norm = layer.linear.weight.data.norm(2).item()
        norms.append(norm)
    return norms


def linear_probe_accuracy(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    layer_index: int = 2,
    n_epochs: int = 10,
    lr: float = 0.01,
) -> float:
    """Train a linear classifier on frozen hidden layer activations.

    Args:
        model: The trained LocalMLP model.
        train_loader: Training data loader.
        test_loader: Test data loader.
        device: Device to run on.
        layer_index: Which hidden layer to probe (0-indexed).
        n_epochs: Number of training epochs for the probe.
        lr: Learning rate for the probe.

    Returns:
        Test accuracy of the linear probe.
    """
    model.eval()

    # Collect activations from the specified layer
    train_features = []
    train_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device).view(images.size(0), -1)
            activations = model.get_layer_activations(images)
            train_features.append(activations[layer_index].cpu())
            train_labels.append(labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Train a simple linear probe
    n_features = train_features.shape[1]
    probe = nn.Linear(n_features, 10).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Simple training loop on collected features
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    probe_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    probe.train()
    for _ in range(n_epochs):
        for feats, labs in probe_loader:
            feats, labs = feats.to(device), labs.to(device)
            logits = probe(feats)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    test_features = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device).view(images.size(0), -1)
            activations = model.get_layer_activations(images)
            test_features.append(activations[layer_index].cpu())
            test_labels.append(labels)

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    probe.eval()
    with torch.no_grad():
        logits = probe(test_features.to(device))
        predictions = logits.argmax(dim=1)
        correct = (predictions == test_labels.to(device)).sum().item()

    return correct / len(test_labels)


class PerformanceMetrics:
    """Collects and stores per-epoch performance metrics.

    Tracks train/test accuracy, loss, weight norms, and representation
    quality for a single rule/seed combination.
    """

    def __init__(self, rule_name: str, seed: int):
        self.rule_name = rule_name
        self.seed = seed
        self.epochs: list[EpochMetrics] = []

    def record_epoch(
        self,
        epoch: int,
        train_accuracy: float,
        test_accuracy: float,
        train_loss: float,
        test_loss: float,
        weight_norms: list[float] | None = None,
        representation_quality: list[float] | None = None,
    ) -> None:
        """Record metrics for one epoch."""
        self.epochs.append(
            EpochMetrics(
                epoch=epoch,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                train_loss=train_loss,
                test_loss=test_loss,
                weight_norms=weight_norms or [],
                representation_quality=representation_quality or [],
            )
        )

    @property
    def accuracy_history(self) -> list[float]:
        """Test accuracy values across all epochs."""
        return [e.test_accuracy for e in self.epochs]

    @property
    def convergence_epoch(self) -> int | None:
        """First epoch reaching 90% of final accuracy."""
        return compute_convergence_epoch(self.accuracy_history)

    @property
    def stability(self) -> float:
        """Std of test accuracy over final 10 epochs."""
        return compute_stability(self.accuracy_history)

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Convert to list of CSV-compatible row dicts."""
        rows = []
        for e in self.epochs:
            rows.append({
                "rule": self.rule_name,
                "seed": self.seed,
                "epoch": e.epoch,
                "train_accuracy": e.train_accuracy,
                "test_accuracy": e.test_accuracy,
                "train_loss": e.train_loss,
                "test_loss": e.test_loss,
                "weight_norms": ";".join(f"{n:.4f}" for n in e.weight_norms),
                "repr_quality": ";".join(
                    f"{q:.4f}" for q in e.representation_quality
                ),
            })
        return rows

    @staticmethod
    def save_all_to_csv(
        metrics_list: list["PerformanceMetrics"], path: Path
    ) -> None:
        """Save multiple PerformanceMetrics to a single CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "rule",
            "seed",
            "epoch",
            "train_accuracy",
            "test_accuracy",
            "train_loss",
            "test_loss",
            "weight_norms",
            "repr_quality",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in metrics_list:
                writer.writerows(metrics.to_csv_rows())
