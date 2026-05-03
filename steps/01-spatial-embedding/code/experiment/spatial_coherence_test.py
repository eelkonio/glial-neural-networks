"""Spatial coherence test.

Trains with and without spatial coupling using a good embedding and
compares coherence scores to validate the mechanism.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code.embeddings.spectral import SpectralEmbedding
from code.experiment.reproducibility import set_seeds
from code.model import BaselineMLP, get_device
from code.spatial.coherence import SpatialCoherence
from code.spatial.knn_graph import KNNGraph
from code.spatial.lr_coupling import SpatialLRCoupling
from code.visualization.plots import plot_spatial_coherence_comparison


@dataclass
class SpatialCoherenceResult:
    """Result of spatial coherence comparison."""

    coupled_coherence: float
    uncoupled_coherence: float
    mechanism_confirmed: bool


def run_spatial_coherence_test(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 10,
    seed: int = 42,
    results_dir: str | Path | None = None,
) -> SpatialCoherenceResult:
    """Train with and without coupling, compare coherence scores.

    Uses spectral embedding (a good structural embedding) to test whether
    spatial coupling produces more spatially organized weight structure.

    Args:
        train_loader: Training data loader.
        test_loader: Test data loader.
        n_epochs: Number of training epochs.
        seed: Random seed.
        results_dir: Directory to save outputs.

    Returns:
        SpatialCoherenceResult with coupled vs uncoupled scores.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    embedding = SpectralEmbedding()
    coherence_metric = SpatialCoherence(n_components=10)

    # --- Train WITHOUT coupling ---
    print("  Training without coupling...")
    set_seeds(seed)
    model_uncoupled = BaselineMLP().to(device)
    positions = embedding.embed(model_uncoupled.cpu())
    model_uncoupled = model_uncoupled.to(device)

    optimizer = torch.optim.Adam(model_uncoupled.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model_uncoupled.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_uncoupled(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Compute uncoupled coherence
    weights_uncoupled = model_uncoupled.cpu().get_flat_weights().numpy()
    uncoupled_coherence = coherence_metric.compute_coherence(
        weights_uncoupled, positions
    )

    # --- Train WITH coupling ---
    print("  Training with coupling...")
    set_seeds(seed)
    model_coupled = BaselineMLP().to(device)
    # Recompute positions for the fresh model (same embedding, same result)
    positions = embedding.embed(model_coupled.cpu())
    model_coupled = model_coupled.to(device)

    knn_graph = KNNGraph(positions, k=10)
    coupling = SpatialLRCoupling(knn_graph, alpha=0.5)

    optimizer = torch.optim.Adam(model_coupled.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model_coupled.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_coupled(data)
            loss = criterion(output, target)
            loss.backward()
            coupling.apply_to_optimizer(optimizer)
            optimizer.step()

    # Compute coupled coherence
    weights_coupled = model_coupled.cpu().get_flat_weights().numpy()
    coupled_coherence = coherence_metric.compute_coherence(
        weights_coupled, positions
    )

    # Determine if mechanism is confirmed
    mechanism_confirmed = coupled_coherence > uncoupled_coherence

    result = SpatialCoherenceResult(
        coupled_coherence=coupled_coherence,
        uncoupled_coherence=uncoupled_coherence,
        mechanism_confirmed=mechanism_confirmed,
    )

    # Save to CSV
    csv_path = results_dir / "spatial_coherence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "coupled", "uncoupled", "coherence_score"]
        )
        writer.writeheader()
        writer.writerow({
            "method": "spectral",
            "coupled": coupled_coherence,
            "uncoupled": uncoupled_coherence,
            "coherence_score": coupled_coherence - uncoupled_coherence,
        })

    # Generate plot
    plot_spatial_coherence_comparison(
        coupled_score=coupled_coherence,
        uncoupled_score=uncoupled_coherence,
        output_path=results_dir / "spatial_coherence_comparison.png",
    )

    print(f"  Spatial coherence: coupled={coupled_coherence:.4f}, "
          f"uncoupled={uncoupled_coherence:.4f}, "
          f"mechanism_confirmed={mechanism_confirmed}")
    return result
