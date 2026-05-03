"""Temporal quality tracking experiment.

Tracks embedding quality at intervals during training for fixed embeddings
and compares against the differentiable embedding which co-adapts.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code.embeddings.base import EmbeddingStrategy
from code.embeddings.differentiable import DifferentiableEmbedding
from code.embeddings.linear import LinearEmbedding
from code.embeddings.random import RandomEmbedding
from code.embeddings.spectral import SpectralEmbedding
from code.experiment.reproducibility import set_seeds
from code.model import BaselineMLP, get_device
from code.spatial.quality import QualityMeasurement
from code.spatial.temporal_tracking import TemporalQualityTracker
from code.visualization.plots import plot_temporal_quality


@dataclass
class TemporalQualityResult:
    """Result of temporal quality tracking for one method."""

    method_name: str
    trajectory: list[tuple[int, float]]  # (epoch, score)
    initial_quality: float
    final_quality: float
    degraded: bool
    min_quality: float
    min_quality_epoch: int


def run_temporal_quality_tracking(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 10,
    record_interval: int = 2,
    seed: int = 42,
    results_dir: str | Path | None = None,
) -> list[TemporalQualityResult]:
    """Track quality at intervals during training for multiple embeddings.

    Tests whether fixed embeddings degrade as the network's functional
    structure evolves during learning.

    Args:
        train_loader: Training data loader.
        test_loader: Test data loader.
        n_epochs: Number of training epochs.
        record_interval: Record quality every N epochs.
        seed: Random seed.
        results_dir: Directory to save outputs.

    Returns:
        List of TemporalQualityResult for each method.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Embeddings to track (use fast ones for tractability)
    embeddings: list[tuple[str, EmbeddingStrategy]] = [
        ("linear", LinearEmbedding()),
        ("random", RandomEmbedding(seed=42)),
        ("spectral", SpectralEmbedding()),
    ]

    all_results: list[TemporalQualityResult] = []
    trajectories_dict: dict[str, list[tuple[int, float]]] = {}

    for method_name, embedding in embeddings:
        print(f"  Tracking temporal quality for: {method_name}")
        set_seeds(seed)

        model = BaselineMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Compute embedding positions
        positions = embedding.embed(model.cpu())
        model = model.to(device)

        # Set up quality measurement
        quality_measurement = QualityMeasurement(positions, max_pairs=100_000)
        tracker = TemporalQualityTracker(record_interval_epochs=record_interval)

        # Training loop with quality tracking
        step = 0
        for epoch in range(n_epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                step += 1

            # Record quality at intervals
            if (epoch + 1) % record_interval == 0:
                model_cpu = model.cpu()
                tracker.record(
                    epoch=epoch + 1,
                    step=step,
                    quality_measurement=quality_measurement,
                    model=model_cpu,
                    data_loader=train_loader,
                )
                model = model_cpu.to(device)

        # Collect results
        trajectory = tracker.get_trajectory()
        epoch_scores = [(epoch, score) for epoch, _, score in trajectory]
        degraded = tracker.detect_degradation(threshold=0.5)

        scores_only = [score for _, _, score in trajectory]
        initial_quality = scores_only[0] if scores_only else 0.0
        final_quality = scores_only[-1] if scores_only else 0.0
        min_quality = min(scores_only) if scores_only else 0.0
        min_epoch = trajectory[scores_only.index(min_quality)][0] if scores_only else 0

        result = TemporalQualityResult(
            method_name=method_name,
            trajectory=epoch_scores,
            initial_quality=initial_quality,
            final_quality=final_quality,
            degraded=degraded,
            min_quality=min_quality,
            min_quality_epoch=min_epoch,
        )
        all_results.append(result)
        trajectories_dict[method_name] = epoch_scores

    # Save to CSV
    csv_path = results_dir / "temporal_quality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "epoch", "quality_score"]
        )
        writer.writeheader()
        for result in all_results:
            for epoch, score in result.trajectory:
                writer.writerow({
                    "method": result.method_name,
                    "epoch": epoch,
                    "quality_score": score,
                })

    # Generate plot
    plot_temporal_quality(
        trajectories_dict=trajectories_dict,
        output_path=results_dir / "temporal_quality_trajectories.png",
    )

    return all_results
