"""Developmental convergence analysis.

Runs the developmental embedding with quality tracking and analyzes
whether the quality score converges during position optimization.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from code.embeddings.developmental import DevelopmentalEmbedding
from code.model import BaselineMLP, get_device
from code.visualization.plots import plot_developmental_trajectory


@dataclass
class ConvergenceResult:
    """Result of developmental embedding convergence analysis."""

    quality_trajectory: list[float]
    converged: bool
    final_quality: float
    n_steps_to_stability: int | None


def detect_convergence(
    trajectory: list[float],
    threshold: float = 0.05,
) -> tuple[bool, int | None]:
    """Detect convergence in a quality trajectory.

    Convergence is defined as: max relative change between consecutive
    values in the final 20% of the trajectory is less than threshold.

    Args:
        trajectory: List of quality scores over time.
        threshold: Maximum relative change for convergence (default 5%).

    Returns:
        (converged, step_index) where step_index is the first step in the
        convergence region, or None if not converged.
    """
    if len(trajectory) < 5:
        return False, None

    # Final 20% of trajectory
    final_start = int(len(trajectory) * 0.8)
    final_segment = trajectory[final_start:]

    if len(final_segment) < 2:
        return False, None

    # Compute max relative change in final segment
    max_rel_change = 0.0
    for i in range(1, len(final_segment)):
        prev = final_segment[i - 1]
        curr = final_segment[i]

        if abs(prev) < 1e-12:
            # Avoid division by zero; use absolute change
            rel_change = abs(curr - prev)
        else:
            rel_change = abs(curr - prev) / abs(prev)

        max_rel_change = max(max_rel_change, rel_change)

    converged = max_rel_change < threshold

    # Find the step where convergence begins (first step in final 20%)
    step_index = final_start if converged else None

    return converged, step_index


def run_convergence_analysis(
    train_loader: DataLoader,
    n_steps: int = 100,
    record_interval: int = 5,
    subsample_pairs: int = 5000,
    results_dir: str | Path | None = None,
) -> ConvergenceResult:
    """Run developmental embedding with quality tracking.

    Args:
        train_loader: Training data loader.
        n_steps: Number of position update steps.
        record_interval: Record quality every N steps.
        subsample_pairs: Number of pairs for force computation.
        results_dir: Directory to save outputs.

    Returns:
        ConvergenceResult with trajectory and convergence status.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create and partially train a model (needed for gradient correlations)
    import torch
    import torch.nn as nn

    device = get_device()
    model = BaselineMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Warmup: train for 2 epochs to get meaningful gradients
    print("Warming up model for developmental embedding...")
    for epoch in range(2):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Move model to CPU for embedding computation
    model = model.cpu()

    # Run developmental embedding with quality tracking
    print(f"Running developmental embedding ({n_steps} steps)...")
    dev_embedding = DevelopmentalEmbedding(
        n_steps=n_steps,
        position_lr=0.01,
        n_correlation_batches=5,
        record_interval=record_interval,
        subsample_pairs=subsample_pairs,
    )

    positions = dev_embedding.embed(model, data_loader=train_loader)
    quality_trajectory = dev_embedding.get_convergence_history()

    # Detect convergence
    converged, step_index = detect_convergence(quality_trajectory)
    final_quality = quality_trajectory[-1] if quality_trajectory else 0.0

    result = ConvergenceResult(
        quality_trajectory=quality_trajectory,
        converged=converged,
        final_quality=final_quality,
        n_steps_to_stability=step_index,
    )

    # Save trajectory to CSV
    csv_path = results_dir / "developmental_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "quality_score"])
        writer.writeheader()
        for i, score in enumerate(quality_trajectory):
            writer.writerow({
                "step": (i + 1) * record_interval,
                "quality_score": score,
            })

    # Generate trajectory plot
    history_for_plot = [
        ((i + 1) * record_interval, score)
        for i, score in enumerate(quality_trajectory)
    ]
    plot_developmental_trajectory(
        quality_history=history_for_plot,
        output_path=results_dir / "developmental_trajectory.png",
    )

    print(f"Convergence analysis: converged={converged}, "
          f"final_quality={final_quality:.4f}")
    return result
