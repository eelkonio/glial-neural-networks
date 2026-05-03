"""Visualization functions for spatial embedding experiments.

All plots are saved to a given output path as PNG files.
Uses matplotlib with a clean style for publication-quality figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Use a clean plot style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_quality_vs_performance(
    quality_scores: list[float] | np.ndarray,
    performance_deltas: list[float] | np.ndarray,
    labels: list[str] | None = None,
    output_path: str | Path = "results/quality_vs_performance.png",
) -> Path:
    """Scatter plot of embedding quality score vs performance delta.

    Args:
        quality_scores: Quality scores for each embedding method.
        performance_deltas: Performance deltas (accuracy improvement over baseline).
        labels: Optional labels for each point.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    quality_scores = np.asarray(quality_scores)
    performance_deltas = np.asarray(performance_deltas)

    ax.scatter(quality_scores, performance_deltas, s=80, alpha=0.7, edgecolors="k", linewidths=0.5)

    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (quality_scores[i], performance_deltas[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("Embedding Quality Score", fontsize=12)
    ax.set_ylabel("Performance Delta (vs. baseline)", fontsize=12)
    ax.set_title("Embedding Quality vs Performance Improvement", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_boundary_regression(
    quality_scores: list[float] | np.ndarray,
    performance_deltas: list[float] | np.ndarray,
    output_path: str | Path = "results/boundary_regression.png",
) -> Path:
    """Scatter plot with fitted regression line, annotated with r and p-value.

    Args:
        quality_scores: Quality scores for each embedding method.
        performance_deltas: Performance deltas.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    quality_scores = np.asarray(quality_scores, dtype=np.float64)
    performance_deltas = np.asarray(performance_deltas, dtype=np.float64)

    # Scatter plot
    ax.scatter(quality_scores, performance_deltas, s=80, alpha=0.7, edgecolors="k", linewidths=0.5)

    # Fit regression line
    if len(quality_scores) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            quality_scores, performance_deltas
        )

        # Plot regression line
        x_line = np.linspace(quality_scores.min(), quality_scores.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r-", linewidth=2, label="Regression line")

        # Annotate with r and p-value
        ax.annotate(
            f"r = {r_value:.3f}\np = {p_value:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    ax.set_xlabel("Embedding Quality Score", fontsize=12)
    ax.set_ylabel("Performance Delta", fontsize=12)
    ax.set_title("Boundary Condition: Quality → Performance Regression", fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_three_point_curve(
    adversarial_delta: float,
    random_delta: float,
    best_delta: float,
    output_path: str | Path = "results/three_point_curve.png",
) -> Path:
    """Bar chart showing the three-point validation (adversarial, random, best).

    Args:
        adversarial_delta: Performance delta for adversarial embedding.
        random_delta: Performance delta for random embedding.
        best_delta: Performance delta for best embedding.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Adversarial", "Random", "Best"]
    deltas = [adversarial_delta, random_delta, best_delta]
    colors = ["#d62728", "#7f7f7f", "#2ca02c"]

    bars = ax.bar(categories, deltas, color=colors, edgecolor="k", linewidth=0.5, width=0.6)

    # Add value labels on bars
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        y_pos = height + 0.001 if height >= 0 else height - 0.003
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{delta:.4f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
        )

    ax.set_ylabel("Performance Delta (vs. uncoupled baseline)", fontsize=12)
    ax.set_title("Three-Point Validation: Adversarial → Random → Best", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_developmental_trajectory(
    quality_history: list[tuple[int, float]] | np.ndarray,
    output_path: str | Path = "results/developmental_trajectory.png",
) -> Path:
    """Line plot of quality score over position update steps.

    Args:
        quality_history: List of (step, quality_score) tuples or (N, 2) array.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    quality_history = np.asarray(quality_history)
    if quality_history.ndim == 2:
        steps = quality_history[:, 0]
        scores = quality_history[:, 1]
    else:
        steps = np.arange(len(quality_history))
        scores = quality_history

    ax.plot(steps, scores, "b-o", markersize=4, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Position Update Steps", fontsize=12)
    ax.set_ylabel("Quality Score", fontsize=12)
    ax.set_title("Developmental Embedding: Quality Convergence", fontsize=14)

    # Add convergence region shading (final 20%)
    if len(steps) > 5:
        final_start = int(len(steps) * 0.8)
        ax.axvspan(
            steps[final_start],
            steps[-1],
            alpha=0.1,
            color="green",
            label="Convergence region (final 20%)",
        )
        ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_temporal_quality(
    trajectories_dict: dict[str, list[tuple[int, float]]],
    output_path: str | Path = "results/temporal_quality_trajectories.png",
) -> Path:
    """Multiple lines showing quality over training time per embedding method.

    Args:
        trajectories_dict: Mapping of method name to list of (epoch, quality_score).
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, trajectory in trajectories_dict.items():
        trajectory = np.asarray(trajectory)
        if trajectory.ndim == 2:
            epochs = trajectory[:, 0]
            scores = trajectory[:, 1]
        else:
            epochs = np.arange(len(trajectory))
            scores = trajectory

        ax.plot(epochs, scores, "-o", markersize=3, linewidth=1.5, label=method_name, alpha=0.8)

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Quality Score", fontsize=12)
    ax.set_title("Temporal Quality Tracking by Embedding Method", fontsize=14)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_spatial_coherence_comparison(
    coupled_score: float,
    uncoupled_score: float,
    output_path: str | Path = "results/spatial_coherence_comparison.png",
) -> Path:
    """Bar chart comparing spatial coherence for coupled vs uncoupled training.

    Args:
        coupled_score: Spatial coherence score with coupling enabled.
        uncoupled_score: Spatial coherence score without coupling.
        output_path: Path to save the plot.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    categories = ["Uncoupled", "Coupled"]
    scores = [uncoupled_score, coupled_score]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax.bar(categories, scores, color=colors, edgecolor="k", linewidth=0.5, width=0.5)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        y_pos = height + 0.005 if height >= 0 else height - 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{score:.4f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=11,
        )

    ax.set_ylabel("Spatial Coherence Score", fontsize=12)
    ax.set_title("Spatial Coherence: Coupled vs Uncoupled Training", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
