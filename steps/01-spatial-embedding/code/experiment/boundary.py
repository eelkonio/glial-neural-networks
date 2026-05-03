"""Boundary condition test and three-point validation.

Tests whether embedding quality predicts performance improvement:
- Pearson correlation between quality scores and performance deltas
- Three-point validation: adversarial < random < best
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from code.experiment.runner import ComparisonResult
from code.visualization.plots import (
    plot_boundary_regression,
    plot_three_point_curve,
)


@dataclass
class BoundaryResult:
    """Result of the boundary condition test."""

    correlation_coefficient: float
    p_value: float
    embedding_scores: dict[str, float]
    performance_deltas: dict[str, float]


@dataclass
class ThreePointValidation:
    """Result of the three-point validation curve."""

    adversarial_delta: float
    random_delta: float
    best_delta: float
    monotonic: bool


def run_boundary_condition(
    comparison_results: list[ComparisonResult],
    results_dir: str | Path | None = None,
) -> BoundaryResult:
    """Run boundary condition test on comparison results.

    Computes Pearson correlation between quality scores and performance
    deltas across all embedding methods.

    Args:
        comparison_results: Results from run_comparison().
        results_dir: Directory to save outputs.

    Returns:
        BoundaryResult with correlation and per-method data.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find baseline accuracy
    baseline_acc = None
    for r in comparison_results:
        if r.condition_name == "uncoupled_baseline":
            baseline_acc = r.mean_accuracy
            break

    if baseline_acc is None:
        # Use the minimum accuracy as baseline
        baseline_acc = min(r.mean_accuracy for r in comparison_results)

    # Collect quality scores and performance deltas
    embedding_scores: dict[str, float] = {}
    performance_deltas: dict[str, float] = {}

    for r in comparison_results:
        if r.condition_name == "uncoupled_baseline":
            continue
        embedding_scores[r.condition_name] = r.mean_quality_score
        performance_deltas[r.condition_name] = r.mean_accuracy - baseline_acc

    # Compute Pearson correlation
    quality_arr = np.array(list(embedding_scores.values()))
    delta_arr = np.array(list(performance_deltas.values()))

    if len(quality_arr) < 3 or np.std(quality_arr) < 1e-12 or np.std(delta_arr) < 1e-12:
        corr_coef = 0.0
        p_value = 1.0
    else:
        corr_coef, p_value = stats.pearsonr(quality_arr, delta_arr)

    result = BoundaryResult(
        correlation_coefficient=float(corr_coef),
        p_value=float(p_value),
        embedding_scores=embedding_scores,
        performance_deltas=performance_deltas,
    )

    # Save to CSV
    csv_path = results_dir / "boundary_condition.csv"
    _save_boundary_csv(result, csv_path)

    # Generate plot
    plot_boundary_regression(
        quality_scores=quality_arr.tolist(),
        performance_deltas=delta_arr.tolist(),
        output_path=results_dir / "boundary_regression.png",
    )

    print(f"Boundary condition: r={corr_coef:.4f}, p={p_value:.4f}")
    return result


def run_three_point_validation(
    comparison_results: list[ComparisonResult],
    results_dir: str | Path | None = None,
) -> ThreePointValidation:
    """Run three-point validation from comparison results.

    Extracts adversarial, random, and best embedding deltas and verifies
    monotonicity: adversarial_delta < random_delta < best_delta.

    Args:
        comparison_results: Results from run_comparison().
        results_dir: Directory to save outputs.

    Returns:
        ThreePointValidation with deltas and monotonicity check.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find baseline accuracy
    baseline_acc = None
    for r in comparison_results:
        if r.condition_name == "uncoupled_baseline":
            baseline_acc = r.mean_accuracy
            break

    if baseline_acc is None:
        baseline_acc = min(r.mean_accuracy for r in comparison_results)

    # Extract deltas for the three key conditions
    adversarial_delta = 0.0
    random_delta = 0.0
    best_delta = -float("inf")

    for r in comparison_results:
        delta = r.mean_accuracy - baseline_acc

        if r.condition_name == "adversarial_coupled":
            adversarial_delta = delta
        elif r.condition_name == "random_coupled":
            random_delta = delta
        elif r.condition_name not in ("uncoupled_baseline", "adversarial_coupled", "random_coupled"):
            # Track the best non-adversarial, non-random, non-baseline condition
            if delta > best_delta:
                best_delta = delta

    if best_delta == -float("inf"):
        best_delta = 0.0

    monotonic = adversarial_delta < random_delta < best_delta

    result = ThreePointValidation(
        adversarial_delta=adversarial_delta,
        random_delta=random_delta,
        best_delta=best_delta,
        monotonic=monotonic,
    )

    # Save to CSV
    csv_path = results_dir / "three_point_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["adversarial_delta", "random_delta", "best_delta", "monotonic"]
        )
        writer.writeheader()
        writer.writerow({
            "adversarial_delta": adversarial_delta,
            "random_delta": random_delta,
            "best_delta": best_delta,
            "monotonic": monotonic,
        })

    # Generate plot
    plot_three_point_curve(
        adversarial_delta=adversarial_delta,
        random_delta=random_delta,
        best_delta=best_delta,
        output_path=results_dir / "three_point_curve.png",
    )

    print(f"Three-point validation: adversarial={adversarial_delta:.4f}, "
          f"random={random_delta:.4f}, best={best_delta:.4f}, "
          f"monotonic={monotonic}")
    return result


def _save_boundary_csv(result: BoundaryResult, output_path: Path) -> None:
    """Save boundary condition results to CSV."""
    fieldnames = ["method", "quality_score", "performance_delta"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in result.embedding_scores:
            writer.writerow({
                "method": method,
                "quality_score": result.embedding_scores[method],
                "performance_delta": result.performance_deltas[method],
            })
