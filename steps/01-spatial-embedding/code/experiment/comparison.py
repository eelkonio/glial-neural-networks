"""Full comparison experiment across all embedding strategies.

Defines all 10 experimental conditions, runs them on both MNIST and
TopographicTask with multiple seeds, and saves results to CSV + plots.
"""

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from code.data import get_mnist_loaders
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
from code.experiment.runner import (
    ComparisonResult,
    ConditionResult,
    CouplingConfig,
    ExperimentRunner,
)
from code.model import BaselineMLP
from code.topographic_task import TopographicTask
from code.visualization.plots import plot_quality_vs_performance


# Default coupling configuration
DEFAULT_COUPLING = CouplingConfig(k=10, alpha=0.5)

# Seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456]


def _model_factory() -> BaselineMLP:
    """Create a fresh BaselineMLP instance."""
    return BaselineMLP()


def get_conditions(
    train_loader: DataLoader,
) -> list[tuple[str, Callable[[], BaselineMLP], object | None, CouplingConfig | None]]:
    """Define all 10 experimental conditions.

    Data-dependent embeddings use reduced parameters for tractability:
    - correlation/adversarial: n_batches=5, subsample_size=500
    - developmental: n_steps=100, subsample_pairs=5000

    Args:
        train_loader: Training data loader (needed for data-dependent embeddings).

    Returns:
        List of (name, model_factory, embedding, coupling_config) tuples.
    """
    # 1. Uncoupled baseline (Adam only, no embedding)
    conditions = [
        ("uncoupled_baseline", _model_factory, None, None),
    ]

    # 2. Linear + coupling
    conditions.append(
        ("linear_coupled", _model_factory, LinearEmbedding(), DEFAULT_COUPLING)
    )

    # 3. Random + coupling
    conditions.append(
        ("random_coupled", _model_factory, RandomEmbedding(seed=42), DEFAULT_COUPLING)
    )

    # 4. Spectral + coupling
    conditions.append(
        ("spectral_coupled", _model_factory, SpectralEmbedding(), DEFAULT_COUPLING)
    )

    # 5. LayeredClustered + coupling
    conditions.append(
        (
            "layered_clustered_coupled",
            _model_factory,
            LayeredClusteredEmbedding(),
            DEFAULT_COUPLING,
        )
    )

    # 6. Correlation + coupling (reduced params)
    conditions.append(
        (
            "correlation_coupled",
            _model_factory,
            CorrelationEmbedding(n_batches=5, subsample_size=500),
            DEFAULT_COUPLING,
        )
    )

    # 7. Developmental + coupling (reduced params)
    conditions.append(
        (
            "developmental_coupled",
            _model_factory,
            DevelopmentalEmbedding(
                n_steps=100, subsample_pairs=5000, n_correlation_batches=5
            ),
            DEFAULT_COUPLING,
        )
    )

    # 8. Adversarial + coupling (reduced params)
    conditions.append(
        (
            "adversarial_coupled",
            _model_factory,
            AdversarialEmbedding(n_correlation_batches=5, subsample_size=500),
            DEFAULT_COUPLING,
        )
    )

    # 9. Differentiable (jointly trained) — no coupling, positions are learned
    conditions.append(
        (
            "differentiable",
            _model_factory,
            DifferentiableEmbedding(lambda_spatial=0.01, subsample_pairs=5000),
            None,
        )
    )

    # 10. Differentiable + coupling (optional)
    conditions.append(
        (
            "differentiable_coupled",
            _model_factory,
            DifferentiableEmbedding(lambda_spatial=0.01, subsample_pairs=5000),
            DEFAULT_COUPLING,
        )
    )

    return conditions


def save_comparison_results(
    results: list[ComparisonResult],
    output_path: Path,
    task_name: str = "mnist",
) -> Path:
    """Save comparison results to CSV.

    Args:
        results: List of ComparisonResult from run_comparison.
        output_path: Path to save CSV file.
        task_name: Name of the task (for the task column).

    Returns:
        Path to the saved CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "condition",
        "embedding_method",
        "coupling_enabled",
        "seed",
        "final_test_accuracy",
        "steps_to_95pct",
        "quality_score",
        "coherence_score",
        "wall_clock_seconds",
    ]

    rows = []
    for comp_result in results:
        for ind_result in comp_result.individual_results:
            rows.append(
                {
                    "task": task_name,
                    "condition": comp_result.condition_name,
                    "embedding_method": comp_result.embedding_method,
                    "coupling_enabled": comp_result.coupling_enabled,
                    "seed": ind_result.seed,
                    "final_test_accuracy": ind_result.final_test_accuracy,
                    "steps_to_95pct": ind_result.steps_to_95pct,
                    "quality_score": ind_result.quality_score,
                    "coherence_score": ind_result.coherence_score,
                    "wall_clock_seconds": ind_result.wall_clock_seconds,
                }
            )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def run_full_comparison(
    n_epochs: int = 10,
    seeds: list[int] | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, list[ComparisonResult]]:
    """Run the full comparison experiment on both MNIST and TopographicTask.

    Args:
        n_epochs: Number of training epochs per condition.
        seeds: Random seeds to use. Defaults to [42, 123, 456].
        results_dir: Directory to save results. Defaults to results/.

    Returns:
        Dict mapping task name to list of ComparisonResult.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[ComparisonResult]] = {}

    # --- MNIST ---
    print("=" * 60)
    print("Running comparison on MNIST")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    runner = ExperimentRunner(train_loader, test_loader, results_dir=results_dir)
    runner.log_metadata("comparison_mnist", {
        "n_epochs": n_epochs,
        "seeds": seeds,
        "task": "mnist",
    })

    conditions = get_conditions(train_loader)
    mnist_results = runner.run_comparison(
        conditions=conditions,
        n_seeds=len(seeds),
        seeds=seeds,
        n_epochs=n_epochs,
    )
    all_results["mnist"] = mnist_results

    # Save MNIST results
    save_comparison_results(
        mnist_results,
        results_dir / "comparison_results.csv",
        task_name="mnist",
    )

    # --- TopographicTask ---
    print("=" * 60)
    print("Running comparison on TopographicTask")
    print("=" * 60)

    topo_task = TopographicTask(n_train=10000, n_test=2000)
    topo_train, topo_test = topo_task.generate_dataset(seed=42, batch_size=128)
    topo_runner = ExperimentRunner(topo_train, topo_test, results_dir=results_dir)
    topo_runner.log_metadata("comparison_topographic", {
        "n_epochs": n_epochs,
        "seeds": seeds,
        "task": "topographic",
    })

    topo_conditions = get_conditions(topo_train)
    topo_results = topo_runner.run_comparison(
        conditions=topo_conditions,
        n_seeds=len(seeds),
        seeds=seeds,
        n_epochs=n_epochs,
    )
    all_results["topographic"] = topo_results

    # Append topographic results to same CSV
    _append_comparison_results(
        topo_results,
        results_dir / "comparison_results.csv",
        task_name="topographic",
    )

    # --- Generate visualization ---
    _generate_comparison_plot(mnist_results, results_dir)

    print("=" * 60)
    print(f"Results saved to {results_dir}")
    print("=" * 60)

    return all_results


def _append_comparison_results(
    results: list[ComparisonResult],
    output_path: Path,
    task_name: str,
) -> None:
    """Append results to an existing CSV file."""
    output_path = Path(output_path)

    rows = []
    for comp_result in results:
        for ind_result in comp_result.individual_results:
            rows.append(
                {
                    "task": task_name,
                    "condition": comp_result.condition_name,
                    "embedding_method": comp_result.embedding_method,
                    "coupling_enabled": comp_result.coupling_enabled,
                    "seed": ind_result.seed,
                    "final_test_accuracy": ind_result.final_test_accuracy,
                    "steps_to_95pct": ind_result.steps_to_95pct,
                    "quality_score": ind_result.quality_score,
                    "coherence_score": ind_result.coherence_score,
                    "wall_clock_seconds": ind_result.wall_clock_seconds,
                }
            )

    fieldnames = [
        "task", "condition", "embedding_method", "coupling_enabled",
        "seed", "final_test_accuracy", "steps_to_95pct",
        "quality_score", "coherence_score", "wall_clock_seconds",
    ]

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def _generate_comparison_plot(
    results: list[ComparisonResult],
    results_dir: Path,
) -> None:
    """Generate the quality vs performance scatter plot."""
    # Find baseline accuracy for computing deltas
    baseline_acc = None
    for r in results:
        if r.condition_name == "uncoupled_baseline":
            baseline_acc = r.mean_accuracy
            break

    if baseline_acc is None:
        baseline_acc = 0.0

    quality_scores = []
    performance_deltas = []
    labels = []

    for r in results:
        quality_scores.append(r.mean_quality_score)
        performance_deltas.append(r.mean_accuracy - baseline_acc)
        labels.append(r.condition_name)

    plot_quality_vs_performance(
        quality_scores=quality_scores,
        performance_deltas=performance_deltas,
        labels=labels,
        output_path=results_dir / "embedding_vs_performance.png",
    )


if __name__ == "__main__":
    run_full_comparison()
