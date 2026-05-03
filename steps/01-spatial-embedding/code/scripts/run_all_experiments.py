"""Main runner script that orchestrates all experiments in sequence.

Usage:
    .venv/bin/python steps/01-spatial-embedding/code/scripts/run_all_experiments.py

This script runs:
1. Full comparison experiment (10 conditions × 2 tasks × 3 seeds)
2. Boundary condition test (quality → performance correlation)
3. Developmental convergence analysis
4. Three-point validation (adversarial → random → best)
5. Temporal quality tracking
6. Spatial coherence test

Results are saved to steps/01-spatial-embedding/results/ and a summary
markdown file is generated.
"""

import sys
import time
from pathlib import Path

# Add the step directory to path so 'code' package is importable
step_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(step_dir))

from code.data import get_mnist_loaders
from code.experiment.boundary import run_boundary_condition, run_three_point_validation
from code.experiment.comparison import run_full_comparison
from code.experiment.convergence import run_convergence_analysis
from code.experiment.spatial_coherence_test import run_spatial_coherence_test
from code.experiment.temporal import run_temporal_quality_tracking


def generate_summary(
    results_dir: Path,
    comparison_results: dict,
    boundary_result,
    three_point_result,
    convergence_result,
    temporal_results: list,
    coherence_result,
    total_time: float,
) -> Path:
    """Generate results/summary.md with key findings.

    Args:
        results_dir: Directory where results are saved.
        comparison_results: Dict of task → ComparisonResult list.
        boundary_result: BoundaryResult from boundary condition test.
        three_point_result: ThreePointValidation result.
        convergence_result: ConvergenceResult from developmental analysis.
        temporal_results: List of TemporalQualityResult.
        coherence_result: SpatialCoherenceResult.
        total_time: Total experiment wall-clock time in seconds.

    Returns:
        Path to the generated summary file.
    """
    summary_path = results_dir / "summary.md"

    lines = [
        "# Spatial Embedding Experiment Results Summary",
        "",
        f"**Total experiment time**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
        "",
        "## 1. Comparison Experiment",
        "",
        "### MNIST Results",
        "",
        "| Condition | Mean Accuracy | Std | Quality Score |",
        "|-----------|--------------|-----|---------------|",
    ]

    if "mnist" in comparison_results:
        for r in comparison_results["mnist"]:
            lines.append(
                f"| {r.condition_name} | {r.mean_accuracy:.4f} | "
                f"{r.std_accuracy:.4f} | {r.mean_quality_score:.4f} |"
            )

    lines.extend([
        "",
        "### TopographicTask Results",
        "",
        "| Condition | Mean Accuracy | Std | Quality Score |",
        "|-----------|--------------|-----|---------------|",
    ])

    if "topographic" in comparison_results:
        for r in comparison_results["topographic"]:
            lines.append(
                f"| {r.condition_name} | {r.mean_accuracy:.4f} | "
                f"{r.std_accuracy:.4f} | {r.mean_quality_score:.4f} |"
            )

    lines.extend([
        "",
        "## 2. Boundary Condition Test",
        "",
        f"- **Pearson correlation (quality → performance)**: "
        f"r = {boundary_result.correlation_coefficient:.4f}, "
        f"p = {boundary_result.p_value:.4f}",
        "",
    ])

    lines.extend([
        "## 3. Three-Point Validation",
        "",
        f"- **Adversarial delta**: {three_point_result.adversarial_delta:.4f}",
        f"- **Random delta**: {three_point_result.random_delta:.4f}",
        f"- **Best delta**: {three_point_result.best_delta:.4f}",
        f"- **Monotonic (adversarial < random < best)**: {three_point_result.monotonic}",
        "",
    ])

    lines.extend([
        "## 4. Developmental Convergence",
        "",
        f"- **Converged**: {convergence_result.converged}",
        f"- **Final quality**: {convergence_result.final_quality:.4f}",
        f"- **Steps to stability**: {convergence_result.n_steps_to_stability}",
        "",
    ])

    lines.extend([
        "## 5. Temporal Quality Tracking",
        "",
        "| Method | Initial Quality | Final Quality | Degraded |",
        "|--------|----------------|---------------|----------|",
    ])

    for tr in temporal_results:
        lines.append(
            f"| {tr.method_name} | {tr.initial_quality:.4f} | "
            f"{tr.final_quality:.4f} | {tr.degraded} |"
        )

    lines.extend([
        "",
        "## 6. Spatial Coherence Test",
        "",
        f"- **Coupled coherence**: {coherence_result.coupled_coherence:.4f}",
        f"- **Uncoupled coherence**: {coherence_result.uncoupled_coherence:.4f}",
        f"- **Mechanism confirmed (coupled > uncoupled)**: "
        f"{coherence_result.mechanism_confirmed}",
        "",
    ])

    lines.extend([
        "## Key Findings",
        "",
        "1. **Quality-performance correlation**: "
        f"{'Positive' if boundary_result.correlation_coefficient > 0 else 'Negative'} "
        f"correlation (r={boundary_result.correlation_coefficient:.3f}) between "
        "embedding quality and performance improvement.",
        "",
        "2. **Three-point validation**: "
        f"{'Confirmed' if three_point_result.monotonic else 'Not confirmed'} — "
        "spatial structure matters directionally, not just as regularization."
        if three_point_result.monotonic else
        "the expected ordering was not observed.",
        "",
        "3. **Developmental convergence**: "
        f"{'Quality stabilizes' if convergence_result.converged else 'Quality does not stabilize'} "
        "during position optimization.",
        "",
        "4. **Temporal stability**: "
        f"{sum(1 for t in temporal_results if t.degraded)} of "
        f"{len(temporal_results)} fixed embeddings show quality degradation "
        "during training.",
        "",
        "5. **Spatial coherence mechanism**: "
        f"{'Confirmed' if coherence_result.mechanism_confirmed else 'Not confirmed'} — "
        "coupling produces more spatially organized weight structure."
        if coherence_result.mechanism_confirmed else
        "coupling does not produce more organized structure.",
        "",
        "## Implications for Step 02",
        "",
        "- The best embedding strategy for downstream use should be selected "
        "based on the quality-performance correlation results above.",
        "- If temporal degradation is observed, consider periodic recomputation "
        "or the differentiable embedding approach.",
        "- The spatial coherence result informs whether the coupling mechanism "
        "is suitable for modulation field experiments in Step 02.",
        "",
        "## Output Files",
        "",
        "- `comparison_results.csv` — Full comparison data",
        "- `boundary_condition.csv` — Quality vs performance data",
        "- `three_point_validation.csv` — Three-point validation data",
        "- `developmental_convergence.csv` — Convergence trajectory",
        "- `temporal_quality.csv` — Quality over training time",
        "- `spatial_coherence.csv` — Coherence comparison",
        "- `embedding_vs_performance.png` — Quality vs performance scatter",
        "- `boundary_regression.png` — Regression plot",
        "- `three_point_curve.png` — Three-point validation curve",
        "- `developmental_trajectory.png` — Convergence trajectory plot",
        "- `temporal_quality_trajectories.png` — Temporal quality plot",
        "- `spatial_coherence_comparison.png` — Coherence comparison plot",
        "",
    ])

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    return summary_path


def main():
    """Run all experiments in sequence."""
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("=" * 70)
    print("SPATIAL EMBEDDING EXPERIMENTS — FULL RUN")
    print("=" * 70)

    # 1. Full comparison experiment
    print("\n[1/6] Running full comparison experiment...")
    comparison_results = run_full_comparison(
        n_epochs=10,
        seeds=[42, 123, 456],
        results_dir=results_dir,
    )

    # 2. Boundary condition test
    print("\n[2/6] Running boundary condition test...")
    mnist_results = comparison_results.get("mnist", [])
    boundary_result = run_boundary_condition(mnist_results, results_dir=results_dir)

    # 3. Three-point validation
    print("\n[3/6] Running three-point validation...")
    three_point_result = run_three_point_validation(mnist_results, results_dir=results_dir)

    # 4. Developmental convergence analysis
    print("\n[4/6] Running developmental convergence analysis...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    convergence_result = run_convergence_analysis(
        train_loader=train_loader,
        n_steps=100,
        record_interval=5,
        subsample_pairs=5000,
        results_dir=results_dir,
    )

    # 5. Temporal quality tracking
    print("\n[5/6] Running temporal quality tracking...")
    temporal_results = run_temporal_quality_tracking(
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=10,
        record_interval=2,
        seed=42,
        results_dir=results_dir,
    )

    # 6. Spatial coherence test
    print("\n[6/6] Running spatial coherence test...")
    coherence_result = run_spatial_coherence_test(
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=10,
        seed=42,
        results_dir=results_dir,
    )

    total_time = time.time() - start_time

    # Generate summary
    print("\nGenerating summary...")
    summary_path = generate_summary(
        results_dir=results_dir,
        comparison_results=comparison_results,
        boundary_result=boundary_result,
        three_point_result=three_point_result,
        convergence_result=convergence_result,
        temporal_results=temporal_results,
        coherence_result=coherence_result,
        total_time=total_time,
    )

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Summary: {summary_path}")
    print(f"Results directory: {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
