"""Smoke test: runs 1 condition × 1 seed × 2 epochs to verify pipeline.

Usage:
    .venv/bin/python steps/01-spatial-embedding/code/scripts/smoke_test.py

This verifies the full experiment pipeline works end-to-end without
running the full comparison (which takes 30-60 minutes).
"""

import sys
import time
from pathlib import Path

# Add the step directory to path so 'code' package is importable
step_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(step_dir))

from code.data import get_mnist_loaders
from code.embeddings import LinearEmbedding, get_all_strategies
from code.experiment.boundary import run_boundary_condition, run_three_point_validation
from code.experiment.comparison import get_conditions, save_comparison_results
from code.experiment.convergence import detect_convergence, run_convergence_analysis
from code.experiment.runner import CouplingConfig, ExperimentRunner
from code.experiment.spatial_coherence_test import run_spatial_coherence_test
from code.experiment.temporal import run_temporal_quality_tracking
from code.model import BaselineMLP


def main():
    """Run a minimal smoke test of the full pipeline."""
    start_time = time.time()
    results_dir = Path(__file__).parent.parent.parent / "results" / "smoke_test"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SMOKE TEST: Verifying experiment pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/7] Loading MNIST data...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # Test get_all_strategies
    print("\n[2/7] Testing get_all_strategies()...")
    strategies = get_all_strategies()
    assert len(strategies) == 8, f"Expected 8 strategies, got {len(strategies)}"
    print(f"  Got {len(strategies)} strategies: {[s.name for s in strategies]}")

    # Run a minimal comparison (2 conditions, 1 seed, 2 epochs)
    print("\n[3/7] Running minimal comparison (2 conditions × 1 seed × 2 epochs)...")
    runner = ExperimentRunner(train_loader, test_loader, results_dir=results_dir)

    conditions = [
        ("uncoupled_baseline", BaselineMLP, None, None),
        ("linear_coupled", BaselineMLP, LinearEmbedding(), CouplingConfig(k=10, alpha=0.5)),
    ]

    results = runner.run_comparison(
        conditions=conditions,
        n_seeds=1,
        seeds=[42],
        n_epochs=2,
    )

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"  Baseline accuracy: {results[0].mean_accuracy:.4f}")
    print(f"  Linear+coupling accuracy: {results[1].mean_accuracy:.4f}")

    # Save results
    csv_path = save_comparison_results(results, results_dir / "comparison_results.csv")
    print(f"  Saved to: {csv_path}")

    # Test boundary condition
    print("\n[4/7] Testing boundary condition analysis...")
    boundary_result = run_boundary_condition(results, results_dir=results_dir)
    print(f"  Correlation: {boundary_result.correlation_coefficient:.4f}")

    # Test three-point validation (will have limited data)
    print("\n[5/7] Testing three-point validation...")
    three_point = run_three_point_validation(results, results_dir=results_dir)
    print(f"  Monotonic: {three_point.monotonic}")

    # Test convergence detection
    print("\n[6/7] Testing convergence detection...")
    # Use a synthetic trajectory for speed
    trajectory = [0.1, 0.15, 0.18, 0.19, 0.195, 0.198, 0.199, 0.199, 0.199, 0.199]
    converged, step = detect_convergence(trajectory)
    print(f"  Synthetic trajectory converged: {converged} (step={step})")

    # Test temporal quality tracking (minimal)
    print("\n[7/7] Testing temporal quality tracking (2 epochs)...")
    temporal_results = run_temporal_quality_tracking(
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=2,
        record_interval=1,
        seed=42,
        results_dir=results_dir,
    )
    print(f"  Tracked {len(temporal_results)} methods")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST PASSED in {elapsed:.1f}s")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
