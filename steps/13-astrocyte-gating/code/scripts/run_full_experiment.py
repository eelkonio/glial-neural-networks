#!/usr/bin/env python3
"""Run the full Step 13 experiment suite.

This runs ALL conditions with full parameters:
- Performance comparison: 6 conditions × 3 seeds × 50 epochs
- Central prediction test
- Calcium ablation: 4 mechanisms × 3 seeds × 50 epochs
- Spatial ablation: 2 strategies × 3 seeds × 50 epochs

Expected runtime: 3-6 hours on M4 Pro (MPS).

Usage:
    cd steps/13-astrocyte-gating
    python -m code.scripts.run_full_experiment
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

# Ensure step 13 code is importable
step13_dir = str(Path(__file__).parent.parent.parent)
if step13_dir not in sys.path:
    sys.path.insert(0, step13_dir)

from code.experiment.conditions import get_all_conditions
from code.experiment.runner import ExperimentRunner
from code.experiment.comparison import (
    compute_summary_stats,
    save_summary_csv,
    generate_accuracy_bar_chart,
    generate_convergence_curves,
)
from code.experiment.central_prediction import (
    compute_central_prediction,
    generate_central_prediction_chart,
)
from code.experiment.ablation_calcium import run_calcium_ablation
from code.experiment.ablation_spatial import run_spatial_ablation


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    n_epochs = 50
    seeds = [42, 123, 456]
    batch_size = 128

    print(f"{'='*70}")
    print(f"STEP 13: FULL EXPERIMENT SUITE")
    print(f"{'='*70}")
    print(f"Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Epochs: {n_epochs}, Seeds: {seeds}, Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # === Phase 1: Performance Comparison ===
    print(f"\n{'='*70}")
    print("PHASE 1: Performance Comparison (6 conditions × 3 seeds × 50 epochs)")
    print(f"{'='*70}")

    conditions = get_all_conditions()
    runner = ExperimentRunner(
        conditions=conditions,
        seeds=seeds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir),
        verbose=True,
    )
    comparison_results = runner.run_all()

    # Generate comparison outputs
    stats = compute_summary_stats(comparison_results)
    save_summary_csv(stats, str(output_dir))
    try:
        generate_accuracy_bar_chart(stats, str(output_dir))
        generate_convergence_curves(comparison_results, str(output_dir))
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # === Phase 2: Central Prediction Test ===
    print(f"\n{'='*70}")
    print("PHASE 2: Central Prediction Test")
    print(f"{'='*70}")

    prediction = compute_central_prediction(comparison_results)
    print(f"\nConclusion: {prediction['conclusion']}")
    print(f"Benefit under local rules: {prediction['benefit_local_percent']:.2f}%")
    print(f"Benefit under backprop: {prediction['benefit_backprop_percent']:.2f}%")

    try:
        generate_central_prediction_chart(prediction, str(output_dir))
    except Exception as e:
        print(f"Warning: Could not generate prediction chart: {e}")

    # Save prediction result
    with open(output_dir / "central_prediction_result.json", "w") as f:
        json.dump(prediction, f, indent=2, default=str)

    # === Phase 3: Calcium Ablation ===
    print(f"\n{'='*70}")
    print("PHASE 3: Calcium Dynamics Ablation (4 mechanisms × 3 seeds × 50 epochs)")
    print(f"{'='*70}")

    calcium_results = run_calcium_ablation(
        n_epochs=n_epochs,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir),
        verbose=True,
    )

    # === Phase 4: Spatial Ablation ===
    print(f"\n{'='*70}")
    print("PHASE 4: Spatial Domain Ablation (2 strategies × 3 seeds × 50 epochs)")
    print(f"{'='*70}")

    spatial_results = run_spatial_ablation(
        n_epochs=n_epochs,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir),
        verbose=True,
    )

    # === Final Summary ===
    print(f"\n{'='*70}")
    print("FULL EXPERIMENT COMPLETE")
    print(f"End: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")

    print("\nPerformance Comparison:")
    for name, s in stats.items():
        print(f"  {name:25s}: {s['mean_accuracy']*100:.2f}% ± {s['std_accuracy']*100:.2f}%")

    print(f"\nCentral Prediction: {prediction['conclusion']}")

    print("\nCalcium Ablation:")
    ca_stats = compute_summary_stats(calcium_results)
    for name, s in ca_stats.items():
        print(f"  {name:30s}: {s['mean_accuracy']*100:.2f}% ± {s['std_accuracy']*100:.2f}%")

    print("\nSpatial Ablation:")
    sp_stats = compute_summary_stats(spatial_results)
    for name, s in sp_stats.items():
        print(f"  {name:30s}: {s['mean_accuracy']*100:.2f}% ± {s['std_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
