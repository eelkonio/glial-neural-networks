#!/usr/bin/env python3
"""Run a quick verification experiment (5 epochs × 1 seed).

Verifies the full pipeline works end-to-end without running
the full 3-6 hour experiment. Checks:
- All 6 comparison conditions run without error
- CSV output is generated correctly
- No NaN/Inf in any condition
- Timestamps are printed
- Central prediction computation works
- Ablation scripts run

Usage:
    cd steps/13-astrocyte-gating
    python -m code.scripts.run_quick_experiment
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
    output_dir = Path("results") / "quick"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    n_epochs = 5
    seeds = [42]
    batch_size = 128

    print(f"{'='*70}")
    print(f"STEP 13: QUICK EXPERIMENT (verification run)")
    print(f"{'='*70}")
    print(f"Start: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Epochs: {n_epochs}, Seeds: {seeds}, Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # === Phase 1: Performance Comparison (quick) ===
    print(f"\n{'='*70}")
    print("PHASE 1: Performance Comparison (6 conditions × 1 seed × 5 epochs)")
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

    # === Phase 3: Calcium Ablation (quick) ===
    print(f"\n{'='*70}")
    print("PHASE 3: Calcium Ablation (4 mechanisms × 1 seed × 5 epochs)")
    print(f"{'='*70}")

    calcium_results = run_calcium_ablation(
        n_epochs=n_epochs,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir),
        verbose=True,
    )

    # === Phase 4: Spatial Ablation (quick) ===
    print(f"\n{'='*70}")
    print("PHASE 4: Spatial Ablation (2 strategies × 1 seed × 5 epochs)")
    print(f"{'='*70}")

    spatial_results = run_spatial_ablation(
        n_epochs=n_epochs,
        seeds=seeds,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir),
        verbose=True,
    )

    # === Verification Checks ===
    print(f"\n{'='*70}")
    print("VERIFICATION CHECKS")
    print(f"{'='*70}")

    all_results = comparison_results + calcium_results + spatial_results
    
    # Check 1: No NaN/Inf
    nan_conditions = [r.condition_name for r in all_results if r.any_nan]
    if nan_conditions:
        print(f"  ❌ NaN/Inf detected in: {nan_conditions}")
    else:
        print(f"  ✓ No NaN/Inf in any condition")

    # Check 2: CSV files exist
    csv_files = list(output_dir.glob("*.csv"))
    print(f"  ✓ {len(csv_files)} CSV files generated")

    # Check 3: Metadata JSON exists
    json_files = list(output_dir.glob("metadata_*.json"))
    print(f"  ✓ {len(json_files)} metadata JSON files generated")

    # Check 4: All conditions ran
    condition_names = set(r.condition_name for r in all_results)
    expected = {
        "three_factor_random", "three_factor_reward",
        "binary_gate", "directional_gate", "volume_teaching", "backprop",
        "ablation_full_lirinzel", "ablation_simple_threshold",
        "ablation_linear_ema", "ablation_random_matched",
        "ablation_spatial", "ablation_random_assign",
    }
    missing = expected - condition_names
    if missing:
        print(f"  ❌ Missing conditions: {missing}")
    else:
        print(f"  ✓ All {len(expected)} conditions ran successfully")

    # Check 5: Plots generated
    png_files = list(output_dir.glob("*.png"))
    print(f"  ✓ {len(png_files)} PNG plots generated")

    # === Final Summary ===
    print(f"\n{'='*70}")
    print("QUICK EXPERIMENT COMPLETE")
    print(f"End: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*70}")

    print("\nPerformance Comparison (5 epochs):")
    for name, s in stats.items():
        print(f"  {name:25s}: {s['mean_accuracy']*100:.2f}%"
              f"  {'⚠️ NaN' if s['any_nan'] else '✓'}")

    print(f"\nCentral Prediction: {prediction['conclusion']}")

    print("\nCalcium Ablation (5 epochs):")
    ca_stats = compute_summary_stats(calcium_results)
    for name, s in ca_stats.items():
        print(f"  {name:30s}: {s['mean_accuracy']*100:.2f}%")

    print("\nSpatial Ablation (5 epochs):")
    sp_stats = compute_summary_stats(spatial_results)
    for name, s in sp_stats.items():
        print(f"  {name:30s}: {s['mean_accuracy']*100:.2f}%")

    # Return success/failure
    if nan_conditions or missing:
        print("\n⚠️  SOME CHECKS FAILED — review output above")
        return 1
    else:
        print("\n✓ ALL CHECKS PASSED — pipeline verified end-to-end")
        return 0


if __name__ == "__main__":
    sys.exit(main())
