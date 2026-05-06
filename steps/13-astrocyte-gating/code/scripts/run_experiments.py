#!/usr/bin/env python3
"""CLI entry point for Step 13 experiments.

Usage:
    python -m code.scripts.run_experiments --condition binary_gate --epochs 50 --seeds 42 123 456
    python -m code.scripts.run_experiments --all --epochs 50
    python -m code.scripts.run_experiments --condition backprop --epochs 5 --seeds 42

Flags:
    --condition NAME    Run a single condition by name
    --all               Run all six conditions
    --epochs N          Number of training epochs (default: 50)
    --seeds S [S ...]   Random seeds (default: 42 123 456)
    --batch-size N      Batch size (default: 128)
    --device DEVICE     Torch device (default: cpu)
    --output-dir DIR    Output directory (default: results)
    --verbose           Print per-epoch progress
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure step 13 code is importable
step13_dir = str(Path(__file__).parent.parent.parent)
if step13_dir not in sys.path:
    sys.path.insert(0, step13_dir)

from code.experiment.conditions import get_all_conditions, get_condition_by_name
from code.experiment.runner import ExperimentRunner
from code.experiment.comparison import (
    compute_summary_stats,
    save_summary_csv,
    generate_accuracy_bar_chart,
    generate_convergence_curves,
)


def main():
    parser = argparse.ArgumentParser(description="Step 13 Experiment Runner")
    parser.add_argument("--condition", type=str, help="Run a single condition by name")
    parser.add_argument("--all", action="store_true", help="Run all conditions")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print progress")
    parser.add_argument("--no-verbose", action="store_false", dest="verbose")

    args = parser.parse_args()

    if not args.condition and not args.all:
        parser.error("Must specify --condition NAME or --all")

    # Select conditions
    if args.all:
        conditions = get_all_conditions()
    else:
        conditions = [get_condition_by_name(args.condition)]

    print(f"{'='*60}")
    print(f"Step 13: Astrocyte Gating Experiments")
    print(f"{'='*60}")
    print(f"Conditions: {[c.name for c in conditions]}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")

    # Run experiments
    runner = ExperimentRunner(
        conditions=conditions,
        seeds=args.seeds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    results = runner.run_all()

    # Generate summary
    stats = compute_summary_stats(results)
    save_summary_csv(stats, args.output_dir)

    # Generate visualizations
    try:
        generate_accuracy_bar_chart(stats, args.output_dir)
        generate_convergence_curves(results, args.output_dir)
        print("\nVisualizations saved to output directory.")
    except ImportError:
        print("\nMatplotlib not available — skipping visualizations.")

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, s in stats.items():
        print(f"  {name:25s}: {s['mean_accuracy']*100:.2f}% ± {s['std_accuracy']*100:.2f}%"
              f"  (best: {s['best_accuracy']*100:.2f}%)"
              f"  {'⚠️ NaN' if s['any_nan'] else '✓'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
