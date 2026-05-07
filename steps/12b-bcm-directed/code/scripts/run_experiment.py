#!/usr/bin/env python3
"""Run the full Step 12b experiment.

50 epochs × 3 seeds × 5 conditions with timestamp logging.

Expected runtime: ~2 hours total on CPU.

Usage:
    cd steps/12b-bcm-directed
    python -m code.scripts.run_experiment
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

_step12b_dir = str(Path(__file__).parent.parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

from code.experiment import get_all_conditions, run_condition


def main():
    print("=" * 70)
    print("STEP 12b: BCM-DIRECTED FULL EXPERIMENT")
    print("=" * 70)

    start_time = time.time()
    start_timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Start: {start_timestamp}")

    # Configuration
    n_epochs = 50
    seeds = [42, 123, 456]
    batch_size = 128
    device = "cpu"

    conditions = get_all_conditions()

    print(f"Conditions: {[c.name for c in conditions]}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {n_epochs}")
    print(f"Device: {device}")
    print(f"Expected runtime: ~2 hours total")
    print("=" * 70)

    all_results = []

    for condition in conditions:
        print(f"\n{'─' * 70}")
        print(f"CONDITION: {condition.name}")
        print(f"{'─' * 70}")

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            condition_start = time.time()

            result = run_condition(
                condition=condition,
                n_epochs=n_epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                verbose=True,
            )

            condition_duration = time.time() - condition_start
            result["duration_seconds"] = condition_duration
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            all_results.append(result)

            print(f"  → Final accuracy: {result['final_accuracy']:.4f} "
                  f"({condition_duration:.1f}s)")

    end_time = time.time()
    end_timestamp = datetime.now(timezone.utc).isoformat()
    total_duration = end_time - start_time

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"End: {end_timestamp}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    # Summary
    print(f"\n{'─' * 70}")
    print("SUMMARY")
    print(f"{'─' * 70}")
    for condition in conditions:
        cond_results = [r for r in all_results if r["condition"] == condition.name]
        accs = [r["final_accuracy"] for r in cond_results]
        mean_acc = sum(accs) / len(accs) if accs else 0
        print(f"  {condition.name:25s}: {mean_acc*100:.2f}% "
              f"(seeds: {[f'{a*100:.1f}%' for a in accs]})")

    # Save results
    output = {
        "experiment": "bcm_directed_full",
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": total_duration,
        "n_epochs": n_epochs,
        "seeds": seeds,
        "batch_size": batch_size,
        "device": device,
        "results": all_results,
    }

    results_dir = Path(_step12b_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "full_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
