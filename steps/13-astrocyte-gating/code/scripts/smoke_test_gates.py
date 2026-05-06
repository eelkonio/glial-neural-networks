"""Smoke test: Train ThreeFactorRule + each gate variant for 3 epochs.

Reports accuracy and gate statistics (fraction open).
Verifies no NaN/Inf.
Prints timestamps before/after each variant.

Usage:
    .venv/bin/python steps/13-astrocyte-gating/code/scripts/smoke_test_gates.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure step 13 code is importable
_step13_dir = str(Path(__file__).parent.parent.parent)
if _step13_dir not in sys.path:
    sys.path.insert(0, _step13_dir)

import torch

from code.experiment.config import GateConfig, ExperimentCondition
from code.experiment.training import train_with_gate
from code.calcium.config import CalciumConfig
from code.domains.config import DomainConfig


def main():
    print("=" * 60)
    print("SMOKE TEST: Gate Variants + ThreeFactorRule (3 epochs)")
    print("=" * 60)
    print(f"Start: {datetime.now(timezone.utc).isoformat()}")
    print()

    device = "cpu"

    # Use lower threshold so gates actually open
    calcium_config = CalciumConfig(d_serine_threshold=0.02)
    domain_config = DomainConfig(domain_size=16, mode="contiguous")

    conditions = [
        ExperimentCondition(
            name="binary_gate",
            gate_config=GateConfig(variant="binary"),
            calcium_config=calcium_config,
            domain_config=domain_config,
            learning_rate=0.01,
            tau=100.0,
        ),
        ExperimentCondition(
            name="directional_gate",
            gate_config=GateConfig(variant="directional", prediction_decay=0.95),
            calcium_config=calcium_config,
            domain_config=domain_config,
            learning_rate=0.01,
            tau=100.0,
        ),
        ExperimentCondition(
            name="volume_teaching",
            gate_config=GateConfig(
                variant="volume_teaching",
                gap_junction_strength=0.1,
                n_classes=10,
            ),
            calcium_config=calcium_config,
            domain_config=domain_config,
            learning_rate=0.01,
            tau=100.0,
        ),
    ]

    results = []
    for condition in conditions:
        print(f"\n--- {condition.name} ---")
        print(f"  Start: {datetime.now(timezone.utc).isoformat()}")

        result = train_with_gate(
            condition=condition,
            n_epochs=3,
            batch_size=128,
            device=device,
            verbose=True,
        )
        results.append(result)

        print(f"  End: {datetime.now(timezone.utc).isoformat()}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'Accuracy':>10} {'NaN/Inf':>10} {'Gate Open':>10}")
    print("-" * 60)

    all_ok = True
    for result in results:
        final_acc = result["final_accuracy"]
        has_nan = result["any_nan"]
        gate_open = result["epoch_results"][-1]["gate_fraction_open"]

        status = "✗ FAIL" if has_nan else "✓ OK"
        print(f"{result['condition']:<20} {final_acc:>10.4f} {status:>10} {gate_open:>10.3f}")

        if has_nan:
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("✓ All gate variants completed without NaN/Inf")
    else:
        print("✗ Some variants had NaN/Inf issues")

    # Verify gates produce different dynamics
    accuracies = [r["final_accuracy"] for r in results]
    gate_opens = [r["epoch_results"][-1]["gate_fraction_open"] for r in results]
    print(f"\nAccuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Gate fractions: {[f'{g:.3f}' for g in gate_opens]}")

    print(f"\nEnd: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
