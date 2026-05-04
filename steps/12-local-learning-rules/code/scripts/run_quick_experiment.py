#!/usr/bin/env python3
"""Quick experiment run: all 8 rule configurations × 1 seed × 10 epochs.

A shorter version of run_full_experiment.py for pipeline verification.
Expected runtime: ~10-15 minutes on Apple M4 Pro (MPS).

Usage:
    python -m code.scripts.run_quick_experiment
"""

import sys
import time
from pathlib import Path

# Ensure correct imports
step_dir = str(Path(__file__).parent.parent.parent)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

import torch

from code.experiment.runner import (
    ExperimentRunner,
    get_device,
    set_seed,
    train_backprop,
    train_forward_forward,
    train_local_rule,
    train_predictive_coding,
)
from code.experiment.metrics import (
    PerformanceMetrics,
    compute_weight_norms,
)
from code.experiment.comparison import (
    generate_summary_table,
    plot_accuracy_comparison,
    plot_convergence_curves,
    plot_weight_norm_trajectories,
)
from code.experiment.deficiency import (
    run_full_deficiency_analysis,
    generate_credit_assignment_heatmap,
    generate_deficiency_report,
)
from code.experiment.spatial_quality import (
    compute_spatial_quality,
    compute_backprop_spatial_quality,
    save_spatial_quality_results,
)
from code.rules.hebbian import HebbianRule
from code.rules.oja import OjaRule
from code.rules.three_factor import (
    ThreeFactorRule,
    RandomNoiseThirdFactor,
    GlobalRewardThirdFactor,
    LayerWiseErrorThirdFactor,
)
from code.rules.forward_forward import ForwardForwardRule
from code.rules.predictive_coding import PredictiveCodingRule
from code.data.fashion_mnist import get_fashion_mnist_loaders


# Quick configuration
N_EPOCHS = 10
SEEDS = [42]
BATCH_SIZE = 128

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_rule_configs() -> list[tuple[str, callable, dict]]:
    """Get all rule configurations to run."""
    return [
        ("backprop", train_backprop, {}),
        ("hebbian", train_local_rule, {"rule": HebbianRule(lr=0.01, weight_decay=0.001)}),
        ("oja", train_local_rule, {"rule": OjaRule(lr=0.01)}),
        (
            "three_factor_random",
            train_local_rule,
            {"rule": ThreeFactorRule(third_factor=RandomNoiseThirdFactor())},
        ),
        (
            "three_factor_reward",
            train_local_rule,
            {"rule": ThreeFactorRule(third_factor=GlobalRewardThirdFactor())},
        ),
        (
            "three_factor_error",
            train_local_rule,
            {"rule": ThreeFactorRule(third_factor=LayerWiseErrorThirdFactor())},
        ),
        ("forward_forward", train_forward_forward, {"rule": ForwardForwardRule()}),
        ("predictive_coding", train_predictive_coding, {"rule": PredictiveCodingRule()}),
    ]


def main():
    """Run the quick experiment pipeline."""
    total_start = time.time()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  LOCAL LEARNING RULES — QUICK EXPERIMENT (verification)       ║")
    print("║  8 rules × 1 seed × 10 epochs                                ║")
    print("╚" + "═" * 68 + "╝")

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Results: {RESULTS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metadata
    runner = ExperimentRunner(results_dir=RESULTS_DIR, data_dir=DATA_DIR)
    runner.save_metadata({
        "n_epochs": N_EPOCHS,
        "seeds": SEEDS,
        "batch_size": BATCH_SIZE,
        "mode": "quick_verification",
        "rules": [name for name, _, _ in get_rule_configs()],
    })

    # Phase 1: Train all rules
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)

    all_results = []
    for rule_name, train_fn, kwargs in get_rule_configs():
        for seed in SEEDS:
            print(f"\n  Training: {rule_name} (seed={seed}, epochs={N_EPOCHS})")
            start = time.time()

            result = train_fn(
                epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                seed=seed,
                device=device,
                verbose=False,
                **kwargs,
            )

            elapsed = time.time() - start
            result["wall_clock_seconds"] = elapsed
            result["rule_name"] = rule_name
            result["seed"] = seed

            print(f"    ✓ acc={result['final_accuracy']:.4f}, time={elapsed:.1f}s")
            all_results.append((rule_name, result))

    # Phase 2: Metrics
    print("\n" + "=" * 60)
    print("PHASE 2: METRICS & PLOTS")
    print("=" * 60)

    all_metrics = []
    for rule_name, result in all_results:
        metrics = runner.collect_metrics(
            result, rule_name, model=result.get("model")
        )
        all_metrics.append(metrics)

    PerformanceMetrics.save_all_to_csv(
        all_metrics, RESULTS_DIR / "performance_comparison.csv"
    )
    generate_summary_table(all_metrics, RESULTS_DIR / "summary_table.csv")
    plot_accuracy_comparison(all_metrics, RESULTS_DIR / "accuracy_comparison.png")
    plot_convergence_curves(all_metrics, RESULTS_DIR / "convergence_curves.png")
    plot_weight_norm_trajectories(
        all_metrics, RESULTS_DIR / "weight_norm_trajectories.png"
    )
    print("  ✓ Metrics and plots saved")

    # Phase 3: Deficiency analysis
    print("\n" + "=" * 60)
    print("PHASE 3: DEFICIENCY ANALYSIS")
    print("=" * 60)

    train_loader, _ = get_fashion_mnist_loaders(batch_size=BATCH_SIZE)
    credit_data = {}
    deficiency_results = {}
    seen_rules = set()

    for rule_name, result in all_results:
        if rule_name in seen_rules or rule_name == "backprop":
            continue
        seen_rules.add(rule_name)

        model = result.get("model")
        if model is None:
            continue

        rule = None
        for rn, _, kwargs in get_rule_configs():
            if rn == rule_name and "rule" in kwargs:
                rule = kwargs["rule"]
                break

        if rule is None:
            continue

        print(f"  Analyzing: {rule_name}")
        analysis = run_full_deficiency_analysis(
            model, rule, rule_name, train_loader, device=device
        )
        credit_data[rule_name] = analysis["credit_assignment"]
        deficiency_results[rule_name] = analysis

    generate_credit_assignment_heatmap(
        credit_data, RESULTS_DIR / "credit_assignment_heatmap.png"
    )
    generate_deficiency_report(
        deficiency_results, RESULTS_DIR / "deficiency_analysis.md"
    )
    print("  ✓ Deficiency analysis complete")

    # Phase 4: Spatial quality
    print("\n" + "=" * 60)
    print("PHASE 4: SPATIAL QUALITY")
    print("=" * 60)

    backprop_model = None
    for rule_name, result in all_results:
        if rule_name == "backprop":
            backprop_model = result.get("model")
            break

    backprop_corr = 0.0
    if backprop_model is not None:
        backprop_corr = compute_backprop_spatial_quality(
            backprop_model, train_loader, n_batches=10, device=device
        )
        print(f"  Backprop spatial correlation: {backprop_corr:.4f}")

    spatial_results = []
    seen_rules = set()

    for rule_name, result in all_results:
        if rule_name in seen_rules or rule_name == "backprop":
            continue
        seen_rules.add(rule_name)

        model = result.get("model")
        if model is None:
            continue

        rule = None
        for rn, _, kwargs in get_rule_configs():
            if rn == rule_name and "rule" in kwargs:
                rule = kwargs["rule"]
                break

        if rule is None:
            continue

        spatial_corr = compute_spatial_quality(
            model, rule, train_loader, n_batches=10, device=device
        )
        ratio = spatial_corr / backprop_corr if abs(backprop_corr) > 1e-8 else 0.0

        spatial_results.append({
            "rule": rule_name,
            "spatial_correlation": round(spatial_corr, 6),
            "backprop_correlation": round(backprop_corr, 6),
            "ratio": round(ratio, 4),
        })
        print(f"  {rule_name}: corr={spatial_corr:.4f}")

    save_spatial_quality_results(spatial_results, RESULTS_DIR / "spatial_quality.csv")
    print("  ✓ Spatial quality complete")

    # Phase 5: Summary
    print("\n" + "=" * 60)
    print("PHASE 5: SUMMARY")
    print("=" * 60)

    rule_accs: dict[str, list[float]] = {}
    for rule_name, result in all_results:
        rule_accs.setdefault(rule_name, []).append(result.get("final_accuracy", 0.0))

    lines = [
        "# Local Learning Rules — Quick Experiment Summary",
        "",
        f"**Configuration**: {N_EPOCHS} epochs, seed={SEEDS}, batch_size={BATCH_SIZE}",
        "",
        "## Final Test Accuracy",
        "",
        "| Rule | Accuracy |",
        "|------|----------|",
    ]

    for rule_name in sorted(rule_accs.keys()):
        accs = rule_accs[rule_name]
        lines.append(f"| {rule_name} | {accs[0]:.4f} |")

    lines.extend([
        "",
        "## Verification Status",
        "",
        "- [ ] Backprop achieves >85% in 10 epochs",
        "- [ ] All output files generated",
        "- [ ] No training errors",
        "",
        "*This is a quick verification run. See run_full_experiment.py for the full 50-epoch run.*",
    ])

    summary_path = RESULTS_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    # Final report
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"QUICK EXPERIMENT COMPLETE — {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Verify backprop accuracy
    backprop_acc = rule_accs.get("backprop", [0.0])[0]
    print(f"\nBackprop accuracy: {backprop_acc:.4f}")
    if backprop_acc > 0.85:
        print("  ✓ PASS: Backprop > 85% in 10 epochs")
    else:
        print("  ⚠ WARNING: Backprop < 85% (may need more epochs)")

    # Check output files
    expected_files = [
        "performance_comparison.csv",
        "summary_table.csv",
        "accuracy_comparison.png",
        "convergence_curves.png",
        "weight_norm_trajectories.png",
        "credit_assignment_heatmap.png",
        "deficiency_analysis.md",
        "spatial_quality.csv",
        "summary.md",
    ]

    missing = [f for f in expected_files if not (RESULTS_DIR / f).exists()]
    if not missing:
        print("  ✓ PASS: All output files generated")
    else:
        print(f"  ⚠ MISSING: {missing}")


if __name__ == "__main__":
    main()
