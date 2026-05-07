#!/usr/bin/env python3
"""Analyze results from the full Predictive Coding + BCM experiment.

Loads results/full_results.json, computes mean±std across seeds,
evaluates success criteria, generates results/summary.md.

Success criteria:
  1. Above chance (>10%) for predictive_bcm_full
  2. Combination outperforms parts (full > no_astrocyte, full > predictive_only, full > bcm_only)
  3. Prediction errors decrease over training
  4. Domain-level ≈ neuron-level performance (within 5%)
  5. Above forward-forward baseline (16.5%)

Usage:
    cd steps/14-predictive-coding-bcm
    python -m code.scripts.analyze_results
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

_step14_dir = str(Path(__file__).parent.parent.parent)
if _step14_dir not in sys.path:
    sys.path.insert(0, _step14_dir)


def load_results(results_path: Path) -> dict:
    """Load full experiment results."""
    with open(results_path) as f:
        return json.load(f)


def compute_stats(results: list[dict]) -> dict[str, dict]:
    """Compute mean±std accuracy per condition across seeds."""
    conditions = {}
    for r in results:
        name = r["condition"]
        if name not in conditions:
            conditions[name] = {"accuracies": [], "epoch_results": []}
        conditions[name]["accuracies"].append(r["final_accuracy"])
        conditions[name]["epoch_results"].append(r["epoch_results"])

    stats = {}
    for name, data in conditions.items():
        accs = data["accuracies"]
        stats[name] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "min": np.min(accs),
            "max": np.max(accs),
            "n_seeds": len(accs),
            "accuracies": accs,
            "epoch_results": data["epoch_results"],
        }
    return stats


def analyze_prediction_errors(stats: dict) -> dict[str, dict]:
    """Analyze prediction error trends for each condition.

    Returns per-condition analysis of whether prediction errors decrease.
    """
    analysis = {}
    for name, s in stats.items():
        if not s["epoch_results"]:
            continue

        # Average prediction errors across seeds for first and last epoch
        first_errors_all = []
        last_errors_all = []

        for seed_results in s["epoch_results"]:
            if not seed_results:
                continue
            first_ep = seed_results[0]
            last_ep = seed_results[-1]

            first_pred = first_ep.get("prediction_errors", {})
            last_pred = last_ep.get("prediction_errors", {})

            if first_pred:
                first_errors_all.append(sum(first_pred.values()) / len(first_pred))
            if last_pred:
                last_errors_all.append(sum(last_pred.values()) / len(last_pred))

        if first_errors_all and last_errors_all:
            first_mean = np.mean(first_errors_all)
            last_mean = np.mean(last_errors_all)
            decreased = last_mean < first_mean
            reduction_pct = (first_mean - last_mean) / max(first_mean, 1e-8) * 100
        else:
            first_mean = 0.0
            last_mean = 0.0
            decreased = False
            reduction_pct = 0.0

        analysis[name] = {
            "first_epoch_mean_error": first_mean,
            "last_epoch_mean_error": last_mean,
            "decreased": decreased,
            "reduction_percent": reduction_pct,
        }

    return analysis


def evaluate_success_criteria(stats: dict, pred_analysis: dict) -> dict:
    """Evaluate the 5 success criteria."""
    ff_baseline = 0.165  # Forward-forward baseline

    full_mean = stats.get("predictive_bcm_full", {}).get("mean", 0.0)
    no_astro_mean = stats.get("predictive_bcm_no_astrocyte", {}).get("mean", 0.0)
    pred_only_mean = stats.get("predictive_only", {}).get("mean", 0.0)
    bcm_only_mean = stats.get("bcm_only", {}).get("mean", 0.0)
    neuron_mean = stats.get("predictive_neuron_level", {}).get("mean", 0.0)
    backprop_mean = stats.get("backprop", {}).get("mean", 0.0)

    # Criterion 1: Above chance
    above_chance = full_mean > 0.10

    # Criterion 2: Combination outperforms parts
    combination_better = (
        full_mean > no_astro_mean
        and full_mean > pred_only_mean
        and full_mean > bcm_only_mean
    )

    # Criterion 3: Prediction errors decrease
    full_pred = pred_analysis.get("predictive_bcm_full", {})
    pred_errors_decrease = full_pred.get("decreased", False)

    # Criterion 4: Domain ≈ neuron (within 5 percentage points)
    domain_neuron_comparable = abs(full_mean - neuron_mean) < 0.05

    # Criterion 5: Above forward-forward baseline
    above_ff = full_mean > ff_baseline

    return {
        "above_chance": above_chance,
        "combination_outperforms_parts": combination_better,
        "prediction_errors_decrease": pred_errors_decrease,
        "domain_vs_neuron_comparable": domain_neuron_comparable,
        "above_forward_forward": above_ff,
        "details": {
            "full_accuracy": full_mean,
            "no_astrocyte_accuracy": no_astro_mean,
            "predictive_only_accuracy": pred_only_mean,
            "bcm_only_accuracy": bcm_only_mean,
            "neuron_level_accuracy": neuron_mean,
            "backprop_accuracy": backprop_mean,
            "ff_baseline": ff_baseline,
            "pred_error_reduction_pct": full_pred.get("reduction_percent", 0.0),
        },
    }


def generate_summary(stats: dict, pred_analysis: dict, criteria: dict, data: dict) -> str:
    """Generate summary markdown."""
    lines = []
    lines.append("# Step 14: Predictive Coding + BCM — Results Summary\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

    # Experiment metadata
    lines.append("## Experiment Configuration\n")
    lines.append(f"- **Epochs**: {data.get('n_epochs', 50)}")
    lines.append(f"- **Seeds**: {data.get('seeds', [42, 123, 456])}")
    lines.append(f"- **Batch size**: {data.get('batch_size', 128)}")
    lines.append(f"- **Duration**: {data.get('duration_seconds', 0):.0f}s ({data.get('duration_seconds', 0)/60:.1f} min)")
    lines.append(f"- **Conditions**: {data.get('conditions', [])}")
    lines.append("")

    # Results table
    lines.append("## Final Accuracy by Condition\n")
    lines.append("| Condition | Mean Accuracy | Std | Min | Max |")
    lines.append("|-----------|:------------:|:---:|:---:|:---:|")

    condition_order = [
        "predictive_bcm_full", "predictive_bcm_no_astrocyte", "predictive_only",
        "bcm_only", "predictive_neuron_level", "backprop",
    ]
    for name in condition_order:
        if name in stats:
            s = stats[name]
            lines.append(
                f"| {name} | {s['mean']*100:.2f}% | ±{s['std']*100:.2f}% | "
                f"{s['min']*100:.2f}% | {s['max']*100:.2f}% |"
            )
    lines.append("")

    # Prediction error analysis
    lines.append("## Prediction Error Trend Analysis\n")
    lines.append("| Condition | First Epoch Error | Last Epoch Error | Decreased? | Reduction |")
    lines.append("|-----------|:-----------------:|:----------------:|:----------:|:---------:|")
    for name in condition_order:
        if name in pred_analysis:
            pa = pred_analysis[name]
            decreased_str = "✓" if pa["decreased"] else "✗"
            lines.append(
                f"| {name} | {pa['first_epoch_mean_error']:.4f} | "
                f"{pa['last_epoch_mean_error']:.4f} | {decreased_str} | "
                f"{pa['reduction_percent']:.1f}% |"
            )
    lines.append("")

    # Success criteria
    lines.append("## Success Criteria Evaluation\n")
    lines.append("| Criterion | Result | Details |")
    lines.append("|-----------|:------:|---------|")

    c = criteria
    d = c["details"]

    lines.append(f"| Above chance (>10%) | {'✓' if c['above_chance'] else '✗'} | "
                 f"Full: {d['full_accuracy']*100:.2f}% |")
    lines.append(f"| Combination > parts | {'✓' if c['combination_outperforms_parts'] else '✗'} | "
                 f"Full > no_astro ({d['no_astrocyte_accuracy']*100:.2f}%), "
                 f"pred_only ({d['predictive_only_accuracy']*100:.2f}%), "
                 f"bcm_only ({d['bcm_only_accuracy']*100:.2f}%) |")
    lines.append(f"| Prediction errors decrease | {'✓' if c['prediction_errors_decrease'] else '✗'} | "
                 f"Reduction: {d['pred_error_reduction_pct']:.1f}% |")
    lines.append(f"| Domain ≈ neuron (±5%) | {'✓' if c['domain_vs_neuron_comparable'] else '✗'} | "
                 f"Domain: {d['full_accuracy']*100:.2f}%, Neuron: {d['neuron_level_accuracy']*100:.2f}% |")
    lines.append(f"| Above FF baseline (16.5%) | {'✓' if c['above_forward_forward'] else '✗'} | "
                 f"Full: {d['full_accuracy']*100:.2f}% vs 16.5% |")
    lines.append("")

    passed = sum([
        c["above_chance"], c["combination_outperforms_parts"],
        c["prediction_errors_decrease"], c["domain_vs_neuron_comparable"],
        c["above_forward_forward"],
    ])
    lines.append(f"**Overall: {passed}/5 criteria met.**\n")

    # Comparison with baselines
    lines.append("## Comparison with Baselines\n")
    lines.append(f"- **Forward-forward** (literature): 16.5%")
    lines.append(f"- **BCM-only** (Step 12b baseline): {d['bcm_only_accuracy']*100:.2f}%")
    lines.append(f"- **Predictive BCM full** (this work): {d['full_accuracy']*100:.2f}%")
    lines.append(f"- **Backprop** (upper bound): {d['backprop_accuracy']*100:.2f}%")
    lines.append("")

    improvement_over_bcm = d["full_accuracy"] - d["bcm_only_accuracy"]
    if improvement_over_bcm > 0:
        lines.append(f"Predictive coding improves over BCM-only by **+{improvement_over_bcm*100:.2f}%**.")
    else:
        lines.append(f"Predictive coding does not improve over BCM-only (Δ = {improvement_over_bcm*100:.2f}%).")
    lines.append("")

    # Key findings
    lines.append("## Key Findings\n")
    lines.append("### Does prediction error provide the missing task-relevant signal?\n")

    if c["above_chance"] and c["combination_outperforms_parts"]:
        lines.append("**YES** — Domain-level prediction errors between adjacent layers provide ")
        lines.append("task-relevant information that improves upon BCM direction alone. The ")
        lines.append("combination of BCM direction (signed updates) with prediction error ")
        lines.append("(task relevance) outperforms either component in isolation.")
    elif c["above_chance"]:
        lines.append("**PARTIALLY** — The system learns above chance, but the combination ")
        lines.append("does not clearly outperform all individual components. The prediction ")
        lines.append("error signal may need further tuning or more epochs to demonstrate ")
        lines.append("clear synergy with BCM direction.")
    else:
        lines.append("**NOT YET** — The system has not achieved above-chance accuracy within ")
        lines.append("the training budget. Possible explanations:")
        lines.append("")
        lines.append("1. Learning rate may need tuning for the combined signal")
        lines.append("2. Prediction errors may be too noisy early in training")
        lines.append("3. More epochs may be needed for convergence")
        lines.append("4. The multiplicative combination may suppress learning when either signal is weak")
    lines.append("")

    if c["prediction_errors_decrease"]:
        lines.append("### Prediction Error Convergence\n")
        lines.append("Prediction errors decrease over training, confirming that the ")
        lines.append("inter-layer prediction mechanism learns meaningful structure. ")
        lines.append("This validates the core hypothesis that domain-level predictions ")
        lines.append("can capture useful inter-layer relationships.")
        lines.append("")

    if c["domain_vs_neuron_comparable"]:
        lines.append("### Domain vs Neuron Granularity\n")
        lines.append("Domain-level prediction (8×8 matrices) achieves comparable performance ")
        lines.append("to neuron-level prediction (128×128 matrices), validating the design ")
        lines.append("choice to operate at the astrocyte domain level. This is both more ")
        lines.append("biologically faithful and computationally efficient.")
        lines.append("")

    return "\n".join(lines)


def main():
    results_dir = Path(_step14_dir) / "results"
    results_path = results_dir / "full_results.json"

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run the full experiment first: python -m code.scripts.run_experiment")

        # Try quick results as fallback
        quick_path = results_dir / "quick_results.json"
        if quick_path.exists():
            print("\nGenerating preliminary summary from quick results...")
            with open(quick_path) as f:
                quick = json.load(f)

            stats = {
                "predictive_bcm_full": {
                    "mean": quick["final_accuracy"],
                    "std": 0.0,
                    "min": quick["final_accuracy"],
                    "max": quick["final_accuracy"],
                    "n_seeds": 1,
                    "accuracies": [quick["final_accuracy"]],
                    "epoch_results": [quick["epoch_results"]],
                },
            }
            pred_analysis = analyze_prediction_errors(stats)
            criteria = evaluate_success_criteria(stats, pred_analysis)
            data = {
                "n_epochs": quick["n_epochs"],
                "seeds": [quick["seed"]],
                "batch_size": quick["batch_size"],
                "duration_seconds": quick["duration_seconds"],
                "conditions": [quick["condition"]],
            }

            summary = generate_summary(stats, pred_analysis, criteria, data)
            summary_path = results_dir / "summary.md"
            with open(summary_path, "w") as f:
                f.write(summary)
            print(f"Preliminary summary saved to: {summary_path}")
        return

    # Load full results
    data = load_results(results_path)
    results = data["results"]

    # Compute statistics
    stats = compute_stats(results)
    pred_analysis = analyze_prediction_errors(stats)
    criteria = evaluate_success_criteria(stats, pred_analysis)

    # Print summary
    print("=" * 60)
    print("STEP 14: RESULTS ANALYSIS")
    print("=" * 60)

    for name, s in stats.items():
        print(f"  {name:30s}: {s['mean']*100:.2f}% ± {s['std']*100:.2f}%")

    print(f"\nSuccess Criteria:")
    print(f"  Above chance: {'✓' if criteria['above_chance'] else '✗'}")
    print(f"  Combination > parts: {'✓' if criteria['combination_outperforms_parts'] else '✗'}")
    print(f"  Pred errors decrease: {'✓' if criteria['prediction_errors_decrease'] else '✗'}")
    print(f"  Domain ≈ neuron: {'✓' if criteria['domain_vs_neuron_comparable'] else '✗'}")
    print(f"  Above FF baseline: {'✓' if criteria['above_forward_forward'] else '✗'}")

    passed = sum([
        criteria["above_chance"], criteria["combination_outperforms_parts"],
        criteria["prediction_errors_decrease"], criteria["domain_vs_neuron_comparable"],
        criteria["above_forward_forward"],
    ])
    print(f"\n  Overall: {passed}/5 criteria met")

    # Generate summary
    summary = generate_summary(stats, pred_analysis, criteria, data)
    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
