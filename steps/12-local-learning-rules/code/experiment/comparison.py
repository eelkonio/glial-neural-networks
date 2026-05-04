"""Comparison and visualization for local learning rule experiments.

Generates summary tables, accuracy bar charts, convergence curves,
and weight norm trajectory plots.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from code.experiment.metrics import PerformanceMetrics


def generate_summary_table(
    metrics_list: list[PerformanceMetrics], output_path: Path
) -> list[dict[str, Any]]:
    """Generate summary comparison table CSV with mean/std accuracy per rule.

    Args:
        metrics_list: All collected metrics across rules and seeds.
        output_path: Path to write the summary CSV.

    Returns:
        List of summary row dicts.
    """
    # Group by rule name
    rule_results: dict[str, list[PerformanceMetrics]] = {}
    for m in metrics_list:
        rule_results.setdefault(m.rule_name, []).append(m)

    summary_rows = []
    for rule_name, metrics in sorted(rule_results.items()):
        final_accs = [m.accuracy_history[-1] for m in metrics if m.accuracy_history]
        convergence_epochs = [
            m.convergence_epoch for m in metrics if m.convergence_epoch is not None
        ]
        stabilities = [m.stability for m in metrics]

        summary_rows.append({
            "rule": rule_name,
            "mean_accuracy": float(np.mean(final_accs)) if final_accs else 0.0,
            "std_accuracy": float(np.std(final_accs)) if final_accs else 0.0,
            "convergence_epoch": (
                float(np.mean(convergence_epochs)) if convergence_epochs else None
            ),
            "stability": float(np.mean(stabilities)) if stabilities else 0.0,
            "n_seeds": len(metrics),
        })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule",
        "mean_accuracy",
        "std_accuracy",
        "convergence_epoch",
        "stability",
        "n_seeds",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def plot_accuracy_comparison(
    metrics_list: list[PerformanceMetrics], output_path: Path
) -> None:
    """Generate accuracy bar chart comparing all rules.

    Args:
        metrics_list: All collected metrics.
        output_path: Path to save the PNG.
    """
    if not HAS_MATPLOTLIB:
        return

    # Group by rule
    rule_results: dict[str, list[float]] = {}
    for m in metrics_list:
        if m.accuracy_history:
            rule_results.setdefault(m.rule_name, []).append(
                m.accuracy_history[-1]
            )

    rules = sorted(rule_results.keys())
    means = [np.mean(rule_results[r]) for r in rules]
    stds = [np.std(rule_results[r]) for r in rules]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rules))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)

    ax.set_xlabel("Learning Rule")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy by Learning Rule")
    ax.set_xticks(x)
    ax.set_xticklabels(rules, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_convergence_curves(
    metrics_list: list[PerformanceMetrics], output_path: Path
) -> None:
    """Generate convergence curves plot for all rules.

    Shows test accuracy over epochs, averaged across seeds with
    shaded std regions.

    Args:
        metrics_list: All collected metrics.
        output_path: Path to save the PNG.
    """
    if not HAS_MATPLOTLIB:
        return

    # Group by rule
    rule_histories: dict[str, list[list[float]]] = {}
    for m in metrics_list:
        if m.accuracy_history:
            rule_histories.setdefault(m.rule_name, []).append(m.accuracy_history)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(rule_histories)))

    for (rule_name, histories), color in zip(
        sorted(rule_histories.items()), colors
    ):
        # Pad to same length
        max_len = max(len(h) for h in histories)
        padded = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            padded[i, : len(h)] = h

        mean_acc = np.nanmean(padded, axis=0)
        std_acc = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        ax.plot(epochs, mean_acc, label=rule_name, color=color)
        ax.fill_between(
            epochs,
            mean_acc - std_acc,
            mean_acc + std_acc,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Convergence Curves by Learning Rule")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_weight_norm_trajectories(
    metrics_list: list[PerformanceMetrics], output_path: Path
) -> None:
    """Generate weight norm trajectory plot for all rules.

    Shows mean weight norm (averaged across layers) over epochs.

    Args:
        metrics_list: All collected metrics.
        output_path: Path to save the PNG.
    """
    if not HAS_MATPLOTLIB:
        return

    # Group by rule
    rule_norms: dict[str, list[list[float]]] = {}
    for m in metrics_list:
        # Compute mean weight norm per epoch
        epoch_mean_norms = []
        for e in m.epochs:
            if e.weight_norms:
                epoch_mean_norms.append(np.mean(e.weight_norms))
        if epoch_mean_norms:
            rule_norms.setdefault(m.rule_name, []).append(epoch_mean_norms)

    if not rule_norms:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rule_norms), 1)))

    for (rule_name, norm_histories), color in zip(
        sorted(rule_norms.items()), colors
    ):
        max_len = max(len(h) for h in norm_histories)
        if max_len == 0:
            continue
        padded = np.full((len(norm_histories), max_len), np.nan)
        for i, h in enumerate(norm_histories):
            padded[i, : len(h)] = h

        # Skip if all NaN
        if np.all(np.isnan(padded)):
            continue

        with np.errstate(all="ignore"):
            mean_norms = np.nanmean(padded, axis=0)
        epochs = np.arange(max_len)

        ax.plot(epochs, mean_norms, label=rule_name, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Weight Norm (L2)")
    ax.set_title("Weight Norm Trajectories by Learning Rule")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
