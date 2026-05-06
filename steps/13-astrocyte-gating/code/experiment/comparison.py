"""Comparison visualization for Step 13 experiments.

Generates:
- Accuracy bar chart (results/accuracy_comparison.png)
- Convergence curves (results/convergence_curves.png)
- Summary CSV table
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from code.experiment.metrics import ConditionResult


def compute_summary_stats(
    results: list[ConditionResult],
) -> dict[str, dict[str, float]]:
    """Compute mean/std accuracy per condition across seeds.

    Args:
        results: List of ConditionResult from all conditions/seeds.

    Returns:
        Dict mapping condition_name -> {mean_acc, std_acc, best_acc, any_nan}.
    """
    # Group by condition
    by_condition: dict[str, list[ConditionResult]] = {}
    for r in results:
        if r.condition_name not in by_condition:
            by_condition[r.condition_name] = []
        by_condition[r.condition_name].append(r)

    stats = {}
    for name, cond_results in by_condition.items():
        accs = [r.final_accuracy for r in cond_results]
        best_accs = [r.best_accuracy for r in cond_results]
        stats[name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "best_accuracy": float(np.max(best_accs)),
            "n_seeds": len(cond_results),
            "any_nan": any(r.any_nan for r in cond_results),
        }

    return stats


def save_summary_csv(
    stats: dict[str, dict[str, float]],
    output_dir: str | Path,
    filename: str = "summary_comparison.csv",
) -> Path:
    """Save summary comparison table to CSV.

    Args:
        stats: Output from compute_summary_stats.
        output_dir: Directory to write CSV.
        filename: Output filename.

    Returns:
        Path to written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    fieldnames = ["condition", "mean_accuracy", "std_accuracy", "best_accuracy", "n_seeds", "any_nan"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, s in stats.items():
            writer.writerow({
                "condition": name,
                "mean_accuracy": f"{s['mean_accuracy']:.6f}",
                "std_accuracy": f"{s['std_accuracy']:.6f}",
                "best_accuracy": f"{s['best_accuracy']:.6f}",
                "n_seeds": s["n_seeds"],
                "any_nan": s["any_nan"],
            })

    return filepath


def generate_accuracy_bar_chart(
    stats: dict[str, dict[str, float]],
    output_dir: str | Path,
    filename: str = "accuracy_comparison.png",
) -> Path:
    """Generate accuracy bar chart with error bars.

    Args:
        stats: Output from compute_summary_stats.
        output_dir: Directory to write PNG.
        filename: Output filename.

    Returns:
        Path to written file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    names = list(stats.keys())
    means = [stats[n]["mean_accuracy"] * 100 for n in names]
    stds = [stats[n]["std_accuracy"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                  color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"],
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Step 13: Astrocyte Gating — Performance Comparison")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath


def generate_convergence_curves(
    results: list[ConditionResult],
    output_dir: str | Path,
    filename: str = "convergence_curves.png",
) -> Path:
    """Generate convergence curves (accuracy vs epoch) for all conditions.

    Args:
        results: List of ConditionResult from all conditions/seeds.
        output_dir: Directory to write PNG.
        filename: Output filename.

    Returns:
        Path to written file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    # Group by condition
    by_condition: dict[str, list[ConditionResult]] = {}
    for r in results:
        if r.condition_name not in by_condition:
            by_condition[r.condition_name] = []
        by_condition[r.condition_name].append(r)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (name, cond_results) in enumerate(by_condition.items()):
        # Get epoch accuracies across seeds
        all_accs = []
        for r in cond_results:
            accs = [e.test_accuracy * 100 for e in r.epoch_results]
            all_accs.append(accs)

        # Compute mean and std across seeds
        max_epochs = max(len(a) for a in all_accs)
        mean_accs = []
        std_accs = []
        for epoch in range(max_epochs):
            epoch_vals = [a[epoch] for a in all_accs if epoch < len(a)]
            mean_accs.append(np.mean(epoch_vals))
            std_accs.append(np.std(epoch_vals))

        epochs = list(range(max_epochs))
        color = colors[idx % len(colors)]

        ax.plot(epochs, mean_accs, label=name, color=color, linewidth=1.5)
        ax.fill_between(
            epochs,
            [m - s for m, s in zip(mean_accs, std_accs)],
            [m + s for m, s in zip(mean_accs, std_accs)],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Step 13: Convergence Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath
