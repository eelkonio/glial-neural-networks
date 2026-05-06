"""Central prediction test for Step 13.

Tests the hypothesis: astrocyte gating provides greater benefit under
local learning rules than under backpropagation.

benefit_local = best_gated_accuracy - ungated_baseline (three_factor_random = 10%)
benefit_backprop = 0.14% (Phase 1 measured value from Step 01 v2)

Conclusion: confirmed if benefit_local > benefit_backprop with CI not overlapping zero.
"""

from pathlib import Path
from typing import Any

import numpy as np

from code.experiment.metrics import ConditionResult


# Phase 1 measured benefit under backprop (from Step 01 v2)
BENEFIT_BACKPROP_PERCENT = 0.14

# Step 12 baseline: three_factor_random accuracy
UNGATED_BASELINE_PERCENT = 10.0


def compute_central_prediction(
    results: list[ConditionResult],
    ungated_baseline: float = UNGATED_BASELINE_PERCENT,
    benefit_backprop: float = BENEFIT_BACKPROP_PERCENT,
) -> dict[str, Any]:
    """Compute central prediction test.

    Args:
        results: All ConditionResult from the experiment.
        ungated_baseline: Ungated three-factor baseline accuracy (%).
        benefit_backprop: Measured benefit under backprop (%).

    Returns:
        Dict with benefit_local, benefit_backprop, CI, conclusion.
    """
    # Find gated conditions (binary_gate, directional_gate, volume_teaching)
    gated_names = {"binary_gate", "directional_gate", "volume_teaching"}

    # Group gated results by condition
    gated_by_condition: dict[str, list[float]] = {}
    for r in results:
        if r.condition_name in gated_names:
            if r.condition_name not in gated_by_condition:
                gated_by_condition[r.condition_name] = []
            gated_by_condition[r.condition_name].append(r.final_accuracy * 100)

    if not gated_by_condition:
        return {
            "conclusion": "inconclusive",
            "reason": "No gated condition results found",
            "benefit_local_percent": 0.0,
            "benefit_backprop_percent": benefit_backprop,
        }

    # Find best gated condition (highest mean accuracy)
    best_condition = None
    best_mean = -float("inf")
    for name, accs in gated_by_condition.items():
        mean_acc = np.mean(accs)
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_condition = name

    best_accs = gated_by_condition[best_condition]

    # Compute benefit_local = best_gated - ungated_baseline
    benefit_local_values = [acc - ungated_baseline for acc in best_accs]
    benefit_local_mean = float(np.mean(benefit_local_values))
    benefit_local_std = float(np.std(benefit_local_values))

    # 95% CI (t-distribution approximation for small n)
    n_seeds = len(benefit_local_values)
    if n_seeds > 1:
        se = benefit_local_std / np.sqrt(n_seeds)
        # Use t=2.0 for approximate 95% CI with small n
        ci_lower = benefit_local_mean - 2.0 * se
        ci_upper = benefit_local_mean + 2.0 * se
    else:
        ci_lower = benefit_local_mean
        ci_upper = benefit_local_mean

    # Determine conclusion
    if benefit_local_mean > benefit_backprop and ci_lower > 0:
        conclusion = "hypothesis confirmed"
    elif benefit_local_mean <= benefit_backprop or ci_upper < benefit_backprop:
        conclusion = "hypothesis refuted"
    else:
        conclusion = "inconclusive"

    return {
        "conclusion": conclusion,
        "best_gated_condition": best_condition,
        "best_gated_mean_accuracy_percent": float(best_mean),
        "ungated_baseline_percent": ungated_baseline,
        "benefit_local_percent": benefit_local_mean,
        "benefit_local_std_percent": benefit_local_std,
        "benefit_local_ci_lower": ci_lower,
        "benefit_local_ci_upper": ci_upper,
        "benefit_backprop_percent": benefit_backprop,
        "n_seeds": n_seeds,
        "all_gated_accuracies": {
            name: [float(a) for a in accs]
            for name, accs in gated_by_condition.items()
        },
    }


def generate_central_prediction_chart(
    prediction_result: dict[str, Any],
    output_dir: str | Path,
    filename: str = "central_prediction_test.png",
) -> Path:
    """Generate bar chart comparing benefit_local vs benefit_backprop.

    Args:
        prediction_result: Output from compute_central_prediction.
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

    benefit_local = prediction_result["benefit_local_percent"]
    benefit_backprop = prediction_result["benefit_backprop_percent"]
    ci_lower = prediction_result.get("benefit_local_ci_lower", benefit_local)
    ci_upper = prediction_result.get("benefit_local_ci_upper", benefit_local)

    # Error bars: [lower_error, upper_error] for each bar
    local_err_low = max(0, benefit_local - ci_lower)
    local_err_high = max(0, ci_upper - benefit_local)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        [0, 1],
        [benefit_local, benefit_backprop],
        yerr=[[local_err_low, 0], [local_err_high, 0]],
        capsize=8,
        color=["#9C27B0", "#607D8B"],
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"Benefit under\nLocal Rules\n({prediction_result.get('best_gated_condition', 'gated')})",
        "Benefit under\nBackprop\n(Phase 1 measured)",
    ])
    ax.set_ylabel("Accuracy Improvement (%)")
    ax.set_title(f"Central Prediction Test: {prediction_result['conclusion'].upper()}")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, [benefit_local, benefit_backprop]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath
