"""Deficiency analysis for local learning rules.

Characterizes what each local rule lacks compared to backpropagation:
1. Credit assignment reach (correlation with true gradient)
2. Weight stability (L2 norm trajectories)
3. Representation redundancy (pairwise cosine similarity)
4. Inter-layer coordination (CKA between adjacent layers)
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_credit_assignment_reach(
    model: nn.Module,
    rule,
    x: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | None = None,
) -> list[float]:
    """Correlation between local update signal and true gradient per layer.

    Runs a parallel backprop pass (without updating weights) to get
    the true gradient, then correlates with the rule's update signal.

    Args:
        model: The LocalMLP model.
        rule: A local learning rule with compute_update method.
            Rules that don't support compute_update (ForwardForward,
            PredictiveCoding) return zeros.
        x: Input batch (batch_size, 784).
        labels: Labels (batch_size,).
        device: Device to run on.

    Returns:
        List of correlations per layer (0 = no credit, 1 = perfect).
    """
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device).view(x.size(0), -1)
    labels = labels.to(device)

    # Check if rule supports compute_update
    has_compute_update = hasattr(rule, "compute_update")
    if has_compute_update:
        try:
            # Test if it actually works (some rules raise NotImplementedError)
            test_state = model.forward_with_states(x[:1], labels=labels[:1])[0]
            rule.compute_update(test_state)
        except (NotImplementedError, RuntimeError):
            has_compute_update = False

    n_layers = len(model.layers)
    if not has_compute_update:
        return [0.0] * n_layers

    # 1. Get true gradients via backprop (without updating weights)
    model.zero_grad()
    logits = model(x, detach=False)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    true_gradients = []
    for layer in model.layers:
        grad = layer.linear.weight.grad
        if grad is not None:
            true_gradients.append(grad.detach().clone())
        else:
            true_gradients.append(torch.zeros_like(layer.linear.weight))

    model.zero_grad()

    # 2. Get local rule's update signals
    with torch.no_grad():
        states = model.forward_with_states(x, labels=labels, global_loss=loss.item())

    local_updates = []
    with torch.no_grad():
        for state in states:
            delta = rule.compute_update(state)
            local_updates.append(delta)

    # 3. Correlate true gradient with local update per layer
    correlations = []
    for true_grad, local_update in zip(true_gradients, local_updates):
        tg = true_grad.flatten().float().cpu().numpy()
        lu = local_update.flatten().float().cpu().numpy()

        # Pearson correlation
        if np.std(tg) < 1e-12 or np.std(lu) < 1e-12:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(tg, lu)[0, 1]
            correlations.append(float(corr) if not np.isnan(corr) else 0.0)

    return correlations


def compute_weight_stability(
    weight_norm_history: list[list[float]],
) -> dict[str, Any]:
    """Analyze weight norm trajectories for unbounded growth or oscillation.

    Args:
        weight_norm_history: List of per-epoch weight norms.
            Each inner list has one norm per layer.

    Returns:
        Dict with 'growth_rate', 'oscillation', 'stable' per layer,
        plus 'overall_stable' flag.
    """
    if not weight_norm_history or not weight_norm_history[0]:
        return {"overall_stable": True, "layers": []}

    n_layers = len(weight_norm_history[0])
    n_epochs = len(weight_norm_history)

    layer_analysis = []
    for layer_idx in range(n_layers):
        norms = [
            weight_norm_history[e][layer_idx]
            for e in range(n_epochs)
            if layer_idx < len(weight_norm_history[e])
        ]

        if len(norms) < 2:
            layer_analysis.append({
                "growth_rate": 0.0,
                "oscillation": 0.0,
                "stable": True,
            })
            continue

        # Growth rate: ratio of final to initial norm
        growth_rate = norms[-1] / max(norms[0], 1e-8)

        # Oscillation: std of differences between consecutive norms
        diffs = np.diff(norms)
        oscillation = float(np.std(diffs)) if len(diffs) > 0 else 0.0

        # Stable if growth < 10x and oscillation is small relative to mean
        mean_norm = np.mean(norms)
        stable = growth_rate < 10.0 and (
            oscillation < 0.1 * mean_norm if mean_norm > 0 else True
        )

        layer_analysis.append({
            "growth_rate": float(growth_rate),
            "oscillation": float(oscillation),
            "stable": bool(stable),
        })

    overall_stable = all(la["stable"] for la in layer_analysis)

    return {
        "overall_stable": overall_stable,
        "layers": layer_analysis,
    }


def compute_representation_redundancy(
    activations: list[torch.Tensor],
) -> list[float]:
    """Mean pairwise cosine similarity between hidden units per layer.

    High values indicate redundant representations (neurons doing the same thing).

    Args:
        activations: List of activation tensors per layer,
            each of shape (n_samples, n_units).

    Returns:
        List of mean cosine similarities, one per layer.
    """
    redundancies = []

    for act in activations:
        if act.dim() == 1:
            act = act.unsqueeze(0)

        # act shape: (n_samples, n_units)
        # We want cosine similarity between unit activation vectors
        # Each unit has an activation vector across samples
        # Transpose to (n_units, n_samples)
        unit_vectors = act.T.float()

        n_units = unit_vectors.shape[0]
        if n_units < 2:
            redundancies.append(0.0)
            continue

        # Normalize each unit vector
        norms = unit_vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = unit_vectors / norms

        # Compute pairwise cosine similarity matrix
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(n_units, n_units, dtype=torch.bool), diagonal=1)
        pairwise_sims = sim_matrix[mask]

        mean_sim = pairwise_sims.mean().item() if pairwise_sims.numel() > 0 else 0.0
        redundancies.append(float(mean_sim))

    return redundancies


def compute_inter_layer_coordination(
    activations: list[torch.Tensor],
) -> list[float]:
    """CKA (Centered Kernel Alignment) between adjacent layer representations.

    CKA measures similarity between representations in different layers.
    High CKA between adjacent layers suggests coordinated learning.

    Args:
        activations: List of activation tensors per layer,
            each of shape (n_samples, n_units).

    Returns:
        List of CKA values for (layer_0, layer_1), (layer_1, layer_2), etc.
    """
    if len(activations) < 2:
        return []

    cka_values = []
    for i in range(len(activations) - 1):
        X = activations[i].float()
        Y = activations[i + 1].float()

        if X.dim() == 1:
            X = X.unsqueeze(0)
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)

        cka = _linear_cka(X, Y)
        cka_values.append(float(cka))

    return cka_values


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (n_samples, n_features_x) representation matrix.
        Y: (n_samples, n_features_y) representation matrix.

    Returns:
        CKA value in [0, 1].
    """
    # Center the representations
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    # For linear kernel: HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2
    n = X.shape[0]
    if n < 2:
        return 0.0

    # Cross-covariance
    YtX = Y.T @ X  # (features_y, features_x)
    hsic_xy = (YtX * YtX).sum().item()

    # Self-covariance
    XtX = X.T @ X
    hsic_xx = (XtX * XtX).sum().item()

    YtY = Y.T @ Y
    hsic_yy = (YtY * YtY).sum().item()

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denom = np.sqrt(max(hsic_xx * hsic_yy, 0.0))
    if denom < 1e-12:
        return 0.0
    cka = hsic_xy / denom

    return float(np.clip(cka, 0.0, 1.0))


def generate_credit_assignment_heatmap(
    credit_data: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Generate credit assignment heatmap (rules × layers).

    Args:
        credit_data: Dict mapping rule_name -> list of correlations per layer.
        output_path: Path to save the PNG.
    """
    if not HAS_MATPLOTLIB or not credit_data:
        return

    rules = sorted(credit_data.keys())
    n_layers = max(len(v) for v in credit_data.values())

    # Build matrix
    matrix = np.zeros((len(rules), n_layers))
    for i, rule in enumerate(rules):
        for j, val in enumerate(credit_data[rule]):
            matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Learning Rule")
    ax.set_title("Credit Assignment Reach\n(Correlation with True Gradient)")

    # Add text annotations
    for i in range(len(rules)):
        for j in range(n_layers):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="black" if abs(matrix[i, j]) < 0.5 else "white",
            )

    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_deficiency_report(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate deficiency_analysis.md report.

    Args:
        results: Dict mapping rule_name -> analysis results.
        output_path: Path to write the markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Deficiency Analysis Report",
        "",
        "## Summary",
        "",
        "This report characterizes what each local learning rule lacks "
        "compared to backpropagation, identifying specific deficiencies "
        "that the astrocyte gate (Step 13) should address.",
        "",
    ]

    for rule_name, data in sorted(results.items()):
        lines.append(f"## {rule_name}")
        lines.append("")

        # Credit assignment
        credit = data.get("credit_assignment", [])
        if credit:
            lines.append("### Credit Assignment Reach")
            lines.append("")
            for i, c in enumerate(credit):
                lines.append(f"- Layer {i}: {c:.3f}")
            lines.append("")

        # Stability
        stability = data.get("stability", {})
        if stability:
            lines.append("### Weight Stability")
            lines.append(f"- Overall stable: {stability.get('overall_stable', 'N/A')}")
            for i, layer in enumerate(stability.get("layers", [])):
                lines.append(
                    f"- Layer {i}: growth={layer['growth_rate']:.2f}, "
                    f"oscillation={layer['oscillation']:.4f}, "
                    f"stable={layer['stable']}"
                )
            lines.append("")

        # Redundancy
        redundancy = data.get("redundancy", [])
        if redundancy:
            lines.append("### Representation Redundancy")
            for i, r in enumerate(redundancy):
                lines.append(f"- Layer {i}: {r:.3f}")
            lines.append("")

        # Coordination
        coordination = data.get("coordination", [])
        if coordination:
            lines.append("### Inter-Layer Coordination (CKA)")
            for i, c in enumerate(coordination):
                lines.append(f"- Layers {i}-{i+1}: {c:.3f}")
            lines.append("")

        # Dominant deficiency
        dominant = data.get("dominant_deficiency", "unknown")
        intervention = data.get("recommended_intervention", "unknown")
        lines.append(f"### Assessment")
        lines.append(f"- **Dominant deficiency**: {dominant}")
        lines.append(f"- **Recommended intervention**: {intervention}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_full_deficiency_analysis(
    model: nn.Module,
    rule,
    rule_name: str,
    data_loader: DataLoader,
    weight_norm_history: list[list[float]] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run complete deficiency analysis for one rule.

    Args:
        model: Trained model.
        rule: The learning rule instance.
        rule_name: Name of the rule.
        data_loader: Data loader for analysis batches.
        weight_norm_history: Optional weight norm history for stability analysis.
        device: Device.

    Returns:
        Dict with all analysis results.
    """
    if device is None:
        device = next(model.parameters()).device

    # Get a batch for analysis
    batch_iter = iter(data_loader)
    images, labels = next(batch_iter)
    x = images.to(device).view(images.size(0), -1)
    labels = labels.to(device)

    # Credit assignment
    credit = compute_credit_assignment_reach(model, rule, x, labels, device)

    # Representation analysis
    with torch.no_grad():
        activations = model.get_layer_activations(x)

    redundancy = compute_representation_redundancy(activations)
    coordination = compute_inter_layer_coordination(activations)

    # Weight stability
    stability = {}
    if weight_norm_history:
        stability = compute_weight_stability(weight_norm_history)

    # Determine dominant deficiency
    mean_credit = np.mean(credit) if credit else 0.0
    mean_redundancy = np.mean(redundancy) if redundancy else 0.0

    if mean_credit < 0.1:
        dominant = "credit_assignment"
        intervention = "Third-factor signal carrying error information to early layers"
    elif not stability.get("overall_stable", True):
        dominant = "weight_stability"
        intervention = "Homeostatic regulation of weight magnitudes"
    elif mean_redundancy > 0.5:
        dominant = "redundancy"
        intervention = "Decorrelation signal to diversify representations"
    else:
        dominant = "coordination"
        intervention = "Inter-layer synchronization signal"

    return {
        "rule_name": rule_name,
        "credit_assignment": credit,
        "stability": stability,
        "redundancy": redundancy,
        "coordination": coordination,
        "dominant_deficiency": dominant,
        "recommended_intervention": intervention,
    }
