"""Spatial embedding quality measurement under local learning rules.

Measures whether spatial structure is more meaningful under local rules
than under backpropagation, by correlating spatial distances with
update-signal correlations (instead of gradient correlations).

Reuses the spectral embedding from Phase 1 but computes correlations
using local update signals.
"""

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Add Phase 1 to path for spectral embedding import
_phase1_dir = str(Path(__file__).parent.parent.parent.parent / "01-spatial-embedding")
if _phase1_dir not in sys.path:
    sys.path.insert(0, _phase1_dir)


def _get_weight_positions(model: nn.Module) -> np.ndarray:
    """Compute simple spatial positions for weights based on layer structure.

    Uses a simplified layout: weights are positioned based on their
    layer index (x), source neuron (y), and target neuron (z),
    normalized to [0, 1].

    This avoids the full spectral embedding dependency while still
    providing meaningful spatial structure for correlation analysis.

    Args:
        model: LocalMLP model.

    Returns:
        ndarray of shape (N_weights, 3) with positions in [0, 1].
    """
    positions = []
    n_layers = len(model.layers)

    for layer_idx, layer in enumerate(model.layers):
        weight = layer.linear.weight.data
        out_features, in_features = weight.shape

        # x: layer position normalized
        x_pos = layer_idx / max(n_layers - 1, 1)

        for out_idx in range(out_features):
            for in_idx in range(in_features):
                y_pos = in_idx / max(in_features - 1, 1)
                z_pos = out_idx / max(out_features - 1, 1)
                positions.append([x_pos, y_pos, z_pos])

    return np.array(positions, dtype=np.float64)


def compute_update_signal_correlations(
    model: nn.Module,
    rule,
    data_loader: DataLoader,
    n_batches: int = 20,
    max_pairs: int = 50000,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise correlations between local update signals.

    For each batch, collects the flattened update signal (all weight deltas
    concatenated). Then computes pairwise Pearson correlations between
    weight positions across batches.

    Args:
        model: The LocalMLP model.
        rule: A local learning rule with compute_update method.
        data_loader: Data loader for computing updates.
        n_batches: Number of batches to collect.
        max_pairs: Maximum number of pairs to sample.
        device: Device.

    Returns:
        Tuple of (spatial_distances, update_correlations) arrays.
    """
    if device is None:
        device = next(model.parameters()).device

    # Collect update signals across batches
    all_updates = []
    model.eval()

    batch_count = 0
    for images, labels in data_loader:
        if batch_count >= n_batches:
            break

        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images, detach=True)
            loss = nn.CrossEntropyLoss()(logits, labels)
            states = model.forward_with_states(images, labels=labels, global_loss=loss.item())

            # Collect flattened update signal
            try:
                updates = []
                for state in states:
                    delta = rule.compute_update(state)
                    updates.append(delta.flatten())

                flat_update = torch.cat(updates).cpu().numpy()
                all_updates.append(flat_update)
            except (NotImplementedError, RuntimeError):
                # Rule doesn't support compute_update (e.g., ForwardForward, PredictiveCoding)
                model.train()
                return np.array([]), np.array([])

        batch_count += 1

    model.train()

    if len(all_updates) < 2:
        return np.array([]), np.array([])

    # Stack: (n_batches, n_weights)
    update_matrix = np.stack(all_updates, axis=0)
    n_weights = update_matrix.shape[1]

    # Get spatial positions
    positions = _get_weight_positions(model)
    assert positions.shape[0] == n_weights, (
        f"Position count {positions.shape[0]} != weight count {n_weights}"
    )

    # Sample pairs
    rng = np.random.default_rng(42)
    total_pairs = n_weights * (n_weights - 1) // 2

    if total_pairs > max_pairs:
        # Random subsampling
        idx_i = rng.integers(0, n_weights, size=max_pairs)
        idx_j = rng.integers(0, n_weights, size=max_pairs)
        # Ensure i != j
        mask = idx_i == idx_j
        while mask.any():
            idx_j[mask] = rng.integers(0, n_weights, size=mask.sum())
            mask = idx_i == idx_j
    else:
        idx_i, idx_j = np.triu_indices(n_weights, k=1)

    # Compute spatial distances
    diff = positions[idx_i] - positions[idx_j]
    spatial_distances = np.sqrt(np.sum(diff**2, axis=1))

    # Compute update signal correlations per pair
    # For each pair (i, j), correlate their update signals across batches
    signals_i = update_matrix[:, idx_i]  # (n_batches, n_pairs)
    signals_j = update_matrix[:, idx_j]  # (n_batches, n_pairs)

    # Vectorized Pearson correlation across batches for each pair
    n = signals_i.shape[0]
    mean_i = signals_i.mean(axis=0)
    mean_j = signals_j.mean(axis=0)

    centered_i = signals_i - mean_i
    centered_j = signals_j - mean_j

    numerator = (centered_i * centered_j).sum(axis=0)
    denom_i = np.sqrt((centered_i**2).sum(axis=0))
    denom_j = np.sqrt((centered_j**2).sum(axis=0))
    denom = denom_i * denom_j

    with np.errstate(divide="ignore", invalid="ignore"):
        update_correlations = np.where(denom > 1e-12, numerator / denom, 0.0)

    np.clip(update_correlations, -1.0, 1.0, out=update_correlations)

    return spatial_distances, update_correlations


def compute_spatial_quality(
    model: nn.Module,
    rule,
    data_loader: DataLoader,
    n_batches: int = 20,
    max_pairs: int = 50000,
    device: torch.device | None = None,
) -> float:
    """Compute Pearson correlation between spatial distances and update correlations.

    Args:
        model: The LocalMLP model.
        rule: A local learning rule.
        data_loader: Data loader.
        n_batches: Number of batches.
        max_pairs: Maximum pairs to sample.
        device: Device.

    Returns:
        Pearson correlation coefficient.
    """
    spatial_distances, update_correlations = compute_update_signal_correlations(
        model, rule, data_loader, n_batches, max_pairs, device
    )

    if len(spatial_distances) < 10:
        return 0.0

    # Filter out degenerate cases
    if np.std(spatial_distances) < 1e-12 or np.std(update_correlations) < 1e-12:
        return 0.0

    corr, _ = pearsonr(spatial_distances, update_correlations)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_backprop_spatial_quality(
    model: nn.Module,
    data_loader: DataLoader,
    n_batches: int = 20,
    max_pairs: int = 50000,
    device: torch.device | None = None,
) -> float:
    """Compute spatial quality using backprop gradients as reference.

    Args:
        model: The LocalMLP model.
        data_loader: Data loader.
        n_batches: Number of batches.
        max_pairs: Maximum pairs.
        device: Device.

    Returns:
        Pearson correlation between spatial distances and gradient correlations.
    """
    if device is None:
        device = next(model.parameters()).device

    # Collect gradient signals across batches
    all_gradients = []
    criterion = nn.CrossEntropyLoss()

    batch_count = 0
    for images, labels in data_loader:
        if batch_count >= n_batches:
            break

        images = images.to(device).view(images.size(0), -1)
        labels = labels.to(device)

        model.zero_grad()
        logits = model(images, detach=False)
        loss = criterion(logits, labels)
        loss.backward()

        # Collect flattened gradient
        grads = []
        for layer in model.layers:
            if layer.linear.weight.grad is not None:
                grads.append(layer.linear.weight.grad.flatten().detach().cpu().numpy())
            else:
                grads.append(np.zeros(layer.linear.weight.numel()))

        flat_grad = np.concatenate(grads)
        all_gradients.append(flat_grad)
        batch_count += 1

    model.zero_grad()

    if len(all_gradients) < 2:
        return 0.0

    grad_matrix = np.stack(all_gradients, axis=0)
    n_weights = grad_matrix.shape[1]

    positions = _get_weight_positions(model)

    # Sample pairs
    rng = np.random.default_rng(42)
    total_pairs = n_weights * (n_weights - 1) // 2

    if total_pairs > max_pairs:
        idx_i = rng.integers(0, n_weights, size=max_pairs)
        idx_j = rng.integers(0, n_weights, size=max_pairs)
        mask = idx_i == idx_j
        while mask.any():
            idx_j[mask] = rng.integers(0, n_weights, size=mask.sum())
            mask = idx_i == idx_j
    else:
        idx_i, idx_j = np.triu_indices(n_weights, k=1)

    # Spatial distances
    diff = positions[idx_i] - positions[idx_j]
    spatial_distances = np.sqrt(np.sum(diff**2, axis=1))

    # Gradient correlations
    signals_i = grad_matrix[:, idx_i]
    signals_j = grad_matrix[:, idx_j]

    mean_i = signals_i.mean(axis=0)
    mean_j = signals_j.mean(axis=0)
    centered_i = signals_i - mean_i
    centered_j = signals_j - mean_j

    numerator = (centered_i * centered_j).sum(axis=0)
    denom_i = np.sqrt((centered_i**2).sum(axis=0))
    denom_j = np.sqrt((centered_j**2).sum(axis=0))
    denom = denom_i * denom_j

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_correlations = np.where(denom > 1e-12, numerator / denom, 0.0)

    np.clip(grad_correlations, -1.0, 1.0, out=grad_correlations)

    if np.std(spatial_distances) < 1e-12 or np.std(grad_correlations) < 1e-12:
        return 0.0

    corr, _ = pearsonr(spatial_distances, grad_correlations)
    return float(corr) if not np.isnan(corr) else 0.0


def save_spatial_quality_results(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    """Save spatial quality results to CSV.

    Args:
        results: List of dicts with rule, spatial_correlation,
            backprop_correlation, ratio.
        output_path: Path to write CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["rule", "spatial_correlation", "backprop_correlation", "ratio"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
