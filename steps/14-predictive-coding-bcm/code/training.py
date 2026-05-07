"""Training loop for Predictive Coding + BCM learning rule."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_step14_dir = str(Path(__file__).parent.parent)
if _step14_dir not in sys.path:
    sys.path.insert(0, _step14_dir)

from code.step_imports import LocalMLP, DomainAssignment, CalciumDynamics, CalciumConfig, DomainConfig
from code.predictive_bcm_config import PredictiveBCMConfig
from code.predictive_bcm_rule import PredictiveBCMRule


def setup_predictive_bcm_rule(
    config: PredictiveBCMConfig,
    domain_config: DomainConfig,
    calcium_config: CalciumConfig,
    layer_sizes: list[tuple[int, int]],
    device: str = "cpu",
) -> PredictiveBCMRule:
    """Create a fully configured PredictiveBCMRule.

    Args:
        config: PredictiveBCMConfig with learning parameters.
        domain_config: Domain assignment parameters.
        calcium_config: Calcium dynamics parameters.
        layer_sizes: List of (in_features, out_features) per layer.
        device: Torch device.

    Returns:
        Configured PredictiveBCMRule instance.
    """
    domain_assignment = DomainAssignment(layer_sizes, domain_config)
    calcium_dynamics = {}
    for layer_idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
        calcium_dynamics[layer_idx] = CalciumDynamics(
            n_domains=n_domains, config=calcium_config, device=device,
        )
    return PredictiveBCMRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        layer_sizes=layer_sizes,
        config=config,
        device=device,
    )


def train_epoch_predictive(
    model: LocalMLP,
    rule: PredictiveBCMRule,
    train_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """Train one epoch using PredictiveBCMRule.

    For each batch:
      1. Forward pass with states (collects all LayerStates)
      2. Compute cross-entropy loss (for monitoring only)
      3. Compute weight updates for ALL layers via compute_all_updates
      4. Apply weight deltas to model weights

    Args:
        model: LocalMLP network.
        rule: PredictiveBCMRule instance.
        train_loader: Training data loader.
        device: Torch device.

    Returns:
        Dict with 'train_loss' and 'prediction_errors' (per-layer mean abs).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()
    pred_error_accum = {}

    for data, target in train_loader:
        data = data.to(device).view(data.size(0), -1)
        target = target.to(device)

        # Forward pass collecting ALL layer states
        states = model.forward_with_states(data, labels=target)

        # Compute loss for monitoring only
        with torch.no_grad():
            logits = states[-1].post_activation
            loss = criterion(logits, target)
            total_loss += loss.item()

        # Compute updates for ALL layers simultaneously
        deltas = rule.compute_all_updates(states)

        # Apply weight deltas
        with torch.no_grad():
            for layer_idx, delta_w in enumerate(deltas):
                model.layers[layer_idx].weight.data += delta_w

        # Accumulate prediction errors for monitoring
        errors = rule.get_prediction_errors()
        for idx, err in errors.items():
            if idx not in pred_error_accum:
                pred_error_accum[idx] = 0.0
            pred_error_accum[idx] += err.abs().mean().item()

        n_batches += 1

    # Average prediction errors
    prediction_errors = {
        f"layer_{idx}": val / max(n_batches, 1)
        for idx, val in pred_error_accum.items()
    }

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "prediction_errors": prediction_errors,
    }


def evaluate(model: LocalMLP, test_loader: DataLoader, device: str = "cpu") -> dict:
    """Evaluate model accuracy on test set.

    Args:
        model: LocalMLP network.
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        Dict with test_accuracy and test_loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).view(data.size(0), -1)
            target = target.to(device)
            output = model(data, detach=False)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            n_batches += 1

    return {
        "test_accuracy": correct / max(total, 1),
        "test_loss": total_loss / max(n_batches, 1),
    }
