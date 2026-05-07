"""Training loop for BCM-directed learning rule.

Applies BCMDirectedRule per-layer after forward pass.
Reuses the Step 12 pattern: forward_with_states → compute_update → apply delta.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure step 12b code is importable
_step12b_dir = str(Path(__file__).parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

from code.step_imports import LocalMLP, DomainAssignment, CalciumDynamics, CalciumConfig, DomainConfig
from code.bcm_config import BCMConfig
from code.bcm_rule import BCMDirectedRule


def setup_bcm_rule(
    bcm_config: BCMConfig,
    domain_config: DomainConfig,
    calcium_config: CalciumConfig,
    layer_sizes: list[tuple[int, int]],
    device: str = "cpu",
) -> BCMDirectedRule:
    """Create a fully configured BCMDirectedRule.

    Args:
        bcm_config: BCM rule parameters.
        domain_config: Domain assignment parameters.
        calcium_config: Calcium dynamics parameters.
        layer_sizes: List of (in_features, out_features) per layer.
        device: Torch device.

    Returns:
        Configured BCMDirectedRule instance.
    """
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    calcium_dynamics = {}
    for layer_idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
        calcium_dynamics[layer_idx] = CalciumDynamics(
            n_domains=n_domains,
            config=calcium_config,
            device=device,
        )

    return BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=bcm_config,
    )


def train_epoch(
    model: LocalMLP,
    rule: BCMDirectedRule,
    train_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Train one epoch using BCM directed rule.

    For each batch:
      1. Forward pass with states (detached between layers)
      2. Compute cross-entropy loss (for monitoring only, not for learning)
      3. For each layer: compute_update → apply to weights

    Args:
        model: LocalMLP network.
        rule: BCMDirectedRule instance.
        train_loader: Training data loader.
        device: Torch device.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    for data, target in train_loader:
        data = data.to(device).view(data.size(0), -1)  # Flatten to (batch, 784)
        target = target.to(device)

        # Forward pass collecting layer states
        states = model.forward_with_states(data, labels=target)

        # Compute loss for monitoring only
        with torch.no_grad():
            logits = states[-1].post_activation
            loss = criterion(logits, target)
            total_loss += loss.item()

        # Apply BCM rule to each layer
        for state in states:
            delta_w = rule.compute_update(state)
            with torch.no_grad():
                model.layers[state.layer_index].weight.data += delta_w

        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: LocalMLP,
    test_loader: DataLoader,
    device: str = "cpu",
) -> dict:
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
