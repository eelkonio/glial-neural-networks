"""Training loop integrating gate variants with ThreeFactorRule.

Wires astrocyte gates into Step 12's ThreeFactorRule as the third_factor.
Applies stability fix (error clipping, eligibility normalization).
Supports calcium state checkpointing every N epochs.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch import Tensor

from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig
from code.gates.binary_gate import BinaryGate
from code.gates.directional_gate import DirectionalGate
from code.gates.volume_teaching import VolumeTeachingGate
from code.experiment.config import GateConfig, ExperimentCondition
from code.stability import clip_error_signal, normalize_eligibility
from code.step12_imports import (
    ThreeFactorRule,
    LayerState,
    LocalMLP,
    get_fashion_mnist_loaders,
)


def create_gate(
    gate_config: GateConfig,
    domain_assignment: DomainAssignment,
    calcium_config: CalciumConfig,
    device: str = "cpu",
):
    """Create a gate instance from configuration.

    Args:
        gate_config: Gate variant and parameters.
        domain_assignment: Domain assignment for the network.
        calcium_config: Calcium dynamics parameters.
        device: Torch device.

    Returns:
        Gate instance implementing ThirdFactorInterface.
    """
    if gate_config.variant == "binary":
        return BinaryGate(
            domain_assignment=domain_assignment,
            calcium_config=calcium_config,
            device=device,
        )
    elif gate_config.variant == "directional":
        return DirectionalGate(
            domain_assignment=domain_assignment,
            calcium_config=calcium_config,
            prediction_decay=gate_config.prediction_decay,
            device=device,
        )
    elif gate_config.variant == "volume_teaching":
        return VolumeTeachingGate(
            domain_assignment=domain_assignment,
            calcium_config=calcium_config,
            diffusion_sigma=gate_config.diffusion_sigma,
            n_classes=gate_config.n_classes,
            gap_junction_strength=gate_config.gap_junction_strength,
            device=device,
        )
    else:
        raise ValueError(f"Unknown gate variant: {gate_config.variant}")


def train_epoch(
    model: LocalMLP,
    rule: ThreeFactorRule,
    gate,
    train_loader,
    condition: ExperimentCondition,
    device: str = "cpu",
) -> dict:
    """Train for one epoch using three-factor rule with gate.

    Args:
        model: LocalMLP network.
        rule: ThreeFactorRule instance.
        gate: Gate instance (or None for baselines).
        train_loader: Training data loader.
        condition: Experiment condition with stability parameters.
        device: Torch device.

    Returns:
        Dict with epoch metrics: train_loss, n_batches, gate_fraction_open.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    gate_open_sum = 0.0
    gate_open_count = 0

    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device).view(data.size(0), -1)  # Flatten to (batch, 784)
        target = target.to(device)

        # Forward pass collecting layer states
        states = model.forward_with_states(data, labels=target)

        # Compute loss for monitoring
        with torch.no_grad():
            logits = states[-1].post_activation
            loss = criterion(logits, target)
            total_loss += loss.item()

        # Update global_loss in states
        for state in states:
            state.global_loss = loss.item()

        # Apply three-factor rule to each layer
        for state in states:
            # Compute update via three-factor rule
            delta = rule.compute_update(state)

            # Apply stability fix
            if condition.use_stability_fix:
                delta = clip_error_signal(delta, threshold=condition.error_clip_threshold)
                delta = normalize_eligibility(
                    delta,
                    norm_threshold=condition.eligibility_norm_threshold,
                    safe_constant=1.0,
                )

            # Apply weight update
            with torch.no_grad():
                model.layers[state.layer_index].weight.data += delta

            # Track gate statistics
            if gate is not None and hasattr(gate, '_calcium'):
                layer_idx = state.layer_index
                if layer_idx in gate._calcium:
                    ca = gate._calcium[layer_idx]
                    gate_open_sum += ca.get_gate_open().float().mean().item()
                    gate_open_count += 1

        n_batches += 1

    metrics = {
        "train_loss": total_loss / max(n_batches, 1),
        "n_batches": n_batches,
        "gate_fraction_open": gate_open_sum / max(gate_open_count, 1),
    }
    return metrics


def evaluate(model: LocalMLP, test_loader, device: str = "cpu") -> dict:
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
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).view(data.size(0), -1)
            target = target.to(device)

            # Standard forward pass for evaluation (no detach for inference)
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


def train_with_gate(
    condition: ExperimentCondition,
    n_epochs: int = 5,
    batch_size: int = 128,
    device: str = "cpu",
    checkpoint_interval: int = 10,
    checkpoint_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Full training run with a gate variant.

    Args:
        condition: Experiment condition specifying gate, calcium, domain params.
        n_epochs: Number of training epochs.
        batch_size: Batch size for data loading.
        device: Torch device.
        checkpoint_interval: Save calcium state every N epochs.
        checkpoint_dir: Directory for checkpoints. None = no checkpointing.
        verbose: Print progress.

    Returns:
        Dict with per-epoch results and final metrics.
    """
    if verbose:
        print(f"  Training: {condition.name}")
        print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Load data
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    # Create model (784 → 128 → 128 → 128 → 128 → 10)
    model = LocalMLP(
        input_size=784,
        hidden_size=128,
        n_classes=10,
    ).to(device)

    # Create domain assignment
    layer_sizes = []
    prev_size = 784
    for layer in model.layers:
        out_size = layer.linear.out_features
        layer_sizes.append((prev_size, out_size))
        prev_size = out_size

    domain_assignment = DomainAssignment(
        layer_sizes=layer_sizes,
        config=condition.domain_config,
    )

    # Create gate
    gate = None
    if condition.gate_config is not None:
        gate = create_gate(
            gate_config=condition.gate_config,
            domain_assignment=domain_assignment,
            calcium_config=condition.calcium_config,
            device=device,
        )

    # Create three-factor rule with gate as third factor
    rule = ThreeFactorRule(
        lr=condition.learning_rate,
        tau=condition.tau,
        third_factor=gate,
    )

    # Training loop
    epoch_results = []
    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            rule=rule,
            gate=gate,
            train_loader=train_loader,
            condition=condition,
            device=device,
        )

        # Evaluate
        eval_metrics = evaluate(model, test_loader, device=device)

        # Check for NaN/Inf
        has_nan = False
        for layer in model.layers:
            if torch.isnan(layer.weight.data).any() or torch.isinf(layer.weight.data).any():
                has_nan = True
                break

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "test_accuracy": eval_metrics["test_accuracy"],
            "test_loss": eval_metrics["test_loss"],
            "gate_fraction_open": train_metrics["gate_fraction_open"],
            "has_nan": has_nan,
        }
        epoch_results.append(epoch_result)

        if verbose:
            print(f"    Epoch {epoch}: loss={train_metrics['train_loss']:.4f}, "
                  f"acc={eval_metrics['test_accuracy']:.4f}, "
                  f"gate_open={train_metrics['gate_fraction_open']:.3f}, "
                  f"nan={has_nan}")

        # Checkpoint calcium state
        if checkpoint_dir and gate and (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"{condition.name}_epoch{epoch+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(gate.state_dict(), ckpt_path)

        # Reset eligibility traces between epochs
        rule.reset()

    if verbose:
        print(f"  Done: {condition.name}")
        print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")

    return {
        "condition": condition.name,
        "epoch_results": epoch_results,
        "final_accuracy": epoch_results[-1]["test_accuracy"] if epoch_results else 0.0,
        "any_nan": any(r["has_nan"] for r in epoch_results),
    }
