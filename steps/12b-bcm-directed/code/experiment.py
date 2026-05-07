"""Experiment runner for BCM-directed learning rule comparison.

Defines 5 experimental conditions and provides functions to run
individual conditions and collect results.

Conditions:
  1. bcm_no_astrocyte  — BCM direction only, no D-serine, no competition
  2. bcm_d_serine      — BCM + D-serine boost, no competition
  3. bcm_full          — BCM + D-serine + heterosynaptic competition
  4. three_factor_reward — Step 12 baseline (global reward third factor)
  5. backprop          — Upper bound (standard SGD)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_step12b_dir = str(Path(__file__).parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

from code.step_imports import (
    LocalMLP,
    get_fashion_mnist_loaders,
    ThreeFactorRule,
    GlobalRewardThirdFactor,
    CalciumConfig,
    DomainConfig,
)
from code.bcm_config import BCMConfig
from code.training import setup_bcm_rule, train_epoch, evaluate


@dataclass
class ExperimentCondition:
    """A single experimental condition.

    Attributes:
        name: Identifier for this condition.
        bcm_config: BCM rule parameters (None for non-BCM conditions).
        calcium_config: Calcium dynamics parameters.
        domain_config: Domain assignment parameters.
        use_backprop: If True, use standard backprop instead of local rules.
        use_three_factor: If True, use ThreeFactorRule with GlobalReward.
        description: Human-readable description of the condition.
    """

    name: str
    bcm_config: BCMConfig | None = None
    calcium_config: CalciumConfig = field(default_factory=CalciumConfig)
    domain_config: DomainConfig = field(default_factory=lambda: DomainConfig(domain_size=16))
    use_backprop: bool = False
    use_three_factor: bool = False
    description: str = ""


def get_bcm_no_astrocyte() -> ExperimentCondition:
    """BCM direction only — no D-serine gating, no heterosynaptic competition.

    This isolates the contribution of the BCM sliding threshold mechanism
    without any astrocyte-mediated modulation.
    """
    return ExperimentCondition(
        name="bcm_no_astrocyte",
        bcm_config=BCMConfig(
            lr=0.01,
            theta_decay=0.99,
            theta_init=0.1,
            d_serine_boost=1.0,
            competition_strength=1.0,
            clip_delta=1.0,
            use_d_serine=False,
            use_competition=False,
        ),
        description="BCM direction only, no D-serine, no competition",
    )


def get_bcm_d_serine() -> ExperimentCondition:
    """BCM + D-serine boost — no heterosynaptic competition.

    Tests whether D-serine gating alone (enabling LTP in active domains)
    improves over bare BCM direction.
    """
    return ExperimentCondition(
        name="bcm_d_serine",
        bcm_config=BCMConfig(
            lr=0.01,
            theta_decay=0.99,
            theta_init=0.1,
            d_serine_boost=1.0,
            competition_strength=1.0,
            clip_delta=1.0,
            use_d_serine=True,
            use_competition=False,
        ),
        description="BCM + D-serine boost, no competition",
    )


def get_bcm_full() -> ExperimentCondition:
    """BCM + D-serine + heterosynaptic competition (full model).

    The complete biologically-inspired rule with all components active.
    """
    return ExperimentCondition(
        name="bcm_full",
        bcm_config=BCMConfig(
            lr=0.01,
            theta_decay=0.99,
            theta_init=0.1,
            d_serine_boost=1.0,
            competition_strength=1.0,
            clip_delta=1.0,
            use_d_serine=True,
            use_competition=True,
        ),
        description="BCM + D-serine + heterosynaptic competition (full)",
    )


def get_three_factor_reward() -> ExperimentCondition:
    """Step 12 baseline — three-factor rule with global reward signal.

    This is the baseline from Step 12 that achieves ~10% (chance level)
    because the eligibility trace is always positive under ReLU.
    """
    return ExperimentCondition(
        name="three_factor_reward",
        use_three_factor=True,
        description="Step 12 baseline: three-factor with global reward (~10%)",
    )


def get_backprop() -> ExperimentCondition:
    """Standard backpropagation — upper bound reference.

    Uses Adam optimizer with standard cross-entropy loss.
    Provides the upper bound on what the network architecture can achieve.
    """
    return ExperimentCondition(
        name="backprop",
        use_backprop=True,
        description="Standard backprop (upper bound)",
    )


def get_all_conditions() -> list[ExperimentCondition]:
    """Return all 5 experimental conditions in order."""
    return [
        get_bcm_no_astrocyte(),
        get_bcm_d_serine(),
        get_bcm_full(),
        get_three_factor_reward(),
        get_backprop(),
    ]


def _train_epoch_backprop(
    model: LocalMLP,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Train one epoch with standard backpropagation.

    Args:
        model: LocalMLP network.
        optimizer: Torch optimizer (e.g., Adam).
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
        data = data.to(device).view(data.size(0), -1)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data, detach=False)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _train_epoch_three_factor(
    model: LocalMLP,
    rule: ThreeFactorRule,
    train_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Train one epoch with three-factor rule (global reward).

    Args:
        model: LocalMLP network.
        rule: ThreeFactorRule instance.
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
        data = data.to(device).view(data.size(0), -1)
        target = target.to(device)

        # Forward pass collecting layer states
        states = model.forward_with_states(data, labels=target)

        # Compute loss for monitoring and reward signal
        with torch.no_grad():
            logits = states[-1].post_activation
            loss = criterion(logits, target)
            total_loss += loss.item()

        # Set global_loss on states for the reward signal
        for state in states:
            state.global_loss = loss.item()

        # Apply three-factor rule to each layer
        for state in states:
            delta_w = rule.compute_update(state)
            with torch.no_grad():
                model.layers[state.layer_index].weight.data += delta_w

        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_condition(
    condition: ExperimentCondition,
    n_epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Run a single experimental condition.

    Args:
        condition: The experiment condition to run.
        n_epochs: Number of training epochs.
        batch_size: Batch size for data loaders.
        seed: Random seed for reproducibility.
        device: Torch device.
        verbose: If True, print per-epoch progress.

    Returns:
        Dict with condition name, per-epoch metrics, and final results.
    """
    torch.manual_seed(seed)

    # Setup model
    model = LocalMLP()
    layer_sizes = [(784, 128), (128, 128), (128, 128), (128, 128), (128, 10)]

    # Load data
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    # Setup rule/optimizer based on condition
    optimizer = None
    bcm_rule = None
    three_factor_rule = None

    if condition.use_backprop:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elif condition.use_three_factor:
        third_factor = GlobalRewardThirdFactor(baseline_decay=0.99)
        three_factor_rule = ThreeFactorRule(lr=0.01, tau=100.0, third_factor=third_factor)
    else:
        bcm_rule = setup_bcm_rule(
            condition.bcm_config,
            condition.domain_config,
            condition.calcium_config,
            layer_sizes,
            device,
        )

    # Training loop
    epoch_results = []

    for epoch in range(n_epochs):
        # Train
        if condition.use_backprop:
            train_loss = _train_epoch_backprop(model, optimizer, train_loader, device)
        elif condition.use_three_factor:
            train_loss = _train_epoch_three_factor(
                model, three_factor_rule, train_loader, device
            )
        else:
            train_loss = train_epoch(model, bcm_rule, train_loader, device)

        # Evaluate
        metrics = evaluate(model, test_loader, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"],
        }
        epoch_results.append(epoch_result)

        if verbose:
            print(f"    Epoch {epoch:2d}: loss={train_loss:.4f}, "
                  f"acc={metrics['test_accuracy']:.4f}")

    # Final metrics
    final_accuracy = epoch_results[-1]["test_accuracy"] if epoch_results else 0.0
    final_loss = epoch_results[-1]["test_loss"] if epoch_results else float("inf")

    return {
        "condition": condition.name,
        "seed": seed,
        "n_epochs": n_epochs,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "epoch_results": epoch_results,
    }
