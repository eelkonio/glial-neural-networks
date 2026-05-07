"""Experiment conditions for Predictive Coding + BCM comparison.

Defines 6 experimental conditions and provides functions to run
individual conditions and collect results.

Conditions:
  1. predictive_bcm_full      — BCM direction + prediction error + D-serine + competition
  2. predictive_bcm_no_astrocyte — BCM direction + prediction error only
  3. predictive_only           — Prediction error as sole direction (BCM theta disabled)
  4. bcm_only                  — BCM without prediction error (Step 12b baseline)
  5. predictive_neuron_level   — Neuron-level prediction (128×128) for comparison
  6. backprop                  — Standard backpropagation upper bound
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

_step14_dir = str(Path(__file__).parent.parent)
if _step14_dir not in sys.path:
    sys.path.insert(0, _step14_dir)

from code.step_imports import (
    LocalMLP, get_fashion_mnist_loaders, CalciumConfig, DomainConfig,
    ThreeFactorRule, GlobalRewardThirdFactor,
)
from code.predictive_bcm_config import PredictiveBCMConfig
from code.training import setup_predictive_bcm_rule, train_epoch_predictive, evaluate


@dataclass
class ExperimentCondition:
    """A single experimental condition.

    Attributes:
        name: Identifier for this condition.
        config: PredictiveBCMConfig (None for backprop condition).
        calcium_config: Calcium dynamics parameters.
        domain_config: Domain assignment parameters.
        is_backprop: If True, use standard backprop instead of local rules.
        is_bcm_only: If True, this is the BCM-only baseline (no prediction).
        description: Human-readable description of the condition.
    """

    name: str
    config: PredictiveBCMConfig | None = None
    calcium_config: CalciumConfig = field(default_factory=CalciumConfig)
    domain_config: DomainConfig = field(default_factory=lambda: DomainConfig(domain_size=16))
    is_backprop: bool = False
    is_bcm_only: bool = False
    description: str = ""


def get_all_conditions() -> list[ExperimentCondition]:
    """Return all 6 experiment conditions."""
    return [
        ExperimentCondition(
            name="predictive_bcm_full",
            config=PredictiveBCMConfig(
                lr=0.01, lr_pred=0.01,
                clip_pred_delta=0.1,
                use_d_serine=True, use_competition=True, use_domain_modulation=True,
            ),
            description="BCM direction + domain prediction error + D-serine + competition",
        ),
        ExperimentCondition(
            name="predictive_bcm_no_astrocyte",
            config=PredictiveBCMConfig(
                lr=0.01, lr_pred=0.01,
                clip_pred_delta=0.1,
                use_d_serine=False, use_competition=False, use_domain_modulation=False,
            ),
            description="BCM direction + domain prediction error only",
        ),
        ExperimentCondition(
            name="predictive_only",
            config=PredictiveBCMConfig(
                lr=0.01, lr_pred=0.01,
                clip_pred_delta=0.1,
                theta_init=0.0, theta_decay=1.0,
                use_d_serine=False, use_competition=False, use_domain_modulation=True,
            ),
            description="Domain prediction error as sole direction (BCM theta disabled)",
        ),
        ExperimentCondition(
            name="bcm_only",
            config=PredictiveBCMConfig(
                lr=0.001, lr_pred=0.0,
                clip_pred_delta=0.1,
                use_d_serine=True, use_competition=True, use_domain_modulation=False,
                learn_predictions=False, fixed_predictions=True,
            ),
            is_bcm_only=True,
            description="BCM without prediction error (Step 12b baseline)",
        ),
        ExperimentCondition(
            name="predictive_neuron_level",
            config=PredictiveBCMConfig(
                lr=0.01, lr_pred=0.01,
                clip_pred_delta=0.1,
                granularity="neuron",
                use_d_serine=True, use_competition=True, use_domain_modulation=True,
            ),
            description="Neuron-level prediction (128x128) for comparison",
        ),
        ExperimentCondition(
            name="backprop",
            is_backprop=True,
            description="Standard backpropagation upper bound",
        ),
    ]


def run_condition(
    condition: ExperimentCondition,
    n_epochs: int = 50,
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
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
        Dict with condition name, seed, final accuracy, epoch results.
    """
    torch.manual_seed(seed)
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)
    model = LocalMLP().to(device)
    layer_sizes = [(784, 128), (128, 128), (128, 128), (128, 128), (128, 10)]

    epoch_results = []

    if condition.is_backprop:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
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
            train_loss = total_loss / max(n_batches, 1)
            metrics = evaluate(model, test_loader, device)
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_accuracy": metrics["test_accuracy"],
                "test_loss": metrics["test_loss"],
                "prediction_errors": {},
            }
            epoch_results.append(result)
            if verbose:
                print(f"    Epoch {epoch}: loss={train_loss:.4f}, acc={metrics['test_accuracy']:.4f}")
    else:
        rule = setup_predictive_bcm_rule(
            condition.config, condition.domain_config, condition.calcium_config,
            layer_sizes, device,
        )
        for epoch in range(n_epochs):
            train_result = train_epoch_predictive(model, rule, train_loader, device)
            metrics = evaluate(model, test_loader, device)
            has_nan = any(
                torch.isnan(l.weight.data).any() or torch.isinf(l.weight.data).any()
                for l in model.layers
            )
            result = {
                "epoch": epoch,
                "train_loss": train_result["train_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "test_loss": metrics["test_loss"],
                "prediction_errors": train_result["prediction_errors"],
                "has_nan": has_nan,
            }
            epoch_results.append(result)
            if verbose:
                pred_err_str = ", ".join(
                    f"L{k.split('_')[1]}={v:.4f}"
                    for k, v in train_result["prediction_errors"].items()
                )
                print(
                    f"    Epoch {epoch}: loss={train_result['train_loss']:.4f}, "
                    f"acc={metrics['test_accuracy']:.4f}, pred_err=[{pred_err_str}]"
                )
            rule.reset()

    return {
        "condition": condition.name,
        "seed": seed,
        "n_epochs": n_epochs,
        "epoch_results": epoch_results,
        "final_accuracy": epoch_results[-1]["test_accuracy"] if epoch_results else 0.0,
        "any_nan": any(r.get("has_nan", False) for r in epoch_results),
    }


def run_experiment(
    conditions: list[ExperimentCondition] | None = None,
    seeds: list[int] | None = None,
    n_epochs: int = 50,
    batch_size: int = 128,
    device: str = "cpu",
    verbose: bool = True,
) -> list[dict]:
    """Run full experiment across conditions and seeds.

    Args:
        conditions: List of conditions (defaults to all 6).
        seeds: List of random seeds (defaults to [42, 123, 456]).
        n_epochs: Number of training epochs per run.
        batch_size: Batch size for data loaders.
        device: Torch device.
        verbose: If True, print progress.

    Returns:
        List of result dicts, one per (condition, seed) pair.
    """
    if conditions is None:
        conditions = get_all_conditions()
    if seeds is None:
        seeds = [42, 123, 456]

    all_results = []
    for condition in conditions:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"CONDITION: {condition.name} — {condition.description}")
            print(f"{'─' * 60}")
        for seed in seeds:
            if verbose:
                print(f"\n  Seed {seed}:")
            result = run_condition(
                condition=condition,
                n_epochs=n_epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                verbose=verbose,
            )
            all_results.append(result)
            if verbose:
                print(f"  → Final accuracy: {result['final_accuracy']:.4f}")

    return all_results
