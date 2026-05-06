"""Calcium dynamics ablation experiment.

Tests whether the full Li-Rinzel calcium model is necessary,
or if simpler gating mechanisms achieve similar results.

Four mechanisms (all using directional gate architecture):
(a) Full Li-Rinzel (CalciumConfig default but d_serine_threshold=0.02)
(b) Simple threshold (gate=1 if activity > 0.5, else 0)
(c) Linear EMA (gate = EMA of activity, no threshold)
(d) Random gate with matched sparsity
"""

import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig
from code.experiment.config import GateConfig, ExperimentCondition
from code.experiment.metrics import (
    EpochResult,
    ConditionResult,
    save_epoch_results_csv,
)
from code.experiment.runner import (
    ExperimentRunner,
    set_all_seeds,
    run_condition,
)


# --- Simple alternative gate mechanisms ---


class SimpleThresholdGate:
    """Gate = 1 if mean domain activity > threshold, else 0.

    No calcium dynamics — just a direct activity threshold.
    """

    name = "simple_threshold"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.threshold = threshold
        self.device = device
        # Fake _calcium dict for compatibility with gate_fraction_open tracking
        self._calcium = {}

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # Mean absolute activation per domain
        mean_act = layer_activations.abs().mean(dim=0)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()

        # Simple threshold: gate = 1 if activity > threshold
        gate_open = (domain_activities > self.threshold).float()

        # Map to per-neuron
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        return gate_open[neuron_to_domain]

    def reset(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class LinearEMAGate:
    """Gate = EMA of domain activity (continuous, no threshold).

    Provides a smooth gating signal proportional to recent activity.
    """

    name = "linear_ema"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        decay: float = 0.95,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.decay = decay
        self.device = device
        self._ema: dict[int, Tensor] = {}
        self._calcium = {}  # Compatibility

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # Mean absolute activation per domain
        mean_act = layer_activations.abs().mean(dim=0)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()

        # EMA update
        if layer_index not in self._ema:
            self._ema[layer_index] = domain_activities.clone()
        else:
            self._ema[layer_index] = (
                self.decay * self._ema[layer_index]
                + (1 - self.decay) * domain_activities
            )

        # Gate signal = EMA value (continuous)
        gate_signal = self._ema[layer_index]

        # Map to per-neuron
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        return gate_signal[neuron_to_domain]

    def reset(self) -> None:
        self._ema.clear()

    def state_dict(self) -> dict:
        return {f"ema_{k}": v.clone() for k, v in self._ema.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k.startswith("ema_"):
                idx = int(k.split("_")[1])
                self._ema[idx] = v.to(self.device)


class RandomMatchedGate:
    """Random gate with matched sparsity to Li-Rinzel gate.

    Opens a random fraction of domains each step, matching the
    average fraction open observed in the full calcium model.
    """

    name = "random_matched"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        open_fraction: float = 0.5,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.open_fraction = open_fraction
        self.device = device
        self._calcium = {}  # Compatibility

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # Random gate: each domain open with probability = open_fraction
        gate_open = (torch.rand(n_domains, device=device) < self.open_fraction).float()

        # Map to per-neuron
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        return gate_open[neuron_to_domain]

    def reset(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def get_calcium_ablation_conditions() -> list[ExperimentCondition]:
    """Get the four calcium ablation conditions.

    All use directional gate architecture but with different
    calcium/gating mechanisms.
    """
    # (a) Full Li-Rinzel with d_serine_threshold=0.02
    full_lirinzel = ExperimentCondition(
        name="ablation_full_lirinzel",
        gate_config=GateConfig(variant="directional"),
        calcium_config=CalciumConfig(d_serine_threshold=0.02),
        domain_config=DomainConfig(),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    # (b) Simple threshold — handled via special gate class
    simple_threshold = ExperimentCondition(
        name="ablation_simple_threshold",
        gate_config=None,  # Special handling needed
        calcium_config=CalciumConfig(),
        domain_config=DomainConfig(),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    # (c) Linear EMA — handled via special gate class
    linear_ema = ExperimentCondition(
        name="ablation_linear_ema",
        gate_config=None,  # Special handling needed
        calcium_config=CalciumConfig(),
        domain_config=DomainConfig(),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    # (d) Random matched — handled via special gate class
    random_matched = ExperimentCondition(
        name="ablation_random_matched",
        gate_config=None,  # Special handling needed
        calcium_config=CalciumConfig(),
        domain_config=DomainConfig(),
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )

    return [full_lirinzel, simple_threshold, linear_ema, random_matched]


def run_calcium_ablation(
    n_epochs: int = 50,
    seeds: list[int] = None,
    batch_size: int = 128,
    device: str = "cpu",
    output_dir: str = "results",
    verbose: bool = True,
) -> list[ConditionResult]:
    """Run the calcium dynamics ablation experiment.

    Args:
        n_epochs: Number of epochs per condition.
        seeds: Random seeds.
        batch_size: Batch size.
        device: Torch device.
        output_dir: Output directory for results.
        verbose: Print progress.

    Returns:
        List of ConditionResult.
    """
    from code.experiment.training import train_epoch, evaluate
    from code.step12_imports import ThreeFactorRule, LocalMLP, get_fashion_mnist_loaders
    from code.experiment.runner import compute_weight_norms

    seeds = seeds or [42, 123, 456]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_results = []

    conditions = get_calcium_ablation_conditions()

    for condition in conditions:
        for seed in seeds:
            # UTC timestamp before
            start_time = datetime.now(timezone.utc)
            print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                  f"Starting: {condition.name} seed={seed}")

            t0 = time.time()
            set_all_seeds(seed)

            # For full Li-Rinzel, use standard runner
            if condition.name == "ablation_full_lirinzel":
                result = run_condition(
                    condition=condition,
                    seed=seed,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    device=device,
                    verbose=verbose,
                )
            else:
                # Custom gate mechanisms
                train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)
                model = LocalMLP(input_size=784, hidden_size=128, n_classes=10).to(device)

                # Build domain assignment
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

                # Create appropriate gate
                if condition.name == "ablation_simple_threshold":
                    gate = SimpleThresholdGate(domain_assignment, threshold=0.5, device=device)
                elif condition.name == "ablation_linear_ema":
                    gate = LinearEMAGate(domain_assignment, decay=0.95, device=device)
                elif condition.name == "ablation_random_matched":
                    gate = RandomMatchedGate(domain_assignment, open_fraction=0.5, device=device)
                else:
                    gate = None

                rule = ThreeFactorRule(lr=condition.learning_rate, tau=condition.tau, third_factor=gate)

                epoch_results = []
                for epoch in range(n_epochs):
                    epoch_start = time.time()

                    train_metrics = train_epoch(
                        model=model, rule=rule, gate=gate,
                        train_loader=train_loader, condition=condition, device=device,
                    )
                    eval_metrics = evaluate(model, test_loader, device=device)
                    w_norm = compute_weight_norms(model)
                    has_nan = any(
                        torch.isnan(layer.weight.data).any() or torch.isinf(layer.weight.data).any()
                        for layer in model.layers
                    )
                    elapsed_epoch = time.time() - epoch_start

                    er = EpochResult(
                        epoch=epoch,
                        train_loss=train_metrics["train_loss"],
                        test_accuracy=eval_metrics["test_accuracy"],
                        test_loss=eval_metrics["test_loss"],
                        gate_fraction_open=train_metrics.get("gate_fraction_open", 0.0),
                        weight_norm=w_norm,
                        has_nan=has_nan,
                        wall_clock_seconds=elapsed_epoch,
                    )
                    epoch_results.append(er)

                    if verbose:
                        print(f"    Epoch {epoch}: loss={er.train_loss:.4f}, "
                              f"acc={er.test_accuracy:.4f}, norm={w_norm:.2f}")

                    rule.reset()

                final_acc = epoch_results[-1].test_accuracy if epoch_results else 0.0
                best_acc = max(r.test_accuracy for r in epoch_results) if epoch_results else 0.0

                result = ConditionResult(
                    condition_name=condition.name,
                    seed=seed,
                    n_epochs=n_epochs,
                    final_accuracy=final_acc,
                    best_accuracy=best_acc,
                    any_nan=any(r.has_nan for r in epoch_results),
                    epoch_results=epoch_results,
                )

            elapsed = time.time() - t0
            end_time = datetime.now(timezone.utc)
            print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
                  f"Completed: {condition.name} seed={seed} ({elapsed:.1f}s) "
                  f"acc={result.final_accuracy:.4f}")

            # Save CSV
            save_epoch_results_csv(
                results=result.epoch_results,
                condition_name=condition.name,
                seed=seed,
                output_dir=str(output_dir),
                timestamp=timestamp,
            )

            all_results.append(result)

    return all_results
