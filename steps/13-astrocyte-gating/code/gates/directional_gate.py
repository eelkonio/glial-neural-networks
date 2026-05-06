"""Directional astrocyte gate (Variant B).

Implements ThirdFactorInterface protocol.
Signal = calcium_magnitude × normalized(current_activity - predicted_activity)
Shape: (out_features,) — signed value per neuron.

The activity prediction error provides directional credit assignment:
- Positive: neuron is more active than expected → strengthen inputs
- Negative: neuron is less active than expected → weaken inputs
- Zero: neuron matches prediction → no learning needed

This variant adds direction via activity prediction error (novelty detection).
"""

import torch
from torch import Tensor

from code.calcium.li_rinzel import CalciumDynamics
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment


class DirectionalGate:
    """Directional astrocyte gate: calcium × activity prediction error.

    Maintains an EMA prediction of domain activity. The error between
    current and predicted activity provides directional credit assignment.
    Calcium state gates whether the signal is delivered.

    Args:
        domain_assignment: DomainAssignment instance for neuron-to-domain mapping.
        calcium_config: CalciumConfig for creating per-layer CalciumDynamics.
        prediction_decay: EMA decay rate for activity prediction (default 0.95).
        device: Torch device for computation.
    """

    name = "directional_gate"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        calcium_config: CalciumConfig | None = None,
        prediction_decay: float = 0.95,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.calcium_config = calcium_config or CalciumConfig()
        self.prediction_decay = prediction_decay
        self.device = device

        # Create one CalciumDynamics per layer
        self._calcium: dict[int, CalciumDynamics] = {}
        for layer_idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
            self._calcium[layer_idx] = CalciumDynamics(
                n_domains=n_domains,
                config=self.calcium_config,
                device=device,
            )

        # Activity predictions per layer: EMA of domain activities
        self._predictions: dict[int, Tensor] = {}

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        """Compute directional gate signal.

        1. Compute mean activation per domain (batch-averaged)
        2. Compute activity error = current - predicted
        3. Update prediction with EMA: pred = decay * pred + (1-decay) * current
        4. Normalize error per domain (divide by std + eps)
        5. Update calcium dynamics
        6. Output = calcium_magnitude * normalized_error where Ca > threshold, else 0
        7. Map domain signal to per-neuron (out_features,)

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is (0-indexed).
            labels: Unused by directional gate.
            global_loss: Unused by directional gate.
            prev_loss: Unused by directional gate.

        Returns:
            Signed gate tensor of shape (out_features,).
        """
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        # Get domain structure for this layer
        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # 1. Compute mean activation per domain (using absolute values)
        mean_act = layer_activations.abs().mean(dim=0)  # (out_features,)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()

        # 2. Compute activity error = current - predicted
        if layer_index not in self._predictions:
            # Initialize prediction to current activity (no error on first step)
            self._predictions[layer_index] = domain_activities.clone()

        prediction = self._predictions[layer_index]
        activity_error = domain_activities - prediction  # (n_domains,)

        # 3. Update prediction with EMA
        self._predictions[layer_index] = (
            self.prediction_decay * prediction
            + (1 - self.prediction_decay) * domain_activities
        )

        # 4. Normalize error (divide by std + eps to prevent magnitude domination)
        # For single-domain layers, std is undefined — use abs mean instead
        if n_domains > 1:
            error_std = activity_error.std() + 1e-8
        else:
            error_std = activity_error.abs().mean() + 1e-8
        normalized_error = activity_error / error_std

        # 5. Update calcium dynamics
        calcium = self._calcium[layer_index]
        calcium.step(domain_activities)

        # 6. Output = calcium_magnitude * normalized_error where Ca > threshold
        ca_state = calcium.get_calcium()  # (n_domains,)
        gate_open = calcium.get_gate_open()  # (n_domains,) bool

        # Signal: calcium magnitude * error direction, gated by threshold
        domain_signal = ca_state * normalized_error * gate_open.float()  # (n_domains,)

        # 7. Map domain signal to per-neuron (out_features,)
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        gate_signal = domain_signal[neuron_to_domain]  # (out_features,)

        return gate_signal

    def reset(self) -> None:
        """Reset calcium state and activity predictions."""
        for calcium in self._calcium.values():
            calcium.reset()
        self._predictions.clear()

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        state = {}
        for idx, ca in self._calcium.items():
            state[f"calcium_{idx}"] = ca.state_dict()
        for idx, pred in self._predictions.items():
            state[f"prediction_{idx}"] = pred.clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        for idx, ca in self._calcium.items():
            key = f"calcium_{idx}"
            if key in state:
                ca.load_state_dict(state[key])
        for key, val in state.items():
            if key.startswith("prediction_"):
                idx = int(key.split("_")[1])
                self._predictions[idx] = val.to(self.device)
