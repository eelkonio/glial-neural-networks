"""Binary astrocyte gate (Variant A).

Implements ThirdFactorInterface protocol.
Signal = 1.0 if domain calcium > threshold, else 0.0.
Shape: (out_features,) — same value for all neurons in a domain.

This is the simplest gate variant: it restricts WHERE learning happens
(only in domains with sufficient calcium) but provides no directional
information about HOW weights should change.
"""

import torch
from torch import Tensor

from code.calcium.li_rinzel import CalciumDynamics
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment


class BinaryGate:
    """Binary astrocyte gate: plasticity on/off based on calcium threshold.

    Each layer gets its own CalciumDynamics instance (one per layer).
    The gate maintains a dict of CalciumDynamics keyed by layer_index.

    Args:
        domain_assignment: DomainAssignment instance for neuron-to-domain mapping.
        calcium_config: CalciumConfig for creating per-layer CalciumDynamics.
        device: Torch device for computation.
    """

    name = "binary_gate"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        calcium_config: CalciumConfig | None = None,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.calcium_config = calcium_config or CalciumConfig()
        self.device = device

        # Create one CalciumDynamics per layer
        self._calcium: dict[int, CalciumDynamics] = {}
        for layer_idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
            self._calcium[layer_idx] = CalciumDynamics(
                n_domains=n_domains,
                config=self.calcium_config,
                device=device,
            )

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        """Compute binary gate signal.

        1. Compute mean absolute activation per domain for this layer
        2. Update calcium dynamics with domain activities
        3. Get gate_open mask (Ca > threshold)
        4. Map domain gate values to per-neuron output (out_features,)

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is (0-indexed).
            labels: Unused by binary gate.
            global_loss: Unused by binary gate.
            prev_loss: Unused by binary gate.

        Returns:
            Gate tensor of shape (out_features,) with values 0.0 or 1.0.
        """
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        # Get domain structure for this layer
        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # 1. Compute mean absolute activation per domain
        # Average over batch first, then compute per-domain means
        mean_act = layer_activations.abs().mean(dim=0)  # (out_features,)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()

        # 2. Update calcium dynamics
        calcium = self._calcium[layer_index]
        calcium.step(domain_activities)

        # 3. Get gate_open mask (Ca > threshold)
        gate_open = calcium.get_gate_open()  # (n_domains,) bool

        # 4. Map domain gate values to per-neuron output
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        gate_signal = gate_open.float()[neuron_to_domain]  # (out_features,)

        return gate_signal

    def reset(self) -> None:
        """Reset calcium state for all layers."""
        for calcium in self._calcium.values():
            calcium.reset()

    def state_dict(self) -> dict:
        """Serialize calcium state for checkpointing."""
        return {
            f"calcium_{idx}": ca.state_dict()
            for idx, ca in self._calcium.items()
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore calcium state from checkpoint."""
        for idx, ca in self._calcium.items():
            key = f"calcium_{idx}"
            if key in state:
                ca.load_state_dict(state[key])
