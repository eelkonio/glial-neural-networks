"""Volume-transmitted teaching signal gate (Variant C).

Implements ThirdFactorInterface protocol.
Signal = sum_over_domains(error_source × gaussian_kernel(distance)) × calcium_gate
Shape: (out_features,) — spatially-graded signed value per neuron.

Each domain computes a local error (activity vs label-derived target),
then this error diffuses to neighboring domains via Gaussian kernel
weighted by spatial distance. The calcium gate determines which
domains are receptive to the teaching signal.

This is the most complex variant and most likely to approach backprop
performance, as it approximates backprop's error delivery through
volume transmission.
"""

import torch
from torch import Tensor

from code.calcium.li_rinzel import CalciumDynamics
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment


class VolumeTeachingGate:
    """Volume-transmitted teaching signal with spatial diffusion.

    Computes domain-local error from activity vs label-derived target,
    diffuses it spatially via Gaussian kernel, applies gap junction
    calcium coupling, and gates by calcium threshold.

    Args:
        domain_assignment: DomainAssignment instance for neuron-to-domain mapping.
        calcium_config: CalciumConfig for creating per-layer CalciumDynamics.
        diffusion_sigma: Gaussian diffusion width. If None, uses mean inter-domain distance.
        n_classes: Number of output classes (for label projection).
        gap_junction_strength: Strength of calcium coupling between domains.
        device: Torch device for computation.
    """

    name = "volume_teaching"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        calcium_config: CalciumConfig | None = None,
        diffusion_sigma: float | None = None,
        n_classes: int = 10,
        gap_junction_strength: float = 0.1,
        device: str = "cpu",
    ):
        self.domain_assignment = domain_assignment
        self.calcium_config = calcium_config or CalciumConfig()
        self.diffusion_sigma = diffusion_sigma
        self.n_classes = n_classes
        self.gap_junction_strength = gap_junction_strength
        self.device = device

        # Create one CalciumDynamics per layer
        self._calcium: dict[int, CalciumDynamics] = {}
        for layer_idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
            self._calcium[layer_idx] = CalciumDynamics(
                n_domains=n_domains,
                config=self.calcium_config,
                device=device,
            )

        # Fixed random projections per layer: maps one-hot labels to domain targets
        self._label_projections: dict[int, Tensor] = {}

        # Precomputed Gaussian diffusion kernels per layer
        self._diffusion_kernels: dict[int, Tensor] = {}

        # Precompute kernels and projections
        self._initialize_kernels_and_projections()

    def _initialize_kernels_and_projections(self) -> None:
        """Precompute diffusion kernels and label projections for each layer."""
        for layer_idx, n_domains in enumerate(self.domain_assignment.n_domains_per_layer):
            # Label projection: fixed random mapping from (n_classes,) to (n_domains,)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(42 + layer_idx * 7)
            proj = torch.randn(n_domains, self.n_classes, generator=gen, device=self.device)
            # Normalize so projection doesn't explode
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
            self._label_projections[layer_idx] = proj

            # Diffusion kernel from inter-domain distances
            distances = self.domain_assignment.get_domain_distances(layer_idx)
            distances = distances.to(self.device)

            # Determine sigma
            if self.diffusion_sigma is not None:
                sigma = self.diffusion_sigma
            else:
                # Use mean inter-domain distance (excluding self-distance)
                mask = ~torch.eye(n_domains, device=self.device, dtype=torch.bool)
                if mask.any() and distances[mask].numel() > 0:
                    sigma = distances[mask].mean().item()
                else:
                    sigma = 1.0  # Fallback for single-domain layers

            # Gaussian kernel: exp(-d² / 2σ²)
            if sigma > 0:
                kernel = torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))
            else:
                kernel = torch.eye(n_domains, device=self.device)

            # Clamp minimum to prevent underflow
            kernel = kernel.clamp(min=1e-10)

            # Normalize rows so each domain receives a weighted average
            kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-8)

            self._diffusion_kernels[layer_idx] = kernel

    def compute_signal(
        self,
        layer_activations: Tensor,
        layer_index: int,
        labels: Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> Tensor:
        """Compute volume-transmitted teaching signal.

        1. Compute domain-local error: domain_mean_activity - projected_label_target
        2. Get diffusion kernel for this layer (precomputed Gaussian)
        3. Diffuse error: received = kernel @ source_errors
        4. Apply gap junction calcium coupling between domains
        5. Update calcium dynamics
        6. Gate: received_signal * (Ca > threshold)
        7. Map domain signal to per-neuron (out_features,)

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is (0-indexed).
            labels: Ground truth labels (batch,) — required for teaching signal.
            global_loss: Unused.
            prev_loss: Unused.

        Returns:
            Signed teaching tensor of shape (out_features,).
        """
        out_features = layer_activations.shape[-1]
        device = layer_activations.device

        # Get domain structure for this layer
        domain_indices = self.domain_assignment.get_domain_indices(layer_index)
        n_domains = len(domain_indices)

        # Compute mean activation per domain
        mean_act = layer_activations.abs().mean(dim=0)  # (out_features,)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()

        # 1. Compute domain-local error
        if labels is not None:
            # One-hot encode labels: (batch, n_classes)
            one_hot = torch.zeros(
                labels.shape[0], self.n_classes, device=device
            )
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            mean_one_hot = one_hot.mean(dim=0)  # (n_classes,)

            # Project label to domain targets: (n_domains,)
            projection = self._label_projections[layer_index].to(device)
            domain_target = projection @ mean_one_hot  # (n_domains,)

            # Error: activity - target
            source_errors = domain_activities - domain_target
        else:
            # No labels: return zero signal (unsupervised mode)
            neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
            return torch.zeros(out_features, device=device)

        # 2-3. Diffuse error through spatial kernel
        kernel = self._diffusion_kernels[layer_index].to(device)
        received_signal = kernel @ source_errors  # (n_domains,)

        # 4. Apply gap junction calcium coupling between domains
        calcium = self._calcium[layer_index]
        ca_state = calcium.get_calcium()  # (n_domains,)

        # Gap junction coupling: diffuse calcium toward neighbors
        if self.gap_junction_strength > 0 and n_domains > 1:
            # Compute calcium differences weighted by kernel (proximity)
            ca_diff = ca_state.unsqueeze(0) - ca_state.unsqueeze(1)  # (n_domains, n_domains)
            # Net calcium flow into each domain (from neighbors with higher Ca)
            ca_coupling = self.gap_junction_strength * (kernel * ca_diff).sum(dim=1)
            # Apply coupling to calcium state (modifies in place)
            calcium.ca = (calcium.ca + ca_coupling).clamp(0.0, calcium.config.ca_max)

        # 5. Update calcium dynamics
        calcium.step(domain_activities)

        # 6. Gate: received_signal * (Ca > threshold)
        gate_open = calcium.get_gate_open()  # (n_domains,) bool
        gated_signal = received_signal * gate_open.float()  # (n_domains,)

        # 7. Map domain signal to per-neuron (out_features,)
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        gate_signal = gated_signal[neuron_to_domain]  # (out_features,)

        return gate_signal

    def reset(self) -> None:
        """Reset calcium state."""
        for calcium in self._calcium.values():
            calcium.reset()

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        state = {}
        for idx, ca in self._calcium.items():
            state[f"calcium_{idx}"] = ca.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        for idx, ca in self._calcium.items():
            key = f"calcium_{idx}"
            if key in state:
                ca.load_state_dict(state[key])
