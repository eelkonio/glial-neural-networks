"""BCM-directed local learning rule.

Direction (LTP vs LTD) comes from postsynaptic calcium level relative
to a sliding threshold (BCM theory). The astrocyte D-serine gate
determines whether high calcium (LTP) is achievable. Heterosynaptic
competition zero-centers updates within domains.

Implements LocalLearningRule protocol from Step 12.
"""

import torch
from torch import Tensor

from code.step_imports import LayerState, DomainAssignment, CalciumDynamics
from code.bcm_config import BCMConfig


class BCMDirectedRule:
    """BCM-based directed local learning rule.

    Direction comes from postsynaptic calcium level relative to a
    sliding threshold (BCM theory). The astrocyte D-serine gate
    determines whether high calcium (LTP) is achievable.
    Heterosynaptic competition zero-centers updates within domains.

    Implements LocalLearningRule protocol (compute_update, reset).
    """

    name = "bcm_directed"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        calcium_dynamics: dict[int, CalciumDynamics],
        config: BCMConfig | None = None,
    ):
        """Initialize BCMDirectedRule.

        Args:
            domain_assignment: Spatial partitioning of neurons into domains.
            calcium_dynamics: Per-layer CalciumDynamics instances.
            config: BCMConfig with all parameters. Uses defaults if None.
        """
        self.domain_assignment = domain_assignment
        self.calcium_dynamics = calcium_dynamics
        self.config = config or BCMConfig()

        self.lr = self.config.lr
        self.theta_decay = self.config.theta_decay
        self.theta_init = self.config.theta_init
        self.d_serine_boost = self.config.d_serine_boost
        self.competition_strength = self.config.competition_strength
        self.clip_delta = self.config.clip_delta
        self.use_d_serine = self.config.use_d_serine
        self.use_competition = self.config.use_competition

        # Sliding threshold per layer
        self._theta: dict[int, Tensor] = {}

    def _compute_synapse_calcium(self, state: LayerState) -> Tensor:
        """Compute per-neuron synapse calcium from activations.

        synapse_calcium[j] = mean_over_batch(|post[j]|)

        Returns:
            Tensor of shape (out_features,) with non-negative values.
        """
        # Batch-mean of absolute post-activation per output neuron
        calcium = state.post_activation.abs().mean(dim=0)
        # Handle NaN gracefully
        calcium = torch.nan_to_num(calcium, nan=0.0)
        return calcium

    def _apply_d_serine_boost(self, calcium: Tensor, layer_index: int) -> Tensor:
        """Amplify calcium for neurons in domains where D-serine is available.

        For neurons in open domains: calcium *= (1 + d_serine_boost)
        For neurons in closed domains: calcium unchanged.

        Args:
            calcium: Per-neuron calcium, shape (out_features,).
            layer_index: Which layer.

        Returns:
            Amplified calcium tensor, shape (out_features,).
        """
        if not self.use_d_serine:
            return calcium

        device = calcium.device
        gate_open = self.calcium_dynamics[layer_index].get_gate_open()  # (n_domains,) bool
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_index)
        neuron_to_domain = neuron_to_domain.to(device)
        neuron_gate = gate_open.float().to(device)[neuron_to_domain]  # (out_features,)
        return calcium * (1.0 + self.d_serine_boost * neuron_gate)

    def _update_theta(self, domain_activities: Tensor, layer_index: int) -> None:
        """Update sliding BCM threshold with EMA of domain mean activity.

        theta = theta_decay * theta + (1 - theta_decay) * domain_activities

        Args:
            domain_activities: Mean activity per domain, shape (n_domains,).
            layer_index: Which layer.
        """
        device = domain_activities.device
        n_domains = domain_activities.shape[0]

        if layer_index not in self._theta:
            self._theta[layer_index] = torch.full(
                (n_domains,), self.theta_init, device=device
            )

        self._theta[layer_index] = (
            self.theta_decay * self._theta[layer_index]
            + (1 - self.theta_decay) * domain_activities
        )
        # Clamp to non-negative (float precision safety)
        self._theta[layer_index] = self._theta[layer_index].clamp(min=0.0)

    def _apply_heterosynaptic_competition(
        self, direction: Tensor, layer_index: int
    ) -> Tensor:
        """Zero-center direction within each astrocyte domain.

        For each domain: direction[neurons_in_domain] -= mean(direction[neurons_in_domain])

        Args:
            direction: Per-neuron direction signal, shape (out_features,).
            layer_index: Which layer.

        Returns:
            Zero-centered direction tensor, shape (out_features,).
        """
        if not self.use_competition:
            return direction

        device = direction.device
        domain_indices = self.domain_assignment.get_domain_indices(layer_index)

        # Clone to avoid in-place modification issues
        result = direction.clone()

        for d_idx, indices in enumerate(domain_indices):
            if len(indices) > 1:
                idx_t = torch.tensor(indices, device=device, dtype=torch.long)
                domain_mean = result[idx_t].mean()
                result[idx_t] = result[idx_t] - self.competition_strength * domain_mean

        return result

    def compute_update(self, state: LayerState) -> Tensor:
        """Compute BCM-directed weight update for one layer.

        Algorithm:
            1. synapse_calcium = mean_over_batch(|post|) per output neuron
            2. domain_activities = mean(synapse_calcium) per domain
            3. Step calcium dynamics with domain_activities
            4. Apply D-serine boost (amplify calcium in open domains)
            5. Update theta (EMA of domain mean activity)
            6. direction = synapse_calcium - theta[neuron_domain] (SIGNED)
            7. Heterosynaptic competition: zero-center within domains
            8. weight_delta = lr * outer(direction, mean_pre)
            9. Clip weight_delta norm to clip_delta

        Args:
            state: LayerState with pre/post activations and weights.

        Returns:
            Weight delta of shape (out_features, in_features).
        """
        layer_idx = state.layer_index
        device = state.weights.device
        out_features, in_features = state.weights.shape

        # 1. Compute per-neuron "synapse calcium"
        synapse_calcium = self._compute_synapse_calcium(state)

        # 2. Compute domain-level activities
        domain_indices = self.domain_assignment.get_domain_indices(layer_idx)
        n_domains = len(domain_indices)
        domain_activities = torch.zeros(n_domains, device=device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_t = torch.tensor(indices, device=device, dtype=torch.long)
                domain_activities[d_idx] = synapse_calcium[idx_t].mean()

        # 3. Step calcium dynamics (drives D-serine gate)
        self.calcium_dynamics[layer_idx].step(domain_activities)

        # 4. Apply D-serine boost
        synapse_calcium = self._apply_d_serine_boost(synapse_calcium, layer_idx)

        # 5. Update sliding threshold theta
        self._update_theta(domain_activities, layer_idx)

        # 6. Compute direction = synapse_calcium - theta (SIGNED!)
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_idx).to(device)
        neuron_theta = self._theta[layer_idx][neuron_to_domain]
        direction = synapse_calcium - neuron_theta

        # 7. Heterosynaptic competition: zero-center within domains
        direction = self._apply_heterosynaptic_competition(direction, layer_idx)

        # 8. Compute weight delta: outer(direction, mean_pre)
        mean_pre = state.pre_activation.mean(dim=0)
        delta_w = self.lr * torch.outer(direction, mean_pre)

        # 9. Clip to prevent explosion
        delta_norm = delta_w.norm()
        if delta_norm > self.clip_delta:
            delta_w = delta_w * (self.clip_delta / delta_norm)

        return delta_w

    def reset(self) -> None:
        """Reset sliding thresholds and calcium dynamics."""
        self._theta.clear()
        for cd in self.calcium_dynamics.values():
            cd.reset()
