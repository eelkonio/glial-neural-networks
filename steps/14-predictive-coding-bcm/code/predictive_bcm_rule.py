"""Predictive Coding + BCM directed local learning rule.

Combines BCM-directed signed updates (Step 12b) with inter-layer
domain-level prediction errors as the task-relevant information channel.

Each layer maintains a small (8×8) prediction of the next layer's domain
activities. The prediction error modulates the BCM direction signal,
making weight updates task-relevant.

Implements LocalLearningRule protocol from Step 12.
"""

import torch
import torch.nn.init as init
from torch import Tensor

from code.step_imports import LayerState, DomainAssignment, CalciumDynamics
from code.predictive_bcm_config import PredictiveBCMConfig


class PredictiveBCMRule:
    """Predictive coding + BCM directed local learning rule.

    Combines:
    - BCM direction (synapse_calcium - theta) for signed updates
    - Domain-level prediction errors for task-relevant information
    - D-serine gating driven by prediction error (surprise)
    - Heterosynaptic competition within domains

    The key insight: prediction errors between adjacent layers provide
    the missing task-relevant signal that Step 12b lacked. Domains that
    are "surprised" (high prediction error) learn actively; domains that
    predict well consolidate.

    Implements LocalLearningRule protocol (compute_all_updates, reset).
    """

    name = "predictive_bcm"

    def __init__(
        self,
        domain_assignment: DomainAssignment,
        calcium_dynamics: dict[int, CalciumDynamics],
        layer_sizes: list[tuple[int, int]],
        config: PredictiveBCMConfig | None = None,
        device: str = "cpu",
    ):
        """Initialize PredictiveBCMRule.

        Args:
            domain_assignment: Spatial partitioning of neurons into domains.
            calcium_dynamics: Per-layer CalciumDynamics instances.
            layer_sizes: List of (in_features, out_features) per layer.
            config: PredictiveBCMConfig. Uses defaults if None.
            device: Torch device for tensor allocation.
        """
        self.domain_assignment = domain_assignment
        self.calcium_dynamics = calcium_dynamics
        self.layer_sizes = layer_sizes
        self.config = config or PredictiveBCMConfig()
        self.device = device
        self.n_layers = len(layer_sizes)

        # Initialize prediction weights (domain-to-domain or neuron-to-neuron)
        self._prediction_weights: dict[int, Tensor] = {}
        self._init_prediction_weights()

        # Sliding BCM threshold per layer, per domain
        self._theta: dict[int, Tensor] = {}

        # Last prediction errors for monitoring
        self._last_prediction_errors: dict[int, Tensor] = {}

    def _init_prediction_weights(self) -> None:
        """Initialize prediction weights for each layer (except last)."""
        n_domains_per_layer = self.domain_assignment.n_domains_per_layer

        for layer_idx in range(self.n_layers - 1):
            if self.config.granularity == "domain":
                n_current = n_domains_per_layer[layer_idx]
                n_next = n_domains_per_layer[layer_idx + 1]
                P = torch.empty(n_next, n_current, device=self.device)
            else:  # "neuron" granularity
                _, out_current = self.layer_sizes[layer_idx]
                _, out_next = self.layer_sizes[layer_idx + 1]
                P = torch.empty(out_next, out_current, device=self.device)

            init.xavier_uniform_(P)
            self._prediction_weights[layer_idx] = P

    def _compute_domain_activities(self, post_activation: Tensor, layer_idx: int) -> Tensor:
        """Compute mean absolute activation per domain.

        Args:
            post_activation: Layer output, shape (batch, out_features).
            layer_idx: Which layer.

        Returns:
            Domain activities, shape (n_domains,).
        """
        domain_indices = self.domain_assignment.get_domain_indices(layer_idx)
        n_domains = len(domain_indices)
        # Batch-mean of absolute activations per neuron
        neuron_activities = post_activation.abs().mean(dim=0)  # (out_features,)

        domain_activities = torch.zeros(n_domains, device=self.device)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_t = torch.tensor(indices, device=self.device, dtype=torch.long)
                domain_activities[d_idx] = neuron_activities[idx_t].mean()

        return domain_activities

    def _compute_prediction_error(
        self, domain_activities_current: Tensor, domain_activities_next: Tensor, layer_idx: int
    ) -> Tensor:
        """Compute domain-level prediction error.

        Args:
            domain_activities_current: Current layer domain activities, shape (n_domains_current,).
            domain_activities_next: Next layer domain activities, shape (n_domains_next,).
            layer_idx: Current layer index.

        Returns:
            Prediction error, shape (n_domains_next,). SIGNED.
        """
        P = self._prediction_weights[layer_idx]
        predicted_next = P @ domain_activities_current  # (n_domains_next,)
        error = domain_activities_next - predicted_next  # SIGNED
        return error

    def _compute_information_signal(self, prediction_error: Tensor, layer_idx: int) -> Tensor:
        """Compute per-domain information signal from prediction error.

        Projects prediction error back through P^T to get per-domain
        "responsibility" for the error.

        Args:
            prediction_error: Domain prediction error, shape (n_domains_next,).
            layer_idx: Current layer index.

        Returns:
            Information signal per domain, shape (n_domains_current,). SIGNED.
        """
        P = self._prediction_weights[layer_idx]
        info = P.T @ prediction_error  # (n_domains_current,)

        # Normalize to unit norm (prevent scale issues)
        norm = info.norm() + 1e-8
        info = info / norm

        return info

    def _broadcast_to_neurons(self, domain_signal: Tensor, layer_idx: int) -> Tensor:
        """Broadcast domain-level signal to all neurons in each domain.

        Args:
            domain_signal: Per-domain values, shape (n_domains,).
            layer_idx: Which layer.

        Returns:
            Per-neuron values, shape (out_features,).
        """
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_idx).to(self.device)
        return domain_signal[neuron_to_domain]  # (out_features,)

    def _compute_bcm_direction(
        self, post_activation: Tensor, domain_activities: Tensor, surprise: Tensor, layer_idx: int
    ) -> Tensor:
        """Compute BCM direction with surprise-driven calcium dynamics.

        Args:
            post_activation: Layer output, shape (batch, out_features).
            domain_activities: Domain activities, shape (n_domains,).
            surprise: Absolute prediction error per domain, shape (n_domains,).
            layer_idx: Which layer.

        Returns:
            BCM direction per neuron, shape (out_features,). SIGNED.
        """
        # Synapse calcium: per-neuron activity
        synapse_calcium = post_activation.abs().mean(dim=0)  # (out_features,)
        synapse_calcium = torch.nan_to_num(synapse_calcium, nan=0.0)

        # Step calcium dynamics with SURPRISE (not raw activity)
        self.calcium_dynamics[layer_idx].step(surprise)

        # D-serine boost (if enabled)
        if self.config.use_d_serine:
            gate_open = self.calcium_dynamics[layer_idx].get_gate_open()
            neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_idx).to(self.device)
            neuron_gate = gate_open.float().to(self.device)[neuron_to_domain]
            synapse_calcium = synapse_calcium * (1.0 + self.config.d_serine_boost * neuron_gate)

        # Update theta (EMA of domain activities)
        n_domains = domain_activities.shape[0]
        if layer_idx not in self._theta:
            self._theta[layer_idx] = torch.full(
                (n_domains,), self.config.theta_init, device=self.device
            )
        self._theta[layer_idx] = (
            self.config.theta_decay * self._theta[layer_idx]
            + (1 - self.config.theta_decay) * domain_activities
        ).clamp(min=0.0)

        # Direction = calcium - theta (SIGNED)
        neuron_to_domain = self.domain_assignment.get_neuron_to_domain(layer_idx).to(self.device)
        neuron_theta = self._theta[layer_idx][neuron_to_domain]
        direction = synapse_calcium - neuron_theta

        return direction

    def _apply_competition(self, signal: Tensor, layer_idx: int) -> Tensor:
        """Apply heterosynaptic competition (zero-center within domains).

        Args:
            signal: Per-neuron signal, shape (out_features,).
            layer_idx: Which layer.

        Returns:
            Zero-centered signal, shape (out_features,).
        """
        if not self.config.use_competition:
            return signal

        result = signal.clone()
        domain_indices = self.domain_assignment.get_domain_indices(layer_idx)

        for d_idx, indices in enumerate(domain_indices):
            if len(indices) > 1:
                idx_t = torch.tensor(indices, device=self.device, dtype=torch.long)
                domain_mean = result[idx_t].mean()
                result[idx_t] = result[idx_t] - self.config.competition_strength * domain_mean

        return result

    def _apply_surprise_modulation(self, signal: Tensor, surprise: Tensor, layer_idx: int) -> Tensor:
        """Amplify learning in surprised domains, reduce in unsurprised.

        Args:
            signal: Per-neuron combined signal, shape (out_features,).
            surprise: Absolute prediction error per domain, shape (n_domains,).
            layer_idx: Which layer.

        Returns:
            Modulated signal, shape (out_features,).
        """
        if not self.config.use_domain_modulation:
            return signal

        # Compute per-domain amplification factor
        mean_surprise = surprise.mean() + 1e-8
        # Normalized surprise: 1.0 = average, >1 = more surprised, <1 = less surprised
        normalized = surprise / mean_surprise  # (n_domains,)
        # Clamp amplification
        amplification = normalized.clamp(
            min=1.0 / self.config.max_surprise_amplification,
            max=self.config.max_surprise_amplification,
        )

        # Broadcast to neurons
        neuron_amplification = self._broadcast_to_neurons(amplification, layer_idx)
        return signal * neuron_amplification

    def _update_prediction_weights(
        self, prediction_error: Tensor, domain_activities_current: Tensor, layer_idx: int
    ) -> None:
        """Update prediction weights via outer product rule.

        Args:
            prediction_error: Domain prediction error, shape (n_domains_next,).
            domain_activities_current: Current domain activities, shape (n_domains_current,).
            layer_idx: Current layer index.
        """
        if not self.config.learn_predictions or self.config.fixed_predictions:
            return

        delta_P = self.config.lr_pred * torch.outer(prediction_error, domain_activities_current)

        # Clip prediction weight update
        delta_norm = delta_P.norm()
        if delta_norm > self.config.clip_pred_delta:
            delta_P = delta_P * (self.config.clip_pred_delta / delta_norm)

        self._prediction_weights[layer_idx] = self._prediction_weights[layer_idx] + delta_P

    def compute_all_updates(self, states: list[LayerState]) -> list[Tensor]:
        """Compute weight updates for all layers simultaneously.

        This is the primary interface. Prediction errors require adjacent
        layer information, so all states must be available.

        Args:
            states: List of LayerState for all layers (from forward_with_states).

        Returns:
            List of weight delta tensors, one per layer.
        """
        deltas = []

        for layer_idx, state in enumerate(states):
            out_features, in_features = state.weights.shape

            # For the last layer, we can't compute prediction error to a "next" layer
            # Use a simplified BCM-only update for the last layer
            if layer_idx >= self.n_layers - 1 or layer_idx not in self._prediction_weights:
                # Last layer: BCM direction only (no prediction error available)
                domain_activities = self._compute_domain_activities(state.post_activation, layer_idx)
                # Use domain_activities as surprise proxy for last layer
                surprise = domain_activities
                direction = self._compute_bcm_direction(
                    state.post_activation, domain_activities, surprise, layer_idx
                )
                direction = self._apply_competition(direction, layer_idx)
                mean_pre = state.pre_activation.mean(dim=0)
                delta_w = self.config.lr * torch.outer(direction, mean_pre)
                # Clip
                delta_norm = delta_w.norm()
                if delta_norm > self.config.clip_delta:
                    delta_w = delta_w * (self.config.clip_delta / delta_norm)
                deltas.append(delta_w)
                continue

            # --- Full predictive BCM algorithm for layers 0..n-2 ---

            # 1. Domain activities for current and next layer (always needed for BCM)
            domain_activities_current = self._compute_domain_activities(
                state.post_activation, layer_idx
            )
            domain_activities_next = self._compute_domain_activities(
                states[layer_idx + 1].post_activation, layer_idx + 1
            )

            # 2. Prediction error (SIGNED) — granularity determines what we predict
            if self.config.granularity == "neuron":
                # Neuron-level: predict per-neuron mean activations
                neuron_activities_current = state.post_activation.abs().mean(dim=0)
                neuron_activities_next = states[layer_idx + 1].post_activation.abs().mean(dim=0)
                prediction_error = self._compute_prediction_error(
                    neuron_activities_current, neuron_activities_next, layer_idx
                )
            else:
                # Domain-level: predict per-domain mean activations
                prediction_error = self._compute_prediction_error(
                    domain_activities_current, domain_activities_next, layer_idx
                )
            self._last_prediction_errors[layer_idx] = prediction_error.detach().clone()

            # 3. Information signal (SIGNED, normalized)
            info_signal = self._compute_information_signal(prediction_error, layer_idx)

            # 4. Broadcast information to neurons (or use directly for neuron-level)
            if self.config.granularity == "neuron":
                # Info signal is already per-neuron for neuron-level granularity
                _, out_f = self.layer_sizes[layer_idx]
                # Normalize
                norm = info_signal.norm() + 1e-8
                info_per_neuron = info_signal / norm
            else:
                info_per_neuron = self._broadcast_to_neurons(info_signal, layer_idx)

            # 5. Surprise for calcium dynamics
            if self.config.granularity == "neuron":
                # For neuron-level, compute domain-level surprise from neuron prediction error
                # by averaging absolute error within each domain
                n_domains = len(self.domain_assignment.get_domain_indices(layer_idx))
                surprise_current = torch.zeros(n_domains, device=self.device)
                domain_indices = self.domain_assignment.get_domain_indices(layer_idx)
                # Use P^T @ error projected to domain level
                raw_info = self._prediction_weights[layer_idx].T @ prediction_error
                for d_idx, indices in enumerate(domain_indices):
                    if indices:
                        idx_t = torch.tensor(indices, device=self.device, dtype=torch.long)
                        surprise_current[d_idx] = raw_info[idx_t].abs().mean()
            else:
                # Domain-level: surprise = |P^T @ error| (responsibility per current domain)
                P = self._prediction_weights[layer_idx]
                surprise_current = (P.T @ prediction_error).abs()

            # 6. BCM direction (SIGNED, with surprise-driven calcium)
            # Drive calcium with surprise projected to current layer's domains
            bcm_direction = self._compute_bcm_direction(
                state.post_activation, domain_activities_current, surprise_current, layer_idx
            )

            # 7. Combine BCM direction with information signal
            if self.config.combination_mode == "multiplicative":
                combined = bcm_direction * info_per_neuron
            elif self.config.combination_mode == "additive":
                combined = bcm_direction + info_per_neuron
            elif self.config.combination_mode == "threshold":
                # Use info signal to modulate theta instead
                combined = bcm_direction * (1.0 + info_per_neuron)
            else:
                combined = bcm_direction * info_per_neuron

            # 8. Heterosynaptic competition
            combined = self._apply_competition(combined, layer_idx)

            # 9. Surprise modulation (uses current-layer domain surprise)
            combined = self._apply_surprise_modulation(combined, surprise_current, layer_idx)

            # 10. Weight delta
            mean_pre = state.pre_activation.mean(dim=0)
            delta_w = self.config.lr * torch.outer(combined, mean_pre)

            # 11. Clip
            delta_norm = delta_w.norm()
            if delta_norm > self.config.clip_delta:
                delta_w = delta_w * (self.config.clip_delta / delta_norm)

            # 12. NaN safety
            delta_w = torch.nan_to_num(delta_w, nan=0.0)

            deltas.append(delta_w)

            # 13. Update prediction weights
            if self.config.granularity == "neuron":
                neuron_act_current = state.post_activation.abs().mean(dim=0)
                self._update_prediction_weights(
                    prediction_error, neuron_act_current, layer_idx
                )
            else:
                self._update_prediction_weights(
                    prediction_error, domain_activities_current, layer_idx
                )

        return deltas

    def compute_update(self, state: LayerState) -> Tensor:
        """Single-layer interface — raises error directing to compute_all_updates.

        PredictiveBCMRule requires all layer states simultaneously because
        prediction errors are computed between adjacent layers.
        """
        raise NotImplementedError(
            "PredictiveBCMRule requires all layer states. "
            "Use compute_all_updates(states) instead."
        )

    def get_prediction_errors(self) -> dict[int, Tensor]:
        """Return last-computed prediction errors per layer (for monitoring).

        Returns:
            Dict mapping layer_idx to prediction error tensor.
        """
        return {k: v.clone() for k, v in self._last_prediction_errors.items()}

    def reset(self) -> None:
        """Reset calcium dynamics between epochs.

        Note: Prediction weights and theta are NOT reset — they accumulate
        learning across epochs. Only calcium dynamics are reset to prevent
        stale calcium state from the previous epoch's final batch.
        """
        self._last_prediction_errors.clear()
        for cd in self.calcium_dynamics.values():
            cd.reset()

    def full_reset(self) -> None:
        """Full reset including prediction weights (for new experiment runs)."""
        self._theta.clear()
        self._last_prediction_errors.clear()
        self._init_prediction_weights()
        for cd in self.calcium_dynamics.values():
            cd.reset()
