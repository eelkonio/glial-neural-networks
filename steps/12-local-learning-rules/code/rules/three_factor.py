"""Three-factor learning rule — eligibility trace × third factor × learning rate.

The three-factor rule is the CRITICAL substrate for Step 13's astrocyte gating.
Weight change = eligibility_trace × third_factor_signal × learning_rate.

The eligibility trace accumulates pre/post correlation (Hebbian coincidence),
but actual weight change only occurs when modulated by the third factor signal.
In Step 12, three placeholder third-factor providers are implemented:
  - RandomNoise: lower bound (no useful signal)
  - GlobalReward: scalar reward based on loss improvement
  - LayerWiseError: local error signal per layer (approximates backprop)

Step 13 will replace these with the astrocyte D-serine gate.
"""

import torch

from code.rules.base import LayerState, ThirdFactorInterface


class RandomNoiseThirdFactor:
    """Third factor that provides random Gaussian noise.

    This is the lower bound — no useful learning signal, just noise.
    Useful for ablation: any rule that works with random noise would
    work even better with a meaningful signal.

    Attributes:
        name: Identifier for this third factor source.
        sigma: Standard deviation of the noise.
    """

    name = "random_noise"

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def compute_signal(
        self,
        layer_activations: torch.Tensor,
        layer_index: int,
        labels: torch.Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> torch.Tensor:
        """Return N(0, σ²) noise of shape (out_features,).

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is.
            labels: Unused.
            global_loss: Unused.
            prev_loss: Unused.

        Returns:
            Noise tensor of shape (out_features,).
        """
        out_features = layer_activations.shape[-1]
        return torch.randn(out_features, device=layer_activations.device) * self.sigma


class GlobalRewardThirdFactor:
    """Third factor based on global reward (loss improvement).

    Computes reward = (prev_loss - current_loss) - running_baseline.
    The baseline is updated with exponential moving average so that
    the signal reflects whether the current improvement is above or
    below the recent average improvement.

    Attributes:
        name: Identifier for this third factor source.
        baseline_decay: EMA decay rate for the running baseline.
    """

    name = "global_reward"

    def __init__(self, baseline_decay: float = 0.99):
        self.baseline_decay = baseline_decay
        self.running_baseline: float = 0.0
        self._initialized: bool = False

    def compute_signal(
        self,
        layer_activations: torch.Tensor,
        layer_index: int,
        labels: torch.Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> torch.Tensor:
        """Compute scalar reward signal.

        reward = (prev_loss - global_loss) - running_baseline
        Then update running_baseline with EMA.

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is.
            labels: Unused.
            global_loss: Current batch loss value.
            prev_loss: Previous batch loss value.

        Returns:
            Scalar reward tensor (broadcast-compatible with eligibility trace).
        """
        device = layer_activations.device

        if global_loss is None or prev_loss is None:
            return torch.tensor(0.0, device=device)

        # Raw reward: positive when loss decreases
        raw_reward = prev_loss - global_loss

        # Subtract running baseline
        reward = raw_reward - self.running_baseline

        # Update running baseline with EMA
        if not self._initialized:
            self.running_baseline = raw_reward
            self._initialized = True
        else:
            self.running_baseline = (
                self.baseline_decay * self.running_baseline
                + (1 - self.baseline_decay) * raw_reward
            )

        return torch.tensor(reward, device=device, dtype=torch.float32)


class LayerWiseErrorThirdFactor:
    """Third factor based on layer-wise error signal.

    For the output layer: error = activations - one_hot(label).
    For hidden layers: uses a fixed random projection of the one-hot label
    as a local target, then error = activations - projected_target.

    This provides a local approximation of backprop's error signal
    without requiring gradient flow through the network.

    Attributes:
        name: Identifier for this third factor source.
        n_classes: Number of output classes.
    """

    name = "layer_wise_error"

    def __init__(self, n_classes: int = 10):
        self.n_classes = n_classes
        # Fixed random projections for hidden layers (initialized lazily)
        self._projections: dict[int, torch.Tensor] = {}

    def _get_projection(
        self, layer_index: int, out_features: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create a fixed random projection for a hidden layer.

        The projection maps one-hot labels (n_classes,) to (out_features,).
        It's fixed (not learned) to keep the signal local.
        """
        if layer_index not in self._projections:
            # Use a fixed seed per layer for reproducibility
            gen = torch.Generator(device="cpu")
            gen.manual_seed(42 + layer_index)
            proj = torch.randn(out_features, self.n_classes, generator=gen)
            # Normalize columns so projection doesn't explode
            proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)
            self._projections[layer_index] = proj.to(device)
        return self._projections[layer_index]

    def compute_signal(
        self,
        layer_activations: torch.Tensor,
        layer_index: int,
        labels: torch.Tensor | None = None,
        global_loss: float | None = None,
        prev_loss: float | None = None,
    ) -> torch.Tensor:
        """Compute layer-wise error signal.

        For output layer: error = mean(activations) - one_hot(label)
        For hidden layers: error = mean(activations) - projection @ one_hot(label)

        Args:
            layer_activations: Post-activation output (batch, out_features).
            layer_index: Which layer this is.
            labels: Ground truth labels (batch,).
            global_loss: Unused.
            prev_loss: Unused.

        Returns:
            Error signal of shape (out_features,).
        """
        device = layer_activations.device
        out_features = layer_activations.shape[-1]

        if labels is None:
            return torch.zeros(out_features, device=device)

        # One-hot encode labels: (batch, n_classes)
        one_hot = torch.zeros(
            labels.shape[0], self.n_classes, device=device
        )
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Mean activations over batch: (out_features,)
        mean_activations = layer_activations.mean(dim=0)

        if out_features == self.n_classes:
            # Output layer: target is the mean one-hot label
            target = one_hot.mean(dim=0)
        else:
            # Hidden layer: project one-hot to layer dimension
            projection = self._get_projection(layer_index, out_features, device)
            # Mean projected target: (out_features,)
            target = (projection @ one_hot.T).mean(dim=1)

        # Error signal: difference between activations and target
        error = mean_activations - target
        return error


class ThreeFactorRule:
    """Three-factor learning rule with eligibility traces.

    Weight update = eligibility_trace × third_factor_signal × learning_rate.

    The eligibility trace accumulates pre/post Hebbian correlation over time,
    decaying with time constant τ. The third factor signal modulates whether
    the trace converts to an actual weight change.

    This is the substrate for Step 13's astrocyte gating — the third factor
    interface is designed so the astrocyte gate can be plugged in directly.

    Attributes:
        name: Human-readable identifier.
        lr: Learning rate η.
        tau: Eligibility trace time constant.
        third_factor: The pluggable third factor signal provider.
    """

    name = "three_factor"

    def __init__(
        self,
        lr: float = 0.01,
        tau: float = 100.0,
        third_factor: ThirdFactorInterface | None = None,
    ):
        self.lr = lr
        self.tau = tau
        self.third_factor = third_factor or RandomNoiseThirdFactor()
        # Eligibility traces per layer: dict[layer_index, Tensor]
        self._eligibility: dict[int, torch.Tensor] = {}
        # Track previous loss for reward-based third factors
        self._prev_loss: float | None = None

    def compute_update(self, state: LayerState) -> torch.Tensor:
        """Compute three-factor weight update.

        1. Update eligibility: e = (1 - 1/τ) * e + mean_over_batch(outer(post, pre))
        2. Get third factor signal from interface
        3. Compute delta_w = e * M * η
        4. Decay eligibility after use
        5. Guard against overflow

        Args:
            state: Layer state with pre/post activations and weights.

        Returns:
            Weight delta of shape (out_features, in_features).
        """
        layer_idx = state.layer_index
        device = state.weights.device

        # 1. Compute Hebbian correlation: mean_over_batch(outer(post, pre))
        hebbian = torch.einsum(
            "bo,bi->oi", state.post_activation, state.pre_activation
        )
        hebbian /= state.pre_activation.size(0)  # mean over batch

        # Update eligibility trace with exponential decay
        decay = 1.0 - 1.0 / self.tau
        if layer_idx in self._eligibility:
            self._eligibility[layer_idx] = (
                decay * self._eligibility[layer_idx] + hebbian
            )
        else:
            self._eligibility[layer_idx] = hebbian.clone()

        # 2. Get third factor signal
        M = self.third_factor.compute_signal(
            layer_activations=state.post_activation,
            layer_index=layer_idx,
            labels=state.labels,
            global_loss=state.global_loss,
            prev_loss=self._prev_loss,
        )

        # 3. Compute weight update: e * M * η
        # M can be scalar, (out_features,), or (out_features, in_features)
        eligibility = self._eligibility[layer_idx]

        if M.dim() == 0:
            # Scalar modulation
            delta = eligibility * M * self.lr
        elif M.dim() == 1:
            # Per-output modulation: broadcast (out,) over (out, in)
            delta = eligibility * M.unsqueeze(1) * self.lr
        else:
            # Full matrix modulation
            delta = eligibility * M * self.lr

        # 4. Decay eligibility after use (trace is consumed)
        self._eligibility[layer_idx] = self._eligibility[layer_idx] * decay

        # 5. Overflow guard: normalize if trace magnitude exceeds threshold
        trace_norm = self._eligibility[layer_idx].norm()
        if trace_norm > 1e6:
            self._eligibility[layer_idx] = (
                self._eligibility[layer_idx] / trace_norm * 1e3
            )

        # 6. Clip delta to prevent weight explosion
        delta_norm = delta.norm()
        if delta_norm > 1.0:
            delta = delta / delta_norm

        # Track loss for next step (for reward-based third factors)
        if state.global_loss is not None:
            self._prev_loss = state.global_loss

        return delta

    def reset(self) -> None:
        """Clear all eligibility traces."""
        self._eligibility.clear()
        self._prev_loss = None
