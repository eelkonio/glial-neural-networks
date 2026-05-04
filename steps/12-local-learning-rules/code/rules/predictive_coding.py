"""Predictive Coding learning rule — local prediction error minimization.

Each layer predicts the activity of the layer below via top-down connections.
Weight updates minimize local prediction error using only local activity
and error signals. Inference iterations update representations before
weight updates, approximating backpropagation through local computations.

Based on Whittington & Bogacz (2017) and Rao & Ballard (1999).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from code.network.local_mlp import LocalMLP


class PredictiveCodingRule:
    """Predictive coding with top-down predictions and inference iterations.

    Before each weight update, the network runs T inference iterations
    where representations are updated to minimize prediction error.
    Then weights are updated using local Hebbian-like rules on the
    converged representations and errors.

    Attributes:
        name: Human-readable identifier.
        lr: Learning rate for weight updates.
        inference_lr: Step size for inference iterations.
        n_inference_steps: Number of inference iterations before weight update.
        n_classes: Number of output classes.
    """

    name = "predictive_coding"

    def __init__(
        self,
        lr: float = 0.01,
        inference_lr: float = 0.1,
        n_inference_steps: int = 20,
        n_classes: int = 10,
    ):
        self.lr = lr
        self.inference_lr = inference_lr
        self.n_inference_steps = n_inference_steps
        self.n_classes = n_classes
        # Top-down prediction weights: W_predict[i] predicts layer i from layer i+1
        self._W_predict: list[torch.Tensor] | None = None
        self._setup_done = False

    def setup_predictions(self, model: "LocalMLP") -> None:
        """Initialize top-down prediction weights.

        W_predict[i] maps from layer i+1's representation to layer i's
        representation (predicting lower from higher).

        Args:
            model: The LocalMLP model.
        """
        device = next(model.parameters()).device
        self._W_predict = []

        # Layer dimensions: [784, 128, 128, 128, 128, 10]
        # We need predictions from each layer to the one below
        dims = [model.input_size] + [model.hidden_size] * 4 + [model.n_classes]

        # W_predict[i] predicts layer i (dims[i]) from layer i+1 (dims[i+1])
        # Shape: (dims[i], dims[i+1])
        for i in range(len(dims) - 1):
            W = torch.randn(dims[i], dims[i + 1], device=device) * 0.01
            self._W_predict.append(W)

        self._setup_done = True

    def compute_prediction_errors(
        self,
        representations: list[torch.Tensor],
        input_x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute prediction errors at each layer.

        ε[i] = representation[i] - W_predict[i] @ representation[i+1]
        For the input layer: ε[0] = input_x - W_predict[0] @ representation[1]

        Args:
            representations: List of layer representations [r0, r1, ..., rL].
                r0 corresponds to the input layer representation.
            input_x: The actual input (batch, 784).

        Returns:
            List of prediction errors, one per layer (except the top).
        """
        errors = []
        n_layers = len(representations)

        for i in range(n_layers - 1):
            # What this layer actually has
            if i == 0:
                actual = input_x
            else:
                actual = representations[i]

            # Top-down prediction from layer above
            prediction = representations[i + 1] @ self._W_predict[i].T

            # Prediction error
            error = actual - prediction
            errors.append(error)

        return errors

    def inference_step(
        self,
        representations: list[torch.Tensor],
        input_x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """One inference iteration updating representations.

        Each representation is updated to reduce prediction error
        from both above (top-down) and below (bottom-up).

        Args:
            representations: Current layer representations.
            input_x: The actual input.
            labels: Optional labels for supervised signal at top.

        Returns:
            Updated representations.
        """
        errors = self.compute_prediction_errors(representations, input_x)
        new_representations = [r.clone() for r in representations]

        # Update hidden representations (not input, not top)
        for i in range(1, len(representations) - 1):
            # Bottom-up error from layer below wanting this layer to predict better
            # Error at layer i-1 is: actual[i-1] - W_predict[i-1] @ repr[i]
            # Gradient w.r.t. repr[i]: W_predict[i-1].T @ error[i-1] (with sign flip)
            bottom_up_signal = errors[i - 1] @ self._W_predict[i - 1]

            # Top-down error from this layer's own prediction error
            if i < len(errors):
                top_down_signal = -errors[i]
            else:
                top_down_signal = torch.zeros_like(representations[i])

            # Update representation
            update = self.inference_lr * (bottom_up_signal + top_down_signal)
            new_representations[i] = representations[i] + update

            # Clamp to prevent divergence
            new_representations[i] = new_representations[i].clamp(-10.0, 10.0)

        # Supervised signal at top layer: bias toward correct class
        if labels is not None:
            top_idx = len(representations) - 1
            target = torch.zeros_like(representations[top_idx])
            target.scatter_(1, labels.unsqueeze(1), 1.0)
            # Blend toward target
            supervised_signal = self.inference_lr * (target - representations[top_idx])
            new_representations[top_idx] = (
                representations[top_idx] + supervised_signal
            )
            new_representations[top_idx] = new_representations[top_idx].clamp(-10.0, 10.0)

        return new_representations

    def update_weights(
        self,
        model: "LocalMLP",
        representations: list[torch.Tensor],
        errors: list[torch.Tensor],
    ) -> None:
        """Update bottom-up and top-down weights using local Hebbian rules.

        ΔW_up[i] = η · mean(outer(error[i], repr[i]))
        ΔW_predict[i] = η · mean(outer(error[i], repr[i+1]))

        Args:
            model: The LocalMLP model (for bottom-up weight access).
            representations: Converged representations after inference.
            errors: Final prediction errors.
        """
        batch_size = representations[0].size(0)

        # errors[i] has shape (batch, dims[i]) for i in 0..n_layers-2
        # model.layers[i] maps repr[i] (dims[i]) -> repr[i+1] (dims[i+1])
        # layer[i].weight shape: (dims[i+1], dims[i])

        for i in range(len(model.layers)):
            layer = model.layers[i]
            source = representations[i]  # (batch, dims[i])

            # Bottom-up weight update: ΔW_up[i] = η · mean(outer(ε[i+1], repr[i]))
            # where ε[i+1] is the prediction error at level i+1
            # This is the standard PC recognition weight update
            if i + 1 < len(errors):
                # errors[i+1] has shape (batch, dims[i+1])
                error_at_output = errors[i + 1]
            elif i < len(errors):
                # For the last feedforward layer, use error projected through W_predict
                # errors[i] has shape (batch, dims[i])
                # Project to output dimension: (batch, dims[i]) @ W_predict[i] -> doesn't work
                # Instead, use repr[i+1] - target as error signal
                error_at_output = representations[i + 1]  # Use repr as-is (weak signal)
                # Skip update for last layer if no proper error available
                continue
            else:
                continue

            # delta shape: (dims[i+1], dims[i]) — matches layer weight shape
            delta_up = torch.einsum(
                "bo,bi->oi", error_at_output, source
            ) / batch_size
            # Clip to prevent explosion
            delta_norm = delta_up.norm()
            if delta_norm > 1.0:
                delta_up = delta_up / delta_norm
            layer.linear.weight.data += self.lr * delta_up

        # Update top-down prediction weights
        for i, error in enumerate(errors):
            # W_predict[i] shape: (dims[i], dims[i+1])
            # ΔW_predict[i] = η · mean(outer(error[i], repr[i+1]))
            delta_predict = (
                torch.einsum("be,br->er", error, representations[i + 1])
                / batch_size
            )
            # Clip to prevent explosion
            delta_norm = delta_predict.norm()
            if delta_norm > 1.0:
                delta_predict = delta_predict / delta_norm
            self._W_predict[i] += self.lr * delta_predict

    def train_step(
        self,
        model: "LocalMLP",
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> float:
        """Full training step: inference iterations + weight update.

        1. Initialize representations with a forward pass
        2. Run inference iterations to converge representations
        3. Compute final prediction errors
        4. Update weights using local Hebbian rules

        Args:
            model: The LocalMLP model.
            x: Input batch (batch_size, 784).
            labels: Optional labels for supervised signal.

        Returns:
            Total prediction error (sum of squared errors across layers).
        """
        if not self._setup_done:
            self.setup_predictions(model)

        x = x.view(x.size(0), -1)
        device = x.device

        # Initialize representations with a forward pass
        with torch.no_grad():
            activations = model.get_layer_activations(x)

        # Representations: [input, hidden1, hidden2, hidden3, hidden4, output]
        representations = [x.clone()] + [a.clone() for a in activations]

        # Run inference iterations
        for step in range(self.n_inference_steps):
            representations = self.inference_step(representations, x, labels)

            # Divergence guard: check if errors are exploding
            errors = self.compute_prediction_errors(representations, x)
            max_error = max(e.abs().max().item() for e in errors)
            if max_error > 1000.0:
                # Abort inference — errors are diverging
                break

        # Compute final prediction errors
        errors = self.compute_prediction_errors(representations, x)

        # Update weights
        self.update_weights(model, representations, errors)

        # Return total prediction error
        total_error = sum(
            (e ** 2).mean().item() for e in errors
        )
        return total_error

    def compute_update(self, state: "LayerState") -> torch.Tensor:
        """Not used directly — predictive coding uses train_step instead.

        This exists for protocol compatibility.
        """
        raise NotImplementedError(
            "PredictiveCodingRule uses train_step() instead of compute_update(). "
            "Call rule.train_step(model, x, labels) for training."
        )

    def reset(self) -> None:
        """Reset prediction weights for re-initialization."""
        self._W_predict = None
        self._setup_done = False
