"""Forward-Forward Algorithm — Hinton (2022) layer-local learning.

Each layer independently maximizes "goodness" (sum of squared activations)
for positive data and minimizes it for negative data. No backward pass
through the full network is needed.

Per-layer loss: L = -log(σ(G_pos - θ)) - log(σ(θ - G_neg))
where G = Σ(h²) is the goodness and θ is a threshold.

Classification: find the label embedding that produces the highest
cumulative goodness across all layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.data.fashion_mnist import embed_label, generate_negative

if TYPE_CHECKING:
    from code.network.local_mlp import LocalMLP


class ForwardForwardRule:
    """Forward-Forward algorithm implementation.

    Each layer has its own Adam optimizer and learns to distinguish
    positive (correctly labeled) from negative (incorrectly labeled) data
    by maximizing/minimizing goodness respectively.

    Attributes:
        name: Human-readable identifier.
        lr: Learning rate for per-layer Adam optimizers.
        threshold: Goodness threshold θ (auto-computed from first batch if None).
        n_classes: Number of classes for label embedding.
    """

    name = "forward_forward"

    def __init__(
        self,
        lr: float = 0.03,
        threshold: float | None = None,
        n_classes: int = 10,
    ):
        self.lr = lr
        self.threshold = threshold
        self._threshold_initialized = threshold is not None
        self.n_classes = n_classes
        self._optimizers: list[torch.optim.Adam] | None = None
        self._layer_norms: nn.ModuleList | None = None

    def setup_optimizers(self, model: "LocalMLP") -> None:
        """Create per-layer Adam optimizers.

        Args:
            model: The LocalMLP model to optimize.
        """
        self._optimizers = []
        for layer in model.layers:
            opt = torch.optim.Adam(layer.parameters(), lr=self.lr)
            self._optimizers.append(opt)

        # Create layer norms for normalization between layers
        self._layer_norms = nn.ModuleList()
        for layer in model.layers:
            out_features = layer.linear.out_features
            self._layer_norms.append(nn.LayerNorm(out_features))

        # Move layer norms to same device as model
        device = next(model.parameters()).device
        self._layer_norms = self._layer_norms.to(device)

    def compute_goodness(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute goodness as sum of squared activations.

        G = Σ(h²) summed over the feature dimension.

        Args:
            activations: Layer output (batch_size, features).

        Returns:
            Goodness per sample (batch_size,).
        """
        return (activations ** 2).sum(dim=-1)

    def _forward_one_layer(
        self, layer: nn.Module, x: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Forward through one layer with normalization.

        Args:
            layer: The LocalLayer to forward through.
            x: Input tensor.
            layer_idx: Index of this layer.

        Returns:
            Normalized output activations.
        """
        # Forward without detach so we can compute gradients within this layer
        out = layer(x, detach=False)
        return out

    def _normalize_for_next_layer(
        self, activations: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Normalize activations before passing to next layer.

        Uses layer normalization to prevent goodness from trivially
        increasing through the network.

        Args:
            activations: Layer output (batch_size, features).
            layer_idx: Index of the current layer.

        Returns:
            Normalized activations (detached for next layer).
        """
        if self._layer_norms is not None:
            normalized = self._layer_norms[layer_idx](activations)
        else:
            normalized = F.layer_norm(
                activations, [activations.shape[-1]]
            )
        return normalized.detach()

    def train_step(
        self,
        model: "LocalMLP",
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> list[float]:
        """Train all layers for one batch.

        Each layer independently maximizes goodness for positive data
        and minimizes it for negative data.

        Args:
            model: The LocalMLP model.
            x_pos: Positive samples with correct label embedded (batch, 784).
            x_neg: Negative samples with wrong label embedded (batch, 784).

        Returns:
            List of per-layer losses.
        """
        if self._optimizers is None:
            self.setup_optimizers(model)

        losses = []
        h_pos = x_pos
        h_neg = x_neg

        for i, (layer, optimizer) in enumerate(
            zip(model.layers, self._optimizers)
        ):
            # Forward positive and negative through this layer
            h_pos_out = layer(h_pos, detach=False)
            h_neg_out = layer(h_neg, detach=False)

            # Compute goodness
            g_pos = self.compute_goodness(h_pos_out)
            g_neg = self.compute_goodness(h_neg_out)

            # Auto-compute threshold from first batch
            if not self._threshold_initialized:
                with torch.no_grad():
                    self.threshold = (g_pos.mean() + g_neg.mean()).item() / 2.0
                self._threshold_initialized = True

            # Per-layer loss: -log(σ(G_pos - θ)) - log(σ(θ - G_neg))
            loss = (
                -F.logsigmoid(g_pos - self.threshold).mean()
                - F.logsigmoid(self.threshold - g_neg).mean()
            )

            # Backward and update (local to this layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Normalize and detach for next layer
            h_pos = self._normalize_for_next_layer(h_pos_out.detach(), i)
            h_neg = self._normalize_for_next_layer(h_neg_out.detach(), i)

        return losses

    def classify(self, model: "LocalMLP", x: torch.Tensor) -> torch.Tensor:
        """Classify by finding label with highest cumulative goodness.

        For each possible label, embed it into the input and compute
        the total goodness across all layers. The label with the
        highest cumulative goodness is the prediction.

        Args:
            model: The trained LocalMLP model.
            x: Input images (batch_size, 784) — raw, without label embedding.

        Returns:
            Predicted labels (batch_size,).
        """
        device = x.device
        batch_size = x.size(0)

        # Try each possible label
        all_goodness = torch.zeros(batch_size, self.n_classes, device=device)

        for label_idx in range(self.n_classes):
            # Create label tensor for this candidate
            candidate_labels = torch.full(
                (batch_size,), label_idx, dtype=torch.long, device=device
            )
            # Embed this candidate label into the input
            x_labeled = embed_label(x, candidate_labels, self.n_classes)

            # Forward through all layers, accumulating goodness
            h = x_labeled
            cumulative_goodness = torch.zeros(batch_size, device=device)

            with torch.no_grad():
                for i, layer in enumerate(model.layers):
                    h = layer(h, detach=False)
                    cumulative_goodness += self.compute_goodness(h)
                    # Normalize for next layer
                    h = self._normalize_for_next_layer(h, i)

            all_goodness[:, label_idx] = cumulative_goodness

        # Return label with highest cumulative goodness
        return all_goodness.argmax(dim=1)

    def compute_update(self, state: "LayerState") -> torch.Tensor:
        """Not used directly — forward-forward uses train_step instead.

        This exists for protocol compatibility but raises an error
        since FF uses its own training loop with per-layer optimizers.
        """
        raise NotImplementedError(
            "ForwardForwardRule uses train_step() instead of compute_update(). "
            "Call rule.train_step(model, x_pos, x_neg) for training."
        )

    def reset(self) -> None:
        """Reset threshold for re-initialization."""
        if not self._threshold_initialized or self.threshold is None:
            self._threshold_initialized = False
