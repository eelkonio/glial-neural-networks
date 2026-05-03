"""Spatial learning rate coupling.

Modulates per-weight learning rates by averaging with spatial neighbors.
Integrates with PyTorch's Adam optimizer by scaling the per-parameter
learning rate using the KNN graph.
"""

import numpy as np
import torch

from code.spatial.knn_graph import KNNGraph


class SpatialLRCoupling:
    """Modulates per-weight learning rates by averaging with spatial neighbors.

    Integrates with PyTorch's Adam optimizer by scaling the per-parameter
    learning rate using the KNN graph. Does not break autograd — operates
    on optimizer state, not the computation graph.
    """

    def __init__(self, knn_graph: KNNGraph, alpha: float = 0.5):
        """
        Args:
            knn_graph: Pre-built KNN graph over weight positions.
            alpha: Coupling strength in [0, 1].
                   0 = no coupling (standard Adam).
                   1 = full neighbor averaging.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self._knn_graph = knn_graph
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """Coupling strength."""
        return self._alpha

    @property
    def knn_graph(self) -> KNNGraph:
        """The underlying KNN graph."""
        return self._knn_graph

    def compute_effective_lr(self, base_lr: np.ndarray) -> np.ndarray:
        """Compute spatially-coupled learning rates.

        effective_lr[i] = (1 - alpha) * base_lr[i] + alpha * mean(base_lr[neighbors[i]])

        Args:
            base_lr: (N,) array of per-weight base learning rates.

        Returns:
            (N,) array of effective learning rates, clamped to [0.01, 10.0].
        """
        base_lr = np.asarray(base_lr, dtype=np.float64)
        n = len(base_lr)

        if self._knn_graph.k == 0 or self._alpha == 0.0:
            return base_lr.copy()

        # Compute neighbor mean LR for each weight
        neighbor_indices = self._knn_graph.neighbor_indices  # (N, k)
        neighbor_lrs = base_lr[neighbor_indices]  # (N, k)
        neighbor_mean = neighbor_lrs.mean(axis=1)  # (N,)

        # Apply coupling formula
        effective_lr = (1.0 - self._alpha) * base_lr + self._alpha * neighbor_mean

        # Clamp for stability
        np.clip(effective_lr, 0.01, 10.0, out=effective_lr)

        return effective_lr

    def apply_to_optimizer(self, optimizer: torch.optim.Adam) -> None:
        """Apply spatial coupling to optimizer's per-parameter learning rates.

        Modifies optimizer param_groups in-place. Called once per training step.

        This method assumes the KNN graph was built over ALL weight parameters
        in the model (concatenated in order). It applies per-weight LR scaling
        by modifying gradients before the optimizer step.

        For fine-grained per-weight LR control, we scale the gradient
        before the optimizer step (equivalent to per-weight LR scaling).
        """
        n_graph_nodes = self._knn_graph.n_nodes

        for group in optimizer.param_groups:
            base_lr_value = group['lr']

            # Create uniform base_lr array for all weights in the graph
            base_lr_array = np.full(n_graph_nodes, base_lr_value)

            # Compute effective LR for all weights
            effective_lr = self.compute_effective_lr(base_lr_array)

            # Compute multiplier relative to base LR
            multiplier = effective_lr / base_lr_value

            # Apply multiplier to each parameter's gradients
            offset = 0
            for param in group['params']:
                if param.grad is None:
                    offset += param.numel()
                    continue

                n_weights = param.numel()

                # Only apply if this parameter fits within the graph
                if offset + n_weights <= n_graph_nodes:
                    param_multiplier = multiplier[offset:offset + n_weights]
                    multiplier_tensor = torch.from_numpy(param_multiplier).float()
                    multiplier_tensor = multiplier_tensor.reshape(param.grad.shape)
                    multiplier_tensor = multiplier_tensor.to(param.grad.device)
                    param.grad.data.mul_(multiplier_tensor)

                offset += n_weights
