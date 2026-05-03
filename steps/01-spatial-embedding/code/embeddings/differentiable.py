"""Differentiable embedding strategy.

Learnable spatial positions optimized via spatial coherence loss.
Positions are PyTorch Parameters that are jointly optimized with the
network weights. A spatial coherence loss penalizes configurations where
gradient-correlated weights are spatially distant.

This solves the chicken-and-egg problem more cleanly than the
developmental approach: gradients flow through both the task loss
and the spatial loss simultaneously.

Loss = task_loss + lambda_spatial * spatial_coherence_loss

Where spatial_coherence_loss = mean(spatial_distance(i,j) * gradient_correlation(i,j))
for sampled pairs (i,j) with positive gradient correlation.
"""

import numpy as np
import torch
import torch.nn as nn


class DifferentiableEmbedding:
    """Learnable spatial positions optimized via spatial coherence loss.

    Positions are a torch.nn.Parameter of shape (N_weights, 3).
    Initialized with uniform random values, then sigmoid is applied
    to keep positions in [0, 1].

    The embed() method returns current positions as numpy (sigmoid applied).
    This embedding is special: it's trained jointly with the model, so
    embed() can be called at any point during training.
    """

    def __init__(
        self,
        lambda_spatial: float = 0.01,
        subsample_pairs: int = 10000,
        position_lr: float = 1e-3,
    ):
        """Initialize differentiable embedding.

        Args:
            lambda_spatial: Weight for spatial coherence loss term.
            subsample_pairs: Number of weight pairs to sample for loss computation.
            position_lr: Learning rate for position parameter updates.
        """
        self.lambda_spatial = lambda_spatial
        self.subsample_pairs = subsample_pairs
        self.position_lr = position_lr
        self.positions_param: torch.nn.Parameter | None = None

    @property
    def name(self) -> str:
        return "differentiable"

    def initialize(self, n_weights: int) -> torch.nn.Parameter:
        """Create the learnable positions parameter.

        Returns the Parameter so it can be added to the optimizer.
        Initialized with uniform random values (pre-sigmoid), seeded
        for reproducibility.

        Args:
            n_weights: Number of weights in the model.

        Returns:
            torch.nn.Parameter of shape (n_weights, 3).
        """
        # Use a fixed seed for reproducible initialization
        gen = torch.Generator()
        gen.manual_seed(42)

        # Initialize with values that, after sigmoid, give uniform [0, 1]
        # Using uniform random in [-2, 2] gives sigmoid values spread across [0.12, 0.88]
        init_values = torch.empty(n_weights, 3).uniform_(-2.0, 2.0, generator=gen)
        self.positions_param = torch.nn.Parameter(init_values)
        return self.positions_param

    def compute_spatial_loss(
        self, gradients: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial coherence loss from current gradients.

        Samples pairs of weights, computes gradient correlations, and
        penalizes high-correlation pairs that are spatially distant.

        spatial_loss = mean(distance(i,j) * grad_corr(i,j))
        for pairs where grad_corr > 0.

        Args:
            gradients: Flat gradient tensor of shape (n_weights,) from
                       the current backward pass.

        Returns:
            Scalar loss tensor (differentiable w.r.t. positions_param).
        """
        if self.positions_param is None:
            raise RuntimeError(
                "Must call initialize() before compute_spatial_loss()"
            )

        n_weights = self.positions_param.shape[0]

        # Apply sigmoid to get positions in [0, 1]
        positions = torch.sigmoid(self.positions_param)

        # Sample random pairs
        n_pairs = min(self.subsample_pairs, n_weights * (n_weights - 1) // 2)

        # Use a generator for reproducible pair sampling within each call
        # but different across training steps (based on gradient content)
        idx_i = torch.randint(0, n_weights, (n_pairs,))
        idx_j = torch.randint(0, n_weights, (n_pairs,))

        # Avoid self-pairs
        same_mask = idx_i == idx_j
        idx_j[same_mask] = (idx_j[same_mask] + 1) % n_weights

        # Compute gradient correlations for sampled pairs
        # Use detached gradients (no gradient flow through correlation computation)
        grad_detached = gradients.detach()

        # Normalize gradients for correlation computation
        grad_mean = grad_detached.mean()
        grad_centered = grad_detached - grad_mean
        grad_norm = torch.norm(grad_centered)
        if grad_norm < 1e-10:
            return torch.tensor(0.0, requires_grad=True)

        # Per-weight "signal" is just the gradient value itself
        # For correlation between pairs, use product of centered/normalized values
        grad_normalized = grad_centered / grad_norm

        # Pairwise correlation approximation: product of normalized values
        # This is a simplified correlation for individual gradient snapshots
        corr_approx = grad_normalized[idx_i] * grad_normalized[idx_j] * n_weights

        # Filter to positive correlations only
        positive_mask = corr_approx > 0
        if not positive_mask.any():
            return torch.tensor(0.0, requires_grad=True)

        # Compute spatial distances for positive-correlation pairs
        pos_i = positions[idx_i[positive_mask]]
        pos_j = positions[idx_j[positive_mask]]
        distances = torch.norm(pos_i - pos_j, dim=1)

        # Spatial loss: penalize distance for correlated pairs
        corr_values = corr_approx[positive_mask]
        spatial_loss = torch.mean(distances * corr_values)

        return spatial_loss

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Return current positions as numpy array.

        Can be called at any point during training to snapshot positions.
        Applies sigmoid to ensure [0, 1] range.

        Args:
            model: Neural network (used to get weight count if not initialized).
            **kwargs: Unused for this embedding.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        if self.positions_param is None:
            # Auto-initialize if not already done
            n_weights = model.get_weight_count()
            self.initialize(n_weights)

        # Apply sigmoid to get [0, 1] positions
        with torch.no_grad():
            positions = torch.sigmoid(self.positions_param).cpu().numpy()

        return positions
