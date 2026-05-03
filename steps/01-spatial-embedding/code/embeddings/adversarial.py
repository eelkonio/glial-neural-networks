"""Adversarial embedding strategy (negative control).

Deliberately anti-correlated embedding: computes gradient correlations from
a partially-trained model, then assigns positions that MAXIMIZE spatial
distance between highly-correlated weight pairs.

This is the negative end of the three-point validation curve:
adversarial (should hurt) → random (neutral) → good (should help).

If spatial coupling with this embedding hurts performance, it confirms
that spatial structure matters directionally, not just as regularization.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import MDS


class AdversarialEmbedding:
    """Embeds weights so that correlated weights are maximally distant.

    Computes gradient correlations from a partially-trained model, then
    uses MDS on the NEGATED correlation matrix so that highly correlated
    weight pairs get maximally distant positions.

    The resulting embedding should produce a POSITIVE quality score
    (spatial distance positively correlates with gradient correlation —
    correlated weights are far apart).
    """

    def __init__(
        self,
        n_correlation_batches: int = 10,
        subsample_size: int = 5000,
    ):
        """Initialize adversarial embedding.

        Args:
            n_correlation_batches: Number of data batches for correlation estimation.
            subsample_size: Number of weights to subsample for MDS (tractability).
        """
        self.n_correlation_batches = n_correlation_batches
        self.subsample_size = subsample_size

    @property
    def name(self) -> str:
        return "adversarial"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute adversarial spatial positions for all weights.

        Requires data_loader in kwargs. Model should be partially trained.

        Algorithm:
        1. Compute gradient correlations between weight pairs
        2. Use MDS on NEGATED correlation matrix (anti-MDS)
           - Distance = 1 - |correlation| becomes distance = |correlation|
           - Highly correlated weights get maximally distant positions
        3. Interpolate remaining weight positions from subsampled set
        4. Normalize to [0, 1]

        Args:
            model: Neural network with get_weight_count() and weight_layers.
            **kwargs: Must include 'data_loader'.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        data_loader = kwargs.get("data_loader")
        if data_loader is None:
            raise ValueError(
                "AdversarialEmbedding requires 'data_loader' in kwargs"
            )

        n_weights = model.get_weight_count()

        # Collect gradient signals for all weights across batches
        gradient_signals = self._collect_gradient_signals(model, data_loader)

        # Subsample weights for MDS
        rng = np.random.RandomState(42)
        subsample_size = min(self.subsample_size, n_weights)
        subsample_indices = rng.choice(
            n_weights, size=subsample_size, replace=False
        )
        subsample_indices.sort()

        # Compute correlation matrix for subsampled weights
        sub_signals = gradient_signals[subsample_indices]
        corr_matrix = self._compute_correlation_matrix(sub_signals)

        # NEGATED distance matrix: d = |correlation|
        # Highly correlated weights get LARGE distances → placed far apart
        distance_matrix = np.abs(corr_matrix)
        # Ensure symmetry and zero diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)

        # MDS embedding of subsampled weights into 3D
        mds = MDS(
            n_components=3,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress="auto",
            max_iter=300,
            n_init=1,
        )
        sub_positions = mds.fit_transform(distance_matrix)

        # Interpolate remaining weight positions
        all_positions = self._interpolate_positions(
            gradient_signals, subsample_indices, sub_positions, n_weights, rng
        )

        # Normalize to [0, 1]
        all_positions = self._normalize(all_positions)
        return all_positions

    def _collect_gradient_signals(
        self, model: nn.Module, data_loader
    ) -> np.ndarray:
        """Collect per-weight gradient magnitudes across batches.

        Returns:
            ndarray of shape (N_weights, n_batches) with gradient signals.
        """
        model.eval()
        n_weights = model.get_weight_count()
        signals = np.zeros(
            (n_weights, self.n_correlation_batches), dtype=np.float32
        )

        criterion = torch.nn.CrossEntropyLoss()

        batch_iter = iter(data_loader)
        for batch_idx in range(self.n_correlation_batches):
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                inputs, targets = next(batch_iter)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grads = model.get_flat_gradients()
            signals[:, batch_idx] = grads.cpu().numpy()

        return signals

    def _compute_correlation_matrix(self, signals: np.ndarray) -> np.ndarray:
        """Compute pairwise Pearson correlation matrix.

        Args:
            signals: (n_weights, n_batches) array of gradient signals.

        Returns:
            (n_weights, n_weights) correlation matrix.
        """
        centered = signals - signals.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = centered / norms
        corr_matrix = normalized @ normalized.T
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        return corr_matrix

    def _interpolate_positions(
        self,
        gradient_signals: np.ndarray,
        subsample_indices: np.ndarray,
        sub_positions: np.ndarray,
        n_weights: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Interpolate positions for non-subsampled weights.

        Uses nearest-neighbor in correlation space: for each non-subsampled
        weight, find the most correlated subsampled weight and assign its
        position (with small jitter).

        For adversarial embedding, we want correlated weights far apart,
        so we assign the position of the LEAST correlated subsampled weight.
        """
        all_positions = np.zeros((n_weights, 3), dtype=np.float64)

        # Place subsampled weights at their MDS positions
        all_positions[subsample_indices] = sub_positions

        # For non-subsampled weights, find the LEAST correlated subsampled
        # neighbor (to maintain the adversarial property)
        sub_signals = gradient_signals[subsample_indices]

        # Precompute normalized subsampled signals
        sub_centered = sub_signals - sub_signals.mean(axis=1, keepdims=True)
        sub_norms = np.linalg.norm(sub_centered, axis=1, keepdims=True)
        sub_norms = np.maximum(sub_norms, 1e-10)
        sub_normalized = sub_centered / sub_norms

        # Create mask for non-subsampled indices
        is_subsampled = np.zeros(n_weights, dtype=bool)
        is_subsampled[subsample_indices] = True

        non_sub_indices = np.where(~is_subsampled)[0]
        chunk_size = 5000

        for chunk_start in range(0, len(non_sub_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(non_sub_indices))
            chunk_indices = non_sub_indices[chunk_start:chunk_end]

            chunk_signals = gradient_signals[chunk_indices]
            chunk_centered = chunk_signals - chunk_signals.mean(
                axis=1, keepdims=True
            )
            chunk_norms = np.linalg.norm(chunk_centered, axis=1, keepdims=True)
            chunk_norms = np.maximum(chunk_norms, 1e-10)
            chunk_normalized = chunk_centered / chunk_norms

            # Compute correlations with subsampled weights
            correlations = chunk_normalized @ sub_normalized.T

            # Find LEAST correlated neighbor (lowest absolute correlation)
            # to maintain adversarial property: assign position of the
            # weight that is least related
            nearest_sub_idx = np.argmin(np.abs(correlations), axis=1)

            # Assign positions with small jitter
            jitter = rng.randn(len(chunk_indices), 3) * 0.001
            all_positions[chunk_indices] = (
                sub_positions[nearest_sub_idx] + jitter
            )

        return all_positions

    def _normalize(self, positions: np.ndarray) -> np.ndarray:
        """Normalize positions to [0, 1] range per dimension."""
        for dim in range(3):
            col = positions[:, dim]
            rng_val = col.max() - col.min()
            if rng_val > 0:
                positions[:, dim] = (col - col.min()) / rng_val
            else:
                positions[:, dim] = 0.5
        return positions
