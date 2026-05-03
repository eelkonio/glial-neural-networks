"""Embedding quality measurement.

Measures embedding quality as correlation between spatial distance
and gradient correlation across weight pairs. Uses random subsampling
when N_pairs > max_pairs to keep computation tractable.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from torch.utils.data import DataLoader


@dataclass
class QualityResult:
    """Result of embedding quality measurement."""

    score: float
    ci_lower: float
    ci_upper: float
    n_pairs_sampled: int
    computation_time_seconds: float


class QualityMeasurement:
    """Measures embedding quality as correlation between spatial distance
    and gradient correlation across weight pairs.

    Uses random subsampling when N_pairs > max_pairs to keep computation
    tractable for the ~200K weight network.
    """

    def __init__(
        self,
        positions: np.ndarray,
        max_pairs: int = 10_000_000,
        n_bootstrap: int = 1000,
    ):
        """
        Args:
            positions: (N, 3) array of spatial coordinates.
            max_pairs: Maximum number of pairs before subsampling.
            n_bootstrap: Number of bootstrap samples for CI.
        """
        self._positions = np.asarray(positions, dtype=np.float64)
        self._max_pairs = max_pairs
        self._n_bootstrap = n_bootstrap
        self._n_weights = positions.shape[0]

        # Determine which pairs to sample
        total_pairs = self._n_weights * (self._n_weights - 1) // 2
        self._needs_subsampling = total_pairs > max_pairs

        if self._needs_subsampling:
            self._n_pairs = max_pairs
        else:
            self._n_pairs = total_pairs

    @property
    def needs_subsampling(self) -> bool:
        """Whether subsampling is needed (total pairs > max_pairs)."""
        return self._needs_subsampling

    @property
    def n_pairs(self) -> int:
        """Number of pairs that will be sampled."""
        return self._n_pairs

    def _get_pair_indices(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Get indices of pairs to compute correlations for.

        Returns:
            Tuple of (idx_i, idx_j) arrays, each of shape (n_pairs,).
        """
        n = self._n_weights

        if not self._needs_subsampling:
            # Use all pairs
            idx_i, idx_j = np.triu_indices(n, k=1)
            return idx_i, idx_j
        else:
            # Random subsampling
            idx_i = rng.integers(0, n, size=self._n_pairs)
            idx_j = rng.integers(0, n, size=self._n_pairs)
            # Ensure i != j
            mask = idx_i == idx_j
            while mask.any():
                idx_j[mask] = rng.integers(0, n, size=mask.sum())
                mask = idx_i == idx_j
            return idx_i, idx_j

    def _compute_spatial_distances(
        self, idx_i: np.ndarray, idx_j: np.ndarray
    ) -> np.ndarray:
        """Compute Euclidean distances between pairs of positions."""
        diff = self._positions[idx_i] - self._positions[idx_j]
        return np.sqrt(np.sum(diff**2, axis=1))

    def compute_gradient_correlations(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        n_batches: int = 50,
    ) -> np.ndarray:
        """Compute pairwise gradient correlations over multiple batches.

        Uses Welford's algorithm for memory-efficient running statistics.

        Args:
            model: Neural network model with get_flat_gradients() method.
            data_loader: Data loader for computing gradients.
            n_batches: Number of batches to average over.

        Returns:
            (n_sampled_pairs,) array of Pearson correlations between
            gradient vectors of sampled weight pairs.
        """
        rng = np.random.default_rng(42)
        idx_i, idx_j = self._get_pair_indices(rng)

        # Accumulate gradient statistics using Welford's algorithm
        # We need mean and variance of gradients at each weight position
        n_weights = self._n_weights
        grad_mean = np.zeros(n_weights, dtype=np.float64)
        grad_m2 = np.zeros(n_weights, dtype=np.float64)

        # Also accumulate cross-products for correlation
        cross_sum = np.zeros(len(idx_i), dtype=np.float64)
        grad_i_mean = np.zeros(len(idx_i), dtype=np.float64)
        grad_j_mean = np.zeros(len(idx_i), dtype=np.float64)
        grad_i_sq_sum = np.zeros(len(idx_i), dtype=np.float64)
        grad_j_sq_sum = np.zeros(len(idx_i), dtype=np.float64)
        grad_ij_sum = np.zeros(len(idx_i), dtype=np.float64)

        device = next(model.parameters()).device
        model.train()

        batch_count = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            data, target = data.to(device), target.to(device)

            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            grads = model.get_flat_gradients().cpu().numpy()

            # Extract gradient values at pair positions
            gi = grads[idx_i]
            gj = grads[idx_j]

            # Running sums for Pearson correlation
            batch_count += 1
            grad_i_mean += gi
            grad_j_mean += gj
            grad_i_sq_sum += gi * gi
            grad_j_sq_sum += gj * gj
            grad_ij_sum += gi * gj

        if batch_count == 0:
            return np.zeros(len(idx_i), dtype=np.float64)

        # Compute Pearson correlation for each pair across batches
        # r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2) - sum(x)^2) * (n*sum(y^2) - sum(y)^2))
        n = batch_count
        numerator = n * grad_ij_sum - grad_i_mean * grad_j_mean
        denom_i = n * grad_i_sq_sum - grad_i_mean**2
        denom_j = n * grad_j_sq_sum - grad_j_mean**2

        denom = np.sqrt(np.maximum(denom_i * denom_j, 0.0))

        # Handle degenerate cases (suppress warning for expected zero-division)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.where(denom > 1e-12, numerator / denom, 0.0)

        # Clamp to [-1, 1]
        np.clip(correlations, -1.0, 1.0, out=correlations)

        return correlations

    def compute_quality_score(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        n_batches: int = 50,
    ) -> QualityResult:
        """Compute the embedding quality score with confidence interval.

        The quality score is the Pearson correlation between pairwise
        spatial distances and pairwise gradient correlations.

        Args:
            model: Neural network model.
            data_loader: Data loader for computing gradients.
            n_batches: Number of batches to average over.

        Returns:
            QualityResult with score, ci_lower, ci_upper, n_pairs_sampled.
        """
        start_time = time.time()

        rng = np.random.default_rng(42)
        idx_i, idx_j = self._get_pair_indices(rng)

        # Compute spatial distances
        spatial_distances = self._compute_spatial_distances(idx_i, idx_j)

        # Compute gradient correlations
        gradient_correlations = self.compute_gradient_correlations(
            model, data_loader, n_batches
        )

        # Handle degenerate cases
        if np.std(spatial_distances) < 1e-12 or np.std(gradient_correlations) < 1e-12:
            elapsed = time.time() - start_time
            return QualityResult(
                score=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                n_pairs_sampled=len(idx_i),
                computation_time_seconds=elapsed,
            )

        # Compute Pearson correlation
        score, _ = pearsonr(spatial_distances, gradient_correlations)

        # Bootstrap for 95% CI
        bootstrap_rng = np.random.default_rng(123)
        bootstrap_scores = np.empty(self._n_bootstrap)
        n_pairs = len(spatial_distances)

        for b in range(self._n_bootstrap):
            sample_idx = bootstrap_rng.integers(0, n_pairs, size=n_pairs)
            d_sample = spatial_distances[sample_idx]
            c_sample = gradient_correlations[sample_idx]

            if np.std(d_sample) < 1e-12 or np.std(c_sample) < 1e-12:
                bootstrap_scores[b] = 0.0
            else:
                bootstrap_scores[b], _ = pearsonr(d_sample, c_sample)

        ci_lower = float(np.percentile(bootstrap_scores, 2.5))
        ci_upper = float(np.percentile(bootstrap_scores, 97.5))

        elapsed = time.time() - start_time

        return QualityResult(
            score=float(score),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_pairs_sampled=len(idx_i),
            computation_time_seconds=elapsed,
        )
