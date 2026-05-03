"""Developmental embedding strategy.

Co-evolving embedding that self-organizes based on gradient correlations.

Resolution of the chicken-and-egg problem:
1. Train the model for a warmup period without spatial coupling
2. Compute gradient correlations from the partially-trained model
3. Iteratively update positions: attract correlated weights, repel uncorrelated
4. Track quality score at each step to monitor convergence

The warmup ensures meaningful gradient statistics exist before
position optimization begins.
"""

import numpy as np
import torch
import torch.nn as nn


def compute_force(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    correlation: float,
    max_force: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the force between two weight positions based on correlation.

    If correlation > 0: attractive force (moves positions closer).
    If correlation <= 0: repulsive force (moves positions apart).

    Force magnitude is proportional to |correlation| * distance, capped so
    that the force never exceeds half the distance (prevents overshooting).
    Direction is along the line connecting the two positions.

    Args:
        pos_i: Position of weight i, shape (3,).
        pos_j: Position of weight j, shape (3,).
        correlation: Gradient correlation between weights i and j.
        max_force: Maximum force magnitude (scalar, applied to force vector norm).

    Returns:
        (force_on_i, force_on_j): Forces to apply to each position.
    """
    direction = pos_j - pos_i  # Vector from i to j
    distance = np.linalg.norm(direction)

    if distance < 1e-10:
        # Positions are essentially the same; apply small random repulsion
        # to break symmetry
        rng = np.random.RandomState(0)
        force_on_i = rng.randn(3) * 0.001
        force_norm = np.linalg.norm(force_on_i)
        if force_norm > max_force:
            force_on_i = force_on_i * (max_force / force_norm)
        return force_on_i, -force_on_i

    # Normalize direction
    unit_direction = direction / distance

    # Force magnitude proportional to |correlation|, scaled by distance
    # to prevent overshooting. Cap at max_force and at half the distance.
    raw_magnitude = abs(correlation) * distance
    capped_magnitude = min(raw_magnitude, max_force, distance * 0.49)

    if correlation > 0:
        # Attractive: move i toward j, move j toward i
        force_on_i = unit_direction * capped_magnitude
        force_on_j = -unit_direction * capped_magnitude
    else:
        # Repulsive: move i away from j, move j away from i
        force_on_i = -unit_direction * capped_magnitude
        force_on_j = unit_direction * capped_magnitude

    return force_on_i, force_on_j


class DevelopmentalEmbedding:
    """Co-evolving embedding that self-organizes based on gradient correlations.

    Initializes random positions for all weights, then iteratively updates
    positions based on gradient correlations:
    - Sample pairs of weights
    - Compute gradient correlation between each pair
    - Apply attractive force for positively correlated pairs (move closer)
    - Apply repulsive force for zero/negative correlation pairs (move apart)
    - Clip forces to prevent explosion
    """

    def __init__(
        self,
        n_steps: int = 1000,
        position_lr: float = 0.01,
        n_correlation_batches: int = 10,
        record_interval: int = 50,
        subsample_pairs: int = 50000,
        max_force: float = 0.1,
    ):
        """Initialize developmental embedding.

        Args:
            n_steps: Number of position update steps.
            position_lr: Learning rate for position updates.
            n_correlation_batches: Number of batches for gradient correlation.
            record_interval: Record quality score every N steps.
            subsample_pairs: Number of weight pairs to sample per step.
            max_force: Maximum force magnitude per dimension.
        """
        self.n_steps = n_steps
        self.position_lr = position_lr
        self.n_correlation_batches = n_correlation_batches
        self.record_interval = record_interval
        self.subsample_pairs = subsample_pairs
        self.max_force = max_force
        self._convergence_history: list[float] = []

    @property
    def name(self) -> str:
        return "developmental"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute developmental spatial positions for all weights.

        Requires data_loader in kwargs. Model should be partially trained.

        Algorithm:
        1. Initialize random positions for all weights (deterministic seed)
        2. Collect gradient signals from the model
        3. For each step:
           a. Sample pairs of weights
           b. Compute gradient correlation for each pair
           c. Apply attractive/repulsive forces
           d. Update positions
           e. Record quality score at intervals
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
                "DevelopmentalEmbedding requires 'data_loader' in kwargs"
            )

        n_weights = model.get_weight_count()
        self._convergence_history = []

        # Initialize random positions deterministically
        rng = np.random.RandomState(42)
        positions = rng.uniform(0.0, 1.0, size=(n_weights, 3))

        # Collect gradient signals for correlation computation
        gradient_signals = self._collect_gradient_signals(model, data_loader)

        # Precompute normalized signals for fast correlation computation
        centered = gradient_signals - gradient_signals.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized_signals = centered / norms

        # Iterative position updates
        pair_rng = np.random.RandomState(123)  # Separate RNG for pair sampling

        for step in range(self.n_steps):
            # Sample pairs of weights
            n_pairs = min(self.subsample_pairs, n_weights * (n_weights - 1) // 2)
            pairs_i = pair_rng.randint(0, n_weights, size=n_pairs)
            pairs_j = pair_rng.randint(0, n_weights, size=n_pairs)

            # Avoid self-pairs
            same_mask = pairs_i == pairs_j
            pairs_j[same_mask] = (pairs_j[same_mask] + 1) % n_weights

            # Compute correlations for sampled pairs (vectorized)
            correlations = np.sum(
                normalized_signals[pairs_i] * normalized_signals[pairs_j],
                axis=1,
            )

            # Compute forces and update positions (vectorized)
            directions = positions[pairs_j] - positions[pairs_i]
            distances = np.linalg.norm(directions, axis=1, keepdims=True)
            distances = np.maximum(distances, 1e-10)
            unit_directions = directions / distances

            # Force magnitude: |correlation| * distance, capped
            raw_magnitudes = np.abs(correlations) * distances.flatten()
            cap1 = np.full_like(raw_magnitudes, self.max_force)
            cap2 = distances.flatten() * 0.49
            capped_magnitudes = np.minimum(raw_magnitudes, np.minimum(cap1, cap2))

            # Sign: positive correlation -> attractive (along direction)
            # negative/zero correlation -> repulsive (against direction)
            signs = np.sign(correlations)  # +1 for attractive, -1 for repulsive
            # For zero correlation, sign is 0 so force is zero (correct)
            forces_on_i = unit_directions * (signs * capped_magnitudes)[:, np.newaxis]

            # Accumulate forces per weight
            force_accumulator = np.zeros_like(positions)
            # Add forces on i (attracted toward j for positive corr)
            np.add.at(force_accumulator, pairs_i, forces_on_i)
            # Add opposite forces on j
            np.add.at(force_accumulator, pairs_j, -forces_on_i)

            # Update positions
            positions += self.position_lr * force_accumulator

            # Record quality score at intervals
            if (step + 1) % self.record_interval == 0:
                score = self._compute_quality_score(
                    positions, normalized_signals, pair_rng
                )
                self._convergence_history.append(score)

        # Normalize to [0, 1]
        positions = self._normalize(positions)
        return positions

    def get_convergence_history(self) -> list[float]:
        """Return quality scores recorded at each interval."""
        return self._convergence_history.copy()

    def _collect_gradient_signals(
        self, model: nn.Module, data_loader
    ) -> np.ndarray:
        """Collect per-weight gradient signals across batches.

        Returns:
            ndarray of shape (N_weights, n_batches) with gradient values.
        """
        model.eval()
        n_weights = model.get_weight_count()
        signals = np.zeros(
            (n_weights, self.n_correlation_batches), dtype=np.float32
        )

        criterion = torch.nn.CrossEntropyLoss()
        device = next(model.parameters()).device

        batch_iter = iter(data_loader)
        for batch_idx in range(self.n_correlation_batches):
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                inputs, targets = next(batch_iter)

            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            grads = model.get_flat_gradients()
            signals[:, batch_idx] = grads.cpu().numpy()

        return signals

    def _compute_quality_score(
        self,
        positions: np.ndarray,
        normalized_signals: np.ndarray,
        rng: np.random.RandomState,
    ) -> float:
        """Compute a quick quality score: correlation between spatial distance
        and gradient correlation for a sample of pairs.

        Args:
            positions: Current weight positions (N, 3).
            normalized_signals: Precomputed normalized gradient signals.
            rng: Random state for pair sampling.

        Returns:
            Pearson correlation between spatial distances and gradient correlations.
        """
        n_weights = positions.shape[0]
        n_sample = min(10000, n_weights * (n_weights - 1) // 2)

        # Sample pairs
        idx_i = rng.randint(0, n_weights, size=n_sample)
        idx_j = rng.randint(0, n_weights, size=n_sample)
        same_mask = idx_i == idx_j
        idx_j[same_mask] = (idx_j[same_mask] + 1) % n_weights

        # Spatial distances
        spatial_dists = np.linalg.norm(
            positions[idx_i] - positions[idx_j], axis=1
        )

        # Gradient correlations
        grad_corrs = np.sum(
            normalized_signals[idx_i] * normalized_signals[idx_j], axis=1
        )

        # Pearson correlation
        if np.std(spatial_dists) < 1e-10 or np.std(grad_corrs) < 1e-10:
            return 0.0

        corr = np.corrcoef(spatial_dists, grad_corrs)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

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
