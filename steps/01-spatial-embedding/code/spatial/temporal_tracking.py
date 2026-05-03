"""Temporal quality tracking during training.

Tracks embedding quality at intervals during training to detect whether
an initially good embedding degrades as the network's functional structure
evolves during learning.
"""

import torch.nn as nn
from torch.utils.data import DataLoader

from code.spatial.quality import QualityMeasurement


class TemporalQualityTracker:
    """Tracks embedding quality at intervals during training.

    Detects whether an initially good embedding degrades as the network's
    functional structure evolves during learning. Critical for understanding
    whether fixed embeddings (spectral, correlation) remain valid throughout
    training or need periodic recomputation.
    """

    def __init__(self, record_interval_epochs: int = 2):
        """
        Args:
            record_interval_epochs: How often to record quality (in epochs).
        """
        self._record_interval = record_interval_epochs
        self._history: list[tuple[int, int, float]] = []

    @property
    def record_interval(self) -> int:
        """Recording interval in epochs."""
        return self._record_interval

    def record(
        self,
        epoch: int,
        step: int,
        quality_measurement: QualityMeasurement,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> float:
        """Record quality score at current training point.

        Args:
            epoch: Current epoch number.
            step: Current training step.
            quality_measurement: QualityMeasurement instance for computing score.
            model: Current model state.
            data_loader: Data loader for gradient computation.

        Returns:
            The computed quality score.
        """
        result = quality_measurement.compute_quality_score(
            model, data_loader, n_batches=50
        )
        score = result.score
        self._history.append((epoch, step, score))
        return score

    def get_trajectory(self) -> list[tuple[int, int, float]]:
        """Return full (epoch, step, score) trajectory."""
        return list(self._history)

    def detect_degradation(self, threshold: float = 0.5) -> bool:
        """Return True if quality dropped by more than threshold fraction from initial.

        The quality is considered degraded if the minimum score observed
        is less than (1 - threshold) * initial_score. For example, with
        threshold=0.5, degradation is detected if quality dropped below
        50% of the initial value.

        Args:
            threshold: Fraction of initial quality that constitutes degradation.
                      Default 0.5 means quality dropped > 50% from initial.

        Returns:
            True if quality degraded beyond threshold, False otherwise.
            Returns False if fewer than 2 recordings exist.
        """
        if len(self._history) < 2:
            return False

        initial_score = self._history[0][2]

        # Handle case where initial score is zero or near-zero
        if abs(initial_score) < 1e-12:
            return False

        # Find minimum score in trajectory
        min_score = min(score for _, _, score in self._history)

        # Check if quality dropped by more than threshold fraction
        # For positive scores: degraded if min < initial * (1 - threshold)
        # For negative scores: degraded if min is more negative by threshold fraction
        if initial_score > 0:
            return min_score < initial_score * (1.0 - threshold)
        else:
            # For negative scores, "degradation" means becoming more negative
            return min_score < initial_score * (1.0 + threshold)
