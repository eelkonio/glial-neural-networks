"""Random embedding strategy.

Assigns uniformly random 3D coordinates with a configurable seed.
Deterministic: same seed and model produce identical output.
"""

import numpy as np
import torch.nn as nn


class RandomEmbedding:
    """Assigns uniformly random 3D coordinates with a fixed seed."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "random"

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute random spatial positions for all weights.

        Args:
            model: Neural network with get_weight_count() method.

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        n_weights = model.get_weight_count()
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0.0, 1.0, size=(n_weights, 3))
