"""Base protocol for embedding strategies.

All embedding strategies must conform to the EmbeddingStrategy protocol:
- A `name` property for human-readable identification
- An `embed(model, **kwargs)` method returning (N_weights, 3) in [0, 1]
"""

from typing import Protocol

import numpy as np
import torch.nn as nn


class EmbeddingStrategy(Protocol):
    """Contract for all spatial embedding strategies.

    Every embedding takes a model and returns positions for all weights.
    Some embeddings also require data (correlation, developmental).
    """

    @property
    def name(self) -> str:
        """Human-readable name for results reporting."""
        ...

    def embed(self, model: nn.Module, **kwargs) -> np.ndarray:
        """Compute spatial positions for all weights in the model.

        Args:
            model: The neural network whose weights to embed.
            **kwargs: Strategy-specific arguments (e.g., data_loader, n_steps).

        Returns:
            ndarray of shape (N_weights, 3) with coordinates in [0, 1].
        """
        ...
