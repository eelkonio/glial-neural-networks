# Embedding strategies for spatial coordinate assignment

from code.embeddings.adversarial import AdversarialEmbedding
from code.embeddings.base import EmbeddingStrategy
from code.embeddings.correlation import CorrelationEmbedding
from code.embeddings.developmental import DevelopmentalEmbedding
from code.embeddings.differentiable import DifferentiableEmbedding
from code.embeddings.layered_clustered import LayeredClusteredEmbedding
from code.embeddings.linear import LinearEmbedding
from code.embeddings.random import RandomEmbedding
from code.embeddings.spectral import SpectralEmbedding

__all__ = [
    "AdversarialEmbedding",
    "CorrelationEmbedding",
    "DevelopmentalEmbedding",
    "DifferentiableEmbedding",
    "EmbeddingStrategy",
    "LayeredClusteredEmbedding",
    "LinearEmbedding",
    "RandomEmbedding",
    "SpectralEmbedding",
    "get_all_strategies",
]


def get_all_strategies() -> list[EmbeddingStrategy]:
    """Return instances of all 8 embedding strategies.

    Returns a list of embedding strategy instances with default parameters.
    Data-dependent embeddings (correlation, developmental, adversarial) use
    reduced parameters suitable for quick experiments.

    Returns:
        List of 8 EmbeddingStrategy instances.
    """
    return [
        LinearEmbedding(),
        RandomEmbedding(seed=42),
        SpectralEmbedding(),
        LayeredClusteredEmbedding(),
        CorrelationEmbedding(n_batches=5, subsample_size=500),
        DevelopmentalEmbedding(n_steps=100, subsample_pairs=5000, n_correlation_batches=5),
        AdversarialEmbedding(n_correlation_batches=5, subsample_size=500),
        DifferentiableEmbedding(lambda_spatial=0.01, subsample_pairs=5000),
    ]
