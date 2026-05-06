"""Configuration for astrocyte domain assignment."""

from dataclasses import dataclass


@dataclass
class DomainConfig:
    """Configuration for astrocyte domain assignment.

    Args:
        domain_size: Number of output neurons per domain (default 16).
        mode: Assignment strategy — "spatial" or "random" (for ablation).
        embedding_path: Path to Step 01 spectral embedding coordinates.
            If None, falls back to contiguous index partitioning.
        seed: Random seed for reproducible random assignment.
    """

    domain_size: int = 16
    mode: str = "spatial"  # "spatial" or "random"
    embedding_path: str | None = None
    seed: int = 42
