"""Configuration dataclasses for Step 13 experiments.

GateConfig: Parameters for gate variant selection and tuning.
ExperimentCondition: Full specification of a single experimental condition.
"""

from dataclasses import dataclass, field

from code.calcium.config import CalciumConfig
from code.domains.config import DomainConfig


@dataclass
class GateConfig:
    """Configuration for gate variants.

    Args:
        variant: Which gate to use — "binary", "directional", or "volume_teaching".
        prediction_decay: EMA decay for activity prediction (Variant B).
        diffusion_sigma: Gaussian diffusion width (Variant C). None = mean inter-domain distance.
        gap_junction_strength: Calcium coupling between domains (Variant C).
        n_classes: Number of output classes for label projection (Variant C).
    """

    variant: str = "directional"  # "binary", "directional", "volume_teaching"
    prediction_decay: float = 0.95  # EMA decay (Variant B)
    diffusion_sigma: float | None = None  # Gaussian width (Variant C)
    gap_junction_strength: float = 0.1  # Ca coupling (Variant C)
    n_classes: int = 10


@dataclass
class ExperimentCondition:
    """A single experimental condition to run.

    Args:
        name: Human-readable condition name for results reporting.
        gate_config: Gate configuration. None for baselines (no gate).
        calcium_config: Calcium dynamics parameters.
        domain_config: Domain assignment parameters.
        learning_rate: Three-factor rule learning rate.
        tau: Eligibility trace time constant.
        use_stability_fix: Whether to apply error clipping and eligibility normalization.
        error_clip_threshold: Maximum error signal magnitude.
        eligibility_norm_threshold: Eligibility trace norm above which normalization applies.
    """

    name: str
    gate_config: GateConfig | None = None  # None for baselines
    calcium_config: CalciumConfig = field(default_factory=CalciumConfig)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    learning_rate: float = 0.01
    tau: float = 100.0
    use_stability_fix: bool = True
    error_clip_threshold: float = 10.0
    eligibility_norm_threshold: float = 100.0
