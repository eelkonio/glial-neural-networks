"""Configuration for BCM-directed learning rule."""

from dataclasses import dataclass


@dataclass
class BCMConfig:
    """Configuration for BCMDirectedRule.

    Args:
        lr: Learning rate for weight updates.
        theta_decay: EMA decay for sliding threshold (0-1).
        theta_init: Initial theta value.
        d_serine_boost: Multiplicative calcium amplification when gate open.
        competition_strength: 0 = no competition, 1 = full zero-centering.
        clip_delta: Max Frobenius norm of weight delta per layer.
        use_d_serine: Ablation flag — disable D-serine gating.
        use_competition: Ablation flag — disable heterosynaptic competition.
    """

    lr: float = 0.01
    theta_decay: float = 0.99
    theta_init: float = 0.1
    d_serine_boost: float = 1.0
    competition_strength: float = 1.0
    clip_delta: float = 1.0
    use_d_serine: bool = True
    use_competition: bool = True
