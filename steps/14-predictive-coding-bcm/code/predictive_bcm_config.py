"""Configuration for Predictive Coding + BCM learning rule."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictiveBCMConfig:
    """Configuration for PredictiveBCMRule.

    Frozen dataclass to prevent accidental mutation during training.

    Args:
        lr: Main learning rate for weight updates.
        lr_pred: Prediction weight learning rate (separate, typically higher).
        theta_decay: BCM sliding threshold EMA decay (0-1).
        theta_init: Initial theta value.
        d_serine_boost: Multiplicative calcium amplification when gate open.
        competition_strength: 0 = no competition, 1 = full zero-centering.
        clip_delta: Max Frobenius norm of weight delta per layer.
        clip_pred_delta: Max Frobenius norm of prediction weight delta.
        combination_mode: How to combine BCM direction with information signal.
            "multiplicative" (default), "additive", or "threshold".
        use_d_serine: Ablation flag — disable D-serine gating.
        use_competition: Ablation flag — disable heterosynaptic competition.
        use_domain_modulation: Ablation flag — disable surprise-based LR modulation.
        learn_predictions: Whether prediction weights are updated during training.
        max_surprise_amplification: Maximum LR amplification for surprised domains.
        granularity: Operating level — "domain" (8-dim, primary) or "neuron" (128-dim).
        fixed_predictions: Use fixed random prediction weights (feedback alignment style).
    """

    lr: float = 0.01
    lr_pred: float = 0.1
    theta_decay: float = 0.99
    theta_init: float = 0.1
    d_serine_boost: float = 1.0
    competition_strength: float = 1.0
    clip_delta: float = 1.0
    clip_pred_delta: float = 0.5
    combination_mode: str = "multiplicative"
    use_d_serine: bool = True
    use_competition: bool = True
    use_domain_modulation: bool = True
    learn_predictions: bool = True
    max_surprise_amplification: float = 3.0
    granularity: str = "domain"
    fixed_predictions: bool = False
