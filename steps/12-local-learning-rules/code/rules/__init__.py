"""Learning rule implementations.

Note: ForwardForwardRule and PredictiveCodingRule are imported directly
from their modules to avoid circular imports (they depend on LocalMLP).
"""

from code.rules.base import LayerState, LocalLearningRule, ThirdFactorInterface
from code.rules.hebbian import HebbianRule
from code.rules.oja import OjaRule
from code.rules.three_factor import (
    ThreeFactorRule,
    RandomNoiseThirdFactor,
    GlobalRewardThirdFactor,
    LayerWiseErrorThirdFactor,
)

__all__ = [
    "LayerState",
    "LocalLearningRule",
    "ThirdFactorInterface",
    "HebbianRule",
    "OjaRule",
    "ThreeFactorRule",
    "RandomNoiseThirdFactor",
    "GlobalRewardThirdFactor",
    "LayerWiseErrorThirdFactor",
]
