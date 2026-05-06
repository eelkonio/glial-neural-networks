"""Experimental conditions for Step 13 performance comparison.

Six conditions:
(a) three_factor_random — RandomNoiseThirdFactor baseline (Step 12)
(b) three_factor_reward — GlobalRewardThirdFactor baseline (Step 12)
(c) binary_gate — Variant A with CalciumConfig(d_serine_threshold=0.02)
(d) directional_gate — Variant B with CalciumConfig(d_serine_threshold=0.02)
(e) volume_teaching — Variant C with CalciumConfig(d_serine_threshold=0.02)
(f) backprop — Standard backpropagation baseline

All use: 784→128→128→128→128→10, FashionMNIST, batch_size=128, 50 epochs, seeds=[42,123,456]
"""

from code.calcium.config import CalciumConfig
from code.domains.config import DomainConfig
from code.experiment.config import GateConfig, ExperimentCondition


# Calcium config with threshold=0.02 so gates actually open
# (default 0.4 is too high — calcium only reaches ~0.022)
GATE_CALCIUM_CONFIG = CalciumConfig(d_serine_threshold=0.02)

# Default domain config (spatial mode, domain_size=16)
DEFAULT_DOMAIN_CONFIG = DomainConfig()


def get_condition_three_factor_random() -> ExperimentCondition:
    """Three-factor with random noise third factor (Step 12 baseline)."""
    return ExperimentCondition(
        name="three_factor_random",
        gate_config=None,  # Uses RandomNoiseThirdFactor via special handling
        calcium_config=CalciumConfig(),
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )


def get_condition_three_factor_reward() -> ExperimentCondition:
    """Three-factor with global reward third factor (Step 12 baseline)."""
    return ExperimentCondition(
        name="three_factor_reward",
        gate_config=None,  # Uses GlobalRewardThirdFactor via special handling
        calcium_config=CalciumConfig(),
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )


def get_condition_binary_gate() -> ExperimentCondition:
    """Three-factor with binary astrocyte gate (Variant A)."""
    return ExperimentCondition(
        name="binary_gate",
        gate_config=GateConfig(variant="binary"),
        calcium_config=GATE_CALCIUM_CONFIG,
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )


def get_condition_directional_gate() -> ExperimentCondition:
    """Three-factor with directional astrocyte gate (Variant B)."""
    return ExperimentCondition(
        name="directional_gate",
        gate_config=GateConfig(variant="directional"),
        calcium_config=GATE_CALCIUM_CONFIG,
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )


def get_condition_volume_teaching() -> ExperimentCondition:
    """Three-factor with volume teaching signal (Variant C)."""
    return ExperimentCondition(
        name="volume_teaching",
        gate_config=GateConfig(variant="volume_teaching"),
        calcium_config=GATE_CALCIUM_CONFIG,
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.01,
        tau=100.0,
        use_stability_fix=True,
    )


def get_condition_backprop() -> ExperimentCondition:
    """Standard backpropagation baseline."""
    return ExperimentCondition(
        name="backprop",
        gate_config=None,
        calcium_config=CalciumConfig(),
        domain_config=DEFAULT_DOMAIN_CONFIG,
        learning_rate=0.001,  # Adam default
        tau=100.0,
        use_stability_fix=False,
    )


def get_all_conditions() -> list[ExperimentCondition]:
    """Get all six experimental conditions."""
    return [
        get_condition_three_factor_random(),
        get_condition_three_factor_reward(),
        get_condition_binary_gate(),
        get_condition_directional_gate(),
        get_condition_volume_teaching(),
        get_condition_backprop(),
    ]


def get_condition_by_name(name: str) -> ExperimentCondition:
    """Get a condition by name.

    Args:
        name: Condition name string.

    Returns:
        ExperimentCondition.

    Raises:
        ValueError: If name not found.
    """
    conditions = {c.name: c for c in get_all_conditions()}
    if name not in conditions:
        raise ValueError(
            f"Unknown condition: {name}. "
            f"Available: {list(conditions.keys())}"
        )
    return conditions[name]
