"""Configuration for Li-Rinzel calcium dynamics."""

from dataclasses import dataclass


@dataclass
class CalciumConfig:
    """Configuration for Li-Rinzel calcium dynamics.

    All parameters with defaults from the design document.
    Based on standard Li-Rinzel model parameters from computational
    neuroscience literature.
    """

    # IP3 production
    ip3_production_rate: float = 0.5  # IP3 produced per unit domain activity

    # D-serine threshold
    d_serine_threshold: float = 0.4  # Ca level above which gate opens

    # Pump and leak rates
    serca_pump_rate: float = 0.9  # SERCA pump maximum rate
    er_leak_rate: float = 0.02  # Passive ER leak rate

    # IP3 receptor constants
    ip3_receptor_d1: float = 0.13  # IP3 dissociation constant
    ip3_receptor_d5: float = 0.08  # Ca²⁺ inhibition constant

    # Integration
    dt: float = 0.01  # Integration timestep
    ca_max: float = 10.0  # Physiological maximum calcium

    # Li-Rinzel constants
    c0: float = 2.0  # Total calcium (cytoplasm + ER)
    c1: float = 0.185  # ER/cytoplasm volume ratio
    a2: float = 0.2  # IP3R inactivation rate
    d2: float = 1.049  # Ca²⁺ inactivation dissociation constant
    K_pump: float = 0.1  # SERCA half-activation concentration
