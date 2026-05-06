"""Li-Rinzel two-variable calcium dynamics model.

Implements the Li-Rinzel model vectorized over N astrocyte domains.
Each domain has independent calcium dynamics driven by neural activity
(IP3 production proportional to domain activity).

State variables (per domain):
    ca: cytoplasmic calcium concentration [Ca²⁺] (μM equivalent, [0, ca_max])
    h: IP3 receptor inactivation fraction (dimensionless, [0, 1])

Fluxes:
    J_channel: IP3-dependent release from ER (CICR)
    J_pump: SERCA pump reuptake into ER
    J_leak: Passive ER leak
"""

import torch
from torch import Tensor

from code.calcium.config import CalciumConfig


class CalciumDynamics:
    """Li-Rinzel calcium dynamics vectorized over N domains.

    Args:
        n_domains: Number of astrocyte domains.
        config: CalciumConfig with all parameters.
        device: Torch device for computation.
    """

    def __init__(
        self,
        n_domains: int,
        config: CalciumConfig | None = None,
        device: str = "cpu",
    ):
        self.n_domains = n_domains
        self.config = config or CalciumConfig()
        self.device = device

        # State variables — start at resting state
        self.ca = torch.zeros(n_domains, device=device)  # Resting [Ca²⁺] ≈ 0
        self.h = torch.ones(n_domains, device=device) * 0.8  # High initial h (not inactivated)

    def step(self, domain_activities: Tensor) -> Tensor:
        """Advance calcium dynamics by one timestep.

        Args:
            domain_activities: Mean absolute activation per domain (n_domains,).
                Should be non-negative.

        Returns:
            Current calcium state tensor (n_domains,).
        """
        cfg = self.config

        # Ensure activities are non-negative
        activities = domain_activities.clamp(min=0.0)

        # IP3 production proportional to domain activity (glutamate spillover)
        ip3 = cfg.ip3_production_rate * activities

        # Steady-state activation variables
        m_inf = ip3 / (ip3 + cfg.ip3_receptor_d1 + 1e-8)  # IP3 activation
        n_inf = self.ca / (self.ca + cfg.ip3_receptor_d5 + 1e-8)  # Ca²⁺ activation
        h_inf = cfg.d2 / (cfg.d2 + self.ca + 1e-8)  # Steady-state inactivation
        tau_h = 1.0 / (cfg.a2 * (cfg.d2 + self.ca) + 1e-8)  # Inactivation time constant

        # Channel open probability: (m_inf × n_inf × h)³
        open_prob = (m_inf * n_inf * self.h) ** 3

        # ER calcium driving force
        er_driving = cfg.c0 - (1 + cfg.c1) * self.ca

        # Fluxes
        j_channel = cfg.c1 * open_prob * er_driving  # IP3-dependent ER release (CICR)
        j_pump = cfg.serca_pump_rate * self.ca ** 2 / (cfg.K_pump ** 2 + self.ca ** 2 + 1e-8)  # SERCA uptake
        j_leak = cfg.er_leak_rate * er_driving  # Passive ER leak

        # Derivatives
        dca_dt = j_channel - j_pump + j_leak
        dh_dt = (h_inf - self.h) / (tau_h + 1e-8)

        # Euler integration
        self.ca = self.ca + dca_dt * cfg.dt
        self.h = self.h + dh_dt * cfg.dt

        # Clamp to physiological bounds
        self.ca = self.ca.clamp(0.0, cfg.ca_max)
        self.h = self.h.clamp(0.0, 1.0)

        return self.ca.clone()

    def get_calcium(self) -> Tensor:
        """Current calcium concentration per domain (n_domains,)."""
        return self.ca.clone()

    def get_gate_open(self) -> Tensor:
        """Boolean mask: which domains have Ca > threshold (n_domains,)."""
        return self.ca > self.config.d_serine_threshold

    def reset(self) -> None:
        """Reset calcium and h to resting state."""
        self.ca = torch.zeros(self.n_domains, device=self.device)
        self.h = torch.ones(self.n_domains, device=self.device) * 0.8

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "ca": self.ca.clone(),
            "h": self.h.clone(),
            "n_domains": self.n_domains,
            "config": self.config,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.ca = state["ca"].to(self.device)
        self.h = state["h"].to(self.device)
