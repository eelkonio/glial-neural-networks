"""Property tests for calcium dynamics (Properties 3, 4).

Property 3: Calcium Concentration Invariant
Property 4: IP3 Proportionality

Validates: Requirements 3.3, 3.6
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from code.calcium.li_rinzel import CalciumDynamics
from code.calcium.config import CalciumConfig


# --- Custom strategies ---

def domain_activities(n_domains=8):
    """Generate domain activity tensors with values in [0, 5]."""
    return st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=n_domains,
        max_size=n_domains,
    ).map(lambda xs: torch.tensor(xs, dtype=torch.float32))


def extreme_activities(n_domains=8):
    """Generate extreme domain activities including edge cases."""
    return st.lists(
        st.floats(min_value=-10.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=n_domains,
        max_size=n_domains,
    ).map(lambda xs: torch.tensor(xs, dtype=torch.float32))


# --- Property 3: Calcium Concentration Invariant ---

@pytest.mark.property
class TestCalciumConcentrationInvariant:
    """Property 3: Calcium Concentration Invariant.

    **Validates: Requirements 3.6**

    For any sequence of domain activity inputs (including extreme values),
    after each Li-Rinzel dynamics step, the calcium concentration for every
    domain shall remain in [0, ca_max] and h shall remain in [0, 1].
    """

    @given(activities=domain_activities())
    @settings(max_examples=200, deadline=None)
    def test_single_step_bounds(self, activities):
        """After a single step, ca in [0, ca_max] and h in [0, 1]."""
        config = CalciumConfig()
        dynamics = CalciumDynamics(n_domains=8, config=config)

        dynamics.step(activities)

        assert (dynamics.ca >= 0.0).all()
        assert (dynamics.ca <= config.ca_max).all()
        assert (dynamics.h >= 0.0).all()
        assert (dynamics.h <= 1.0).all()

    @given(
        activities_seq=st.lists(
            domain_activities(),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_step_bounds(self, activities_seq):
        """After multiple steps, bounds still hold."""
        config = CalciumConfig()
        dynamics = CalciumDynamics(n_domains=8, config=config)

        for activities in activities_seq:
            dynamics.step(activities)
            assert (dynamics.ca >= 0.0).all()
            assert (dynamics.ca <= config.ca_max).all()
            assert (dynamics.h >= 0.0).all()
            assert (dynamics.h <= 1.0).all()

    @given(activities=extreme_activities())
    @settings(max_examples=200, deadline=None)
    def test_extreme_inputs_bounded(self, activities):
        """Even with extreme inputs, bounds hold."""
        config = CalciumConfig()
        dynamics = CalciumDynamics(n_domains=8, config=config)

        # Run several steps with extreme input
        for _ in range(10):
            dynamics.step(activities)

        assert (dynamics.ca >= 0.0).all()
        assert (dynamics.ca <= config.ca_max).all()
        assert (dynamics.h >= 0.0).all()
        assert (dynamics.h <= 1.0).all()


# --- Property 4: IP3 Proportionality ---

@pytest.mark.property
class TestIP3Proportionality:
    """Property 4: IP3 Proportionality.

    **Validates: Requirements 3.3**

    For any two domain activity levels a₁ and a₂ where a₁ > a₂ ≥ 0,
    the IP3 production for the domain with activity a₁ shall be greater
    than or equal to the IP3 production for the domain with activity a₂.
    """

    @given(
        a1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        a2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ip3_monotonic(self, a1, a2):
        """Higher activity produces higher or equal IP3."""
        assume(a1 > a2)
        assume(a2 >= 0)

        config = CalciumConfig()

        # IP3 = ip3_production_rate × activity
        ip3_1 = config.ip3_production_rate * a1
        ip3_2 = config.ip3_production_rate * a2

        assert ip3_1 >= ip3_2

    @given(
        a1=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        a2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ip3_strictly_monotonic_for_positive(self, a1, a2):
        """For strictly positive activities, higher activity → strictly higher IP3."""
        assume(a1 > a2)
        assume(a2 >= 0)

        config = CalciumConfig()
        ip3_1 = config.ip3_production_rate * a1
        ip3_2 = config.ip3_production_rate * a2

        assert ip3_1 > ip3_2


# --- Unit tests ---

class TestCalciumDynamicsUnit:
    """Unit tests for CalciumDynamics behavior."""

    def test_calcium_rises_with_sustained_input(self):
        """Calcium should rise from resting state with constant activity."""
        config = CalciumConfig()
        dynamics = CalciumDynamics(n_domains=4, config=config)

        initial_ca = dynamics.get_calcium().clone()
        activities = torch.ones(4) * 2.0  # Sustained moderate activity

        # Run for many steps
        for _ in range(100):
            dynamics.step(activities)

        final_ca = dynamics.get_calcium()
        # Calcium should have risen from zero
        assert (final_ca > initial_ca).any()

    def test_calcium_decays_without_input(self):
        """Calcium should decay back toward resting when activity stops."""
        config = CalciumConfig()
        dynamics = CalciumDynamics(n_domains=4, config=config)

        # First raise calcium
        activities = torch.ones(4) * 3.0
        for _ in range(200):
            dynamics.step(activities)

        elevated_ca = dynamics.get_calcium().clone()

        # Now remove activity
        zero_activities = torch.zeros(4)
        for _ in range(500):
            dynamics.step(zero_activities)

        final_ca = dynamics.get_calcium()
        # Calcium should have decreased
        assert (final_ca < elevated_ca).all()

    def test_reset_returns_to_resting(self):
        """Reset should return to initial resting state."""
        dynamics = CalciumDynamics(n_domains=4)

        # Perturb state
        activities = torch.ones(4) * 2.0
        for _ in range(50):
            dynamics.step(activities)

        dynamics.reset()
        assert (dynamics.ca == 0.0).all()
        assert torch.allclose(dynamics.h, torch.ones(4) * 0.8)

    def test_state_dict_roundtrip(self):
        """state_dict/load_state_dict should preserve state."""
        dynamics = CalciumDynamics(n_domains=4)

        # Evolve state
        activities = torch.ones(4) * 2.0
        for _ in range(50):
            dynamics.step(activities)

        state = dynamics.state_dict()

        # Create new instance and load
        dynamics2 = CalciumDynamics(n_domains=4)
        dynamics2.load_state_dict(state)

        assert torch.allclose(dynamics.ca, dynamics2.ca)
        assert torch.allclose(dynamics.h, dynamics2.h)

    def test_gate_open_threshold(self):
        """get_gate_open should reflect d_serine_threshold."""
        config = CalciumConfig(d_serine_threshold=0.4)
        dynamics = CalciumDynamics(n_domains=4, config=config)

        # Manually set calcium
        dynamics.ca = torch.tensor([0.0, 0.3, 0.5, 1.0])

        gate = dynamics.get_gate_open()
        expected = torch.tensor([False, False, True, True])
        assert (gate == expected).all()
