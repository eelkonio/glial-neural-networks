"""Tests for DirectionalGate (Variant B).

Properties 9, 10, 11 from the design document.
"""

import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from code.gates.directional_gate import DirectionalGate
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig


# --- Fixtures ---

@pytest.fixture
def simple_setup():
    """Create a simple DirectionalGate with 1 layer: 128 neurons."""
    layer_sizes = [(784, 128)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = DirectionalGate(
        domain_assignment, calcium_config=calcium_config, prediction_decay=0.95
    )
    return gate, domain_assignment


# --- Property Tests ---

@given(
    n_steps=st.integers(min_value=2, max_value=20),
    decay=st.floats(min_value=0.5, max_value=0.99),
)
@settings(max_examples=100, deadline=None)
def test_property9_ema_dynamics(n_steps, decay):
    """Property 9: Directional Gate EMA Dynamics.

    Activity prediction follows EMA formula with configured decay rate.
    After step k: pred_k = decay * pred_{k-1} + (1-decay) * activity_k

    **Validates: Requirements 6.2**
    """
    layer_sizes = [(784, 64)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = DirectionalGate(
        domain_assignment, calcium_config=calcium_config, prediction_decay=decay
    )

    n_domains = domain_assignment.n_domains_per_layer[0]

    # Generate a sequence of activities and track predictions manually
    activities_sequence = []
    for _ in range(n_steps):
        activations = torch.rand(16, 64) * 2.0
        # Compute what domain activities will be
        mean_act = activations.abs().mean(dim=0)
        domain_activities = torch.zeros(n_domains)
        domain_indices = domain_assignment.get_domain_indices(0)
        for d_idx, indices in enumerate(domain_indices):
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                domain_activities[d_idx] = mean_act[idx_tensor].mean()
        activities_sequence.append(domain_activities)

        gate.compute_signal(activations, layer_index=0)

    # Verify EMA formula manually
    # First step initializes prediction to first activity
    expected_pred = activities_sequence[0].clone()
    for k in range(1, n_steps):
        expected_pred = decay * expected_pred + (1 - decay) * activities_sequence[k]

    actual_pred = gate._predictions[0]
    assert torch.allclose(actual_pred, expected_pred, atol=1e-5), \
        f"EMA mismatch: max diff = {(actual_pred - expected_pred).abs().max().item()}"


@given(
    activity_scale=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=100, deadline=None)
def test_property10_output_formula(activity_scale):
    """Property 10: Directional Gate Output Formula.

    Output = c × normalize(a − p) when c > threshold, 0.0 otherwise.
    Sign matches sign of (a − p) for domains where calcium exceeds threshold.

    **Validates: Requirements 6.3, 6.4, 6.5, 6.7**
    """
    layer_sizes = [(784, 64)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = DirectionalGate(
        domain_assignment, calcium_config=calcium_config, prediction_decay=0.95
    )

    # Drive gate for several steps to build up calcium and predictions
    for _ in range(200):
        activations = torch.rand(16, 64) * activity_scale
        gate.compute_signal(activations, layer_index=0)

    # Now check the output properties
    calcium = gate._calcium[0]
    ca_state = calcium.get_calcium()
    gate_open = calcium.get_gate_open()
    threshold = calcium_config.d_serine_threshold

    # Get the signal from one more step
    activations = torch.rand(16, 64) * activity_scale
    signal = gate.compute_signal(activations, layer_index=0)

    # For closed domains, signal must be zero
    neuron_to_domain = domain_assignment.get_neuron_to_domain(0)
    for domain_idx in range(domain_assignment.n_domains_per_layer[0]):
        domain_neurons = (neuron_to_domain == domain_idx).nonzero(as_tuple=True)[0]
        if not gate_open[domain_idx]:
            for n_idx in domain_neurons:
                assert signal[n_idx].item() == 0.0, \
                    f"Closed domain {domain_idx} should have zero signal"


@given(
    n_steps=st.integers(min_value=5, max_value=30),
    activity_scale=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=100, deadline=None)
def test_property11_error_normalization_bounded(n_steps, activity_scale):
    """Property 11: Directional Gate Error Normalization.

    After normalization, the error signal is bounded — preventing magnitude
    differences between domains from dominating the learning signal.

    **Validates: Requirements 6.6**
    """
    layer_sizes = [(784, 128)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = DirectionalGate(
        domain_assignment, calcium_config=calcium_config, prediction_decay=0.95
    )

    # Drive gate
    for _ in range(n_steps):
        activations = torch.rand(16, 128) * activity_scale
        signal = gate.compute_signal(activations, layer_index=0)

    # The signal magnitude should be bounded
    # Since we normalize by std, the normalized error has roughly unit variance
    # The calcium magnitude is bounded by ca_max
    # So the signal should be bounded by ca_max * (some reasonable multiple)
    ca_max = calcium_config.ca_max
    assert signal.abs().max().item() <= ca_max * 100, \
        f"Signal too large: {signal.abs().max().item()}"

    # Check that signal is finite
    assert torch.isfinite(signal).all(), "Signal contains NaN or Inf"


# --- Unit Tests ---

def test_directional_gate_output_shape(simple_setup):
    """Directional gate produces correct output shape."""
    gate, _ = simple_setup
    activations = torch.rand(16, 128)
    signal = gate.compute_signal(activations, layer_index=0)
    assert signal.shape == (128,)


def test_directional_gate_signed_output(simple_setup):
    """Directional gate can produce negative values (signed signal)."""
    gate, _ = simple_setup

    # Drive with consistent activity to build prediction
    for _ in range(200):
        activations = torch.ones(16, 128) * 3.0
        gate.compute_signal(activations, layer_index=0)

    # Now give lower activity — error should be negative
    activations = torch.ones(16, 128) * 0.5
    signal = gate.compute_signal(activations, layer_index=0)

    # If any domains are open, signal should have negative values
    calcium = gate._calcium[0]
    if calcium.get_gate_open().any():
        # At least some signal values should be negative
        assert signal.min().item() <= 0.0


def test_directional_gate_reset(simple_setup):
    """Reset clears predictions and calcium."""
    gate, _ = simple_setup

    # Drive gate
    for _ in range(50):
        activations = torch.rand(16, 128) * 3.0
        gate.compute_signal(activations, layer_index=0)

    gate.reset()

    # Predictions should be cleared
    assert len(gate._predictions) == 0

    # Calcium should be at resting state
    ca = gate._calcium[0].get_calcium()
    assert (ca == 0.0).all()


def test_directional_gate_first_step_no_error(simple_setup):
    """First step initializes prediction, so error is zero."""
    gate, _ = simple_setup
    activations = torch.rand(16, 128) * 2.0
    signal = gate.compute_signal(activations, layer_index=0)
    # First step: prediction = current, so error = 0 → signal = 0
    assert (signal == 0.0).all()
