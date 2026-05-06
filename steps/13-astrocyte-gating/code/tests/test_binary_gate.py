"""Tests for BinaryGate (Variant A).

Properties 7, 8 from the design document.
"""

import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from code.gates.binary_gate import BinaryGate
from code.calcium.config import CalciumConfig
from code.domains.assignment import DomainAssignment
from code.domains.config import DomainConfig


# --- Fixtures ---

@pytest.fixture
def simple_setup():
    """Create a simple BinaryGate with 2 layers: 128 and 64 neurons."""
    layer_sizes = [(784, 128), (128, 64)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = BinaryGate(domain_assignment, calcium_config=calcium_config)
    return gate, domain_assignment


# --- Property Tests ---

@given(
    n_steps=st.integers(min_value=1, max_value=50),
    activity_scale=st.floats(min_value=0.0, max_value=10.0),
)
@settings(max_examples=100, deadline=None)
def test_property7_binary_threshold_semantics(n_steps, activity_scale):
    """Property 7: Binary Gate Threshold Semantics.

    Output is exactly 1.0 where Ca > threshold and 0.0 where Ca <= threshold.
    All neurons in same domain get same value.

    **Validates: Requirements 5.2, 5.4**
    """
    layer_sizes = [(784, 128)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = BinaryGate(domain_assignment, calcium_config=calcium_config)

    # Drive the gate for n_steps
    for _ in range(n_steps):
        activations = torch.rand(32, 128) * activity_scale
        signal = gate.compute_signal(activations, layer_index=0)

    # Final signal
    activations = torch.rand(32, 128) * activity_scale
    signal = gate.compute_signal(activations, layer_index=0)

    # Check output is exactly 0.0 or 1.0
    assert signal.shape == (128,)
    unique_vals = signal.unique()
    for v in unique_vals:
        assert v.item() == 0.0 or v.item() == 1.0, f"Got non-binary value: {v.item()}"

    # Check all neurons in same domain get same value
    domain_indices = domain_assignment.get_domain_indices(0)
    for indices in domain_indices:
        if len(indices) > 1:
            domain_vals = signal[torch.tensor(indices)]
            assert (domain_vals == domain_vals[0]).all(), \
                "Neurons in same domain should have same gate value"

    # Verify consistency with calcium state
    calcium = gate._calcium[0]
    ca_state = calcium.get_calcium()
    threshold = calcium_config.d_serine_threshold
    gate_open = ca_state > threshold

    neuron_to_domain = domain_assignment.get_neuron_to_domain(0)
    for neuron_idx in range(128):
        domain_idx = neuron_to_domain[neuron_idx].item()
        expected = 1.0 if gate_open[domain_idx] else 0.0
        assert signal[neuron_idx].item() == expected


@given(
    activity_scale=st.floats(min_value=0.0, max_value=5.0),
)
@settings(max_examples=100, deadline=None)
def test_property8_binary_gate_blocks_closed_domains(activity_scale):
    """Property 8: Binary Gate Blocks Closed-Domain Updates.

    Weight update is exactly zero for weights in closed domains when
    combined with ThreeFactorRule modulation logic.

    **Validates: Requirements 5.5**
    """
    layer_sizes = [(784, 128)]
    domain_config = DomainConfig(domain_size=16, mode="contiguous")
    domain_assignment = DomainAssignment(layer_sizes, config=domain_config)
    calcium_config = CalciumConfig(d_serine_threshold=0.4)
    gate = BinaryGate(domain_assignment, calcium_config=calcium_config)

    # Drive gate with some activity
    activations = torch.rand(32, 128) * activity_scale
    signal = gate.compute_signal(activations, layer_index=0)

    # Simulate ThreeFactorRule modulation: delta_w = eligibility * signal.unsqueeze(1) * lr
    eligibility = torch.randn(128, 784)
    lr = 0.01
    delta_w = eligibility * signal.unsqueeze(1) * lr

    # For closed domains (signal == 0), weight update must be exactly zero
    neuron_to_domain = domain_assignment.get_neuron_to_domain(0)
    calcium = gate._calcium[0]
    gate_open = calcium.get_gate_open()

    for domain_idx in range(len(domain_assignment.get_domain_indices(0))):
        if not gate_open[domain_idx]:
            # All neurons in this closed domain should have zero update
            domain_neurons = (neuron_to_domain == domain_idx).nonzero(as_tuple=True)[0]
            for neuron_idx in domain_neurons:
                assert (delta_w[neuron_idx] == 0.0).all(), \
                    f"Closed domain {domain_idx} neuron {neuron_idx} has non-zero update"


# --- Unit Tests ---

def test_binary_gate_output_shape(simple_setup):
    """Binary gate produces correct output shape for each layer."""
    gate, _ = simple_setup
    act_layer0 = torch.rand(16, 128)
    act_layer1 = torch.rand(16, 64)

    signal0 = gate.compute_signal(act_layer0, layer_index=0)
    signal1 = gate.compute_signal(act_layer1, layer_index=1)

    assert signal0.shape == (128,)
    assert signal1.shape == (64,)


def test_binary_gate_initially_closed(simple_setup):
    """Binary gate starts with all domains closed (Ca = 0 < threshold)."""
    gate, _ = simple_setup
    # With zero activity, calcium stays at 0
    activations = torch.zeros(16, 128)
    signal = gate.compute_signal(activations, layer_index=0)
    assert (signal == 0.0).all()


def test_binary_gate_reset(simple_setup):
    """Reset returns gate to initial state."""
    gate, _ = simple_setup
    # Drive with high activity
    for _ in range(100):
        activations = torch.rand(16, 128) * 5.0
        gate.compute_signal(activations, layer_index=0)

    gate.reset()

    # After reset, calcium should be zero → gate closed
    activations = torch.zeros(16, 128)
    signal = gate.compute_signal(activations, layer_index=0)
    assert (signal == 0.0).all()


def test_binary_gate_state_dict_roundtrip(simple_setup):
    """State dict save/load preserves calcium state."""
    gate, domain_assignment = simple_setup

    # Drive gate
    for _ in range(50):
        activations = torch.rand(16, 128) * 3.0
        gate.compute_signal(activations, layer_index=0)

    # Save state
    state = gate.state_dict()

    # Create new gate and load state
    gate2 = BinaryGate(domain_assignment, calcium_config=gate.calcium_config)
    gate2.load_state_dict(state)

    # Verify calcium states match
    ca1 = gate._calcium[0].get_calcium()
    ca2 = gate2._calcium[0].get_calcium()
    assert torch.allclose(ca1, ca2)
