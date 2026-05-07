"""Property-based tests for BCMDirectedRule.

Tests the 10 correctness properties from the design document.
Uses Hypothesis for property-based testing.
"""

import sys
import math
from pathlib import Path

import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Ensure step 12b code is importable
_step12b_dir = str(Path(__file__).parent.parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

from code.step_imports import LayerState, DomainAssignment, CalciumDynamics, CalciumConfig, DomainConfig
from code.bcm_config import BCMConfig
from code.bcm_rule import BCMDirectedRule
from code.tests.conftest import (
    layer_states,
    domain_configs,
    bcm_configs,
    activity_tensors,
    make_rule_and_state,
)


# Feature: bcm-directed-substrate, Property 1: BCM Direction is Signed
# **Validates: Requirements 1.2, 1.3, 1.4**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=16, max_value=64),
    in_features=st.integers(min_value=8, max_value=32),
    batch_size=st.integers(min_value=8, max_value=32),
)
def test_bcm_direction_is_signed(out_features, in_features, batch_size):
    """Property 1: BCM direction contains both positive and negative values.

    For varied activations and non-degenerate theta, direction contains
    both positive and negative values.
    """
    # Use a theta_init that's in the middle of typical activity range
    config = BCMConfig(
        lr=0.01,
        theta_init=0.3,
        use_d_serine=True,
        use_competition=True,
        clip_delta=100.0,  # High clip to not mask direction
    )
    rule, state, _ = make_rule_and_state(
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        config=config,
    )

    # Run a few steps to let theta settle to non-degenerate values
    for _ in range(5):
        rule.compute_update(state)

    # Now generate a state with varied activations
    # Some neurons high, some low
    post = torch.zeros(batch_size, out_features)
    post[:, :out_features // 2] = torch.rand(batch_size, out_features // 2) * 2.0 + 0.5
    post[:, out_features // 2:] = torch.rand(batch_size, out_features - out_features // 2) * 0.1

    state_varied = LayerState(
        pre_activation=torch.randn(batch_size, in_features),
        post_activation=post,
        weights=state.weights,
        bias=None,
        layer_index=0,
    )

    delta_w = rule.compute_update(state_varied)
    assert (delta_w > 0).any(), "No positive values in weight delta"
    assert (delta_w < 0).any(), "No negative values in weight delta"



# Feature: bcm-directed-substrate, Property 2: Theta Slides Toward Mean
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    n_domains=st.integers(min_value=2, max_value=16),
    n_steps=st.integers(min_value=10, max_value=50),
    constant_activity=st.floats(min_value=0.1, max_value=2.0),
    theta_decay=st.floats(min_value=0.9, max_value=0.99),
)
def test_theta_slides_toward_mean(n_domains, n_steps, constant_activity, theta_decay):
    """Property 2: Theta converges to constant activity level.

    For constant domain activity, theta should converge toward that value.
    """
    config = BCMConfig(theta_decay=theta_decay, theta_init=0.0)
    layer_sizes = [(16, n_domains * 8)]
    domain_config = DomainConfig(domain_size=8, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    calcium_dynamics = {0: CalciumDynamics(n_domains=n_domains)}
    rule = BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=config,
    )

    # Apply constant activity for n_steps
    activities = torch.full((n_domains,), constant_activity)
    for _ in range(n_steps):
        rule._update_theta(activities, layer_index=0)

    theta = rule._theta[0]

    # After many steps with constant activity, theta should approach that value
    # EMA converges: theta_final ≈ activity * (1 - decay^n) / (1 - decay) * (1-decay)
    # Simplified: theta should be between theta_init and constant_activity
    # and closer to constant_activity after many steps
    expected_converged = constant_activity
    # After n_steps, remaining gap = (theta_init - activity) * decay^n
    remaining_gap = abs(0.0 - constant_activity) * (theta_decay ** n_steps)
    actual_gap = abs(theta[0].item() - expected_converged)

    # Actual gap should be close to theoretical remaining gap
    assert actual_gap < remaining_gap + 0.01, (
        f"Theta not converging: actual_gap={actual_gap:.4f}, "
        f"expected_remaining={remaining_gap:.4f}"
    )


# Feature: bcm-directed-substrate, Property 3: D-Serine Amplifies Calcium
# **Validates: Requirements 3.1, 3.2**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=8, max_value=64),
    d_serine_boost=st.floats(min_value=0.1, max_value=5.0),
)
def test_d_serine_amplifies_calcium(out_features, d_serine_boost):
    """Property 3: Open-gate neurons get amplified, closed-gate unchanged.

    D-serine boost multiplies calcium by (1 + boost) for open domains.
    """
    domain_size = 8
    n_domains = math.ceil(out_features / domain_size)
    config = BCMConfig(d_serine_boost=d_serine_boost, use_d_serine=True)

    layer_sizes = [(16, out_features)]
    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    # Create calcium dynamics with some gates open
    calcium_config = CalciumConfig(d_serine_threshold=0.3)
    calcium_dynamics = {0: CalciumDynamics(n_domains=n_domains, config=calcium_config)}

    # Force some domains to have high calcium (gate open)
    calcium_dynamics[0].ca = torch.zeros(n_domains)
    calcium_dynamics[0].ca[0] = 0.5  # Above threshold → gate open
    # Others stay at 0 → gate closed

    rule = BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=config,
    )

    # Create calcium values
    calcium = torch.rand(out_features) * 0.5 + 0.1  # All positive

    # Apply boost
    boosted = rule._apply_d_serine_boost(calcium.clone(), layer_index=0)

    # Check: neurons in domain 0 should be amplified
    gate_open = calcium_dynamics[0].get_gate_open()
    neuron_to_domain = domain_assignment.get_neuron_to_domain(0)

    for i in range(out_features):
        domain_idx = neuron_to_domain[i].item()
        if gate_open[domain_idx]:
            expected = calcium[i] * (1.0 + d_serine_boost)
            assert torch.isclose(boosted[i], expected, rtol=1e-5), (
                f"Neuron {i} in open domain: expected {expected:.4f}, got {boosted[i]:.4f}"
            )
        else:
            assert torch.isclose(boosted[i], calcium[i], rtol=1e-5), (
                f"Neuron {i} in closed domain: expected {calcium[i]:.4f}, got {boosted[i]:.4f}"
            )


# Feature: bcm-directed-substrate, Property 4: Heterosynaptic Zero-Centering
# **Validates: Requirements 4.1**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=16, max_value=64),
    domain_size=st.sampled_from([4, 8, 16]),
)
def test_heterosynaptic_zero_centering(out_features, domain_size):
    """Property 4: After competition with strength=1.0, mean direction per domain ≈ 0."""
    assume(out_features >= domain_size * 2)  # At least 2 domains

    config = BCMConfig(competition_strength=1.0, use_competition=True)
    layer_sizes = [(16, out_features)]
    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    n_domains = domain_assignment.n_domains_per_layer[0]
    calcium_dynamics = {0: CalciumDynamics(n_domains=n_domains)}

    rule = BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=config,
    )

    # Random direction values
    direction = torch.randn(out_features)

    # Apply competition
    result = rule._apply_heterosynaptic_competition(direction.clone(), layer_index=0)

    # Check each domain has approximately zero mean
    domain_indices = domain_assignment.get_domain_indices(0)
    for d_idx, indices in enumerate(domain_indices):
        if len(indices) > 1:
            idx_t = torch.tensor(indices, dtype=torch.long)
            domain_mean = result[idx_t].mean().item()
            assert abs(domain_mean) < 1e-5, (
                f"Domain {d_idx} mean = {domain_mean:.6f}, expected ≈ 0"
            )



# Feature: bcm-directed-substrate, Property 5: Domain Partition Completeness
# **Validates: Requirements 10.1, 10.2, 10.3**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=4, max_value=128),
    domain_size=st.sampled_from([4, 8, 16, 32]),
)
def test_domain_partition_completeness(out_features, domain_size):
    """Property 5: All neurons assigned, no overlaps, correct domain count."""
    layer_sizes = [(16, out_features)]
    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    domain_indices = domain_assignment.get_domain_indices(0)
    expected_n_domains = math.ceil(out_features / domain_size)

    # Correct number of domains
    assert len(domain_indices) == expected_n_domains, (
        f"Expected {expected_n_domains} domains, got {len(domain_indices)}"
    )

    # All neurons assigned (union of all domains = all indices)
    all_indices = set()
    for indices in domain_indices:
        for idx in indices:
            assert idx not in all_indices, f"Neuron {idx} assigned to multiple domains"
            all_indices.add(idx)

    assert len(all_indices) == out_features, (
        f"Expected {out_features} neurons assigned, got {len(all_indices)}"
    )


# Feature: bcm-directed-substrate, Property 6: Output Shape Matches Weight Shape
# **Validates: Requirements 5.1, 5.2**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=8, max_value=64),
    in_features=st.integers(min_value=8, max_value=64),
    batch_size=st.integers(min_value=4, max_value=32),
)
def test_output_shape_matches_weight_shape(out_features, in_features, batch_size):
    """Property 6: compute_update returns (out_features, in_features)."""
    config = BCMConfig()
    rule, state, _ = make_rule_and_state(
        out_features=out_features,
        in_features=in_features,
        batch_size=batch_size,
        config=config,
    )

    delta_w = rule.compute_update(state)
    assert delta_w.shape == (out_features, in_features), (
        f"Expected shape ({out_features}, {in_features}), got {delta_w.shape}"
    )


# Feature: bcm-directed-substrate, Property 7: Delta Norm Bounded
# **Validates: Requirements 6.1**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=8, max_value=64),
    in_features=st.integers(min_value=8, max_value=64),
    clip_delta=st.floats(min_value=0.1, max_value=10.0),
)
def test_delta_norm_bounded(out_features, in_features, clip_delta):
    """Property 7: Frobenius norm ≤ clip_delta for all inputs."""
    config = BCMConfig(clip_delta=clip_delta, lr=1.0)  # High lr to trigger clipping
    rule, state, _ = make_rule_and_state(
        out_features=out_features,
        in_features=in_features,
        batch_size=16,
        config=config,
    )

    # Use extreme activations to trigger clipping
    state.post_activation = torch.ones_like(state.post_activation) * 10.0
    state.pre_activation = torch.ones_like(state.pre_activation) * 10.0

    delta_w = rule.compute_update(state)
    norm = delta_w.norm().item()
    assert norm <= clip_delta + 1e-5, (
        f"Delta norm {norm:.4f} exceeds clip_delta {clip_delta:.4f}"
    )


# Feature: bcm-directed-substrate, Property 8: Competition Preserves Relative Order
# **Validates: Requirements 4.2**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=16, max_value=64),
    domain_size=st.sampled_from([4, 8, 16]),
)
def test_competition_preserves_relative_order(out_features, domain_size):
    """Property 8: Neuron ordering within domain unchanged after zero-centering."""
    assume(out_features >= domain_size * 2)

    config = BCMConfig(competition_strength=1.0, use_competition=True)
    layer_sizes = [(16, out_features)]
    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    n_domains = domain_assignment.n_domains_per_layer[0]
    calcium_dynamics = {0: CalciumDynamics(n_domains=n_domains)}

    rule = BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=config,
    )

    # Create direction with distinct values (no ties)
    direction = torch.randn(out_features)

    # Apply competition
    result = rule._apply_heterosynaptic_competition(direction.clone(), layer_index=0)

    # Check relative order preserved within each domain
    domain_indices = domain_assignment.get_domain_indices(0)
    for d_idx, indices in enumerate(domain_indices):
        if len(indices) > 1:
            idx_t = torch.tensor(indices, dtype=torch.long)
            original_order = direction[idx_t].argsort()
            result_order = result[idx_t].argsort()
            assert torch.equal(original_order, result_order), (
                f"Domain {d_idx}: relative order changed after competition"
            )



# Feature: bcm-directed-substrate, Property 9: Ablation Independence
# **Validates: Requirements 9.1, 9.2, 9.3**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    out_features=st.integers(min_value=8, max_value=32),
    in_features=st.integers(min_value=8, max_value=32),
)
def test_ablation_independence(out_features, in_features):
    """Property 9: With both flags False, output independent of calcium state."""
    config_ablated = BCMConfig(use_d_serine=False, use_competition=False)

    # Create two rules with different calcium states
    layer_sizes = [(in_features, out_features)]
    domain_config = DomainConfig(domain_size=8, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    n_domains = domain_assignment.n_domains_per_layer[0]

    # Rule 1: calcium at 0
    cd1 = {0: CalciumDynamics(n_domains=n_domains)}
    rule1 = BCMDirectedRule(domain_assignment=domain_assignment, calcium_dynamics=cd1, config=config_ablated)

    # Rule 2: calcium at high values
    cd2 = {0: CalciumDynamics(n_domains=n_domains)}
    cd2[0].ca = torch.ones(n_domains) * 5.0  # High calcium
    rule2 = BCMDirectedRule(domain_assignment=domain_assignment, calcium_dynamics=cd2, config=config_ablated)

    # Same state for both
    pre = torch.randn(8, in_features)
    post = torch.randn(8, out_features).abs() + 0.01
    weights = torch.randn(out_features, in_features) * 0.1

    state = LayerState(
        pre_activation=pre,
        post_activation=post,
        weights=weights,
        bias=None,
        layer_index=0,
    )

    delta1 = rule1.compute_update(state)
    delta2 = rule2.compute_update(state)

    # Both should produce the same output (calcium state doesn't matter when ablated)
    assert torch.allclose(delta1, delta2, atol=1e-5), (
        f"Ablated outputs differ: max diff = {(delta1 - delta2).abs().max():.6f}"
    )


# Feature: bcm-directed-substrate, Property 10: Calcium Dynamics Bounded
# **Validates: Requirements 6.4**
@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    n_domains=st.integers(min_value=2, max_value=16),
    n_steps=st.integers(min_value=1, max_value=100),
    activity_scale=st.floats(min_value=0.0, max_value=100.0),
)
def test_calcium_dynamics_bounded(n_domains, n_steps, activity_scale):
    """Property 10: calcium ∈ [0, ca_max] and h ∈ [0, 1] after any step sequence."""
    config = CalciumConfig()
    cd = CalciumDynamics(n_domains=n_domains, config=config)

    for _ in range(n_steps):
        # Random activities, possibly extreme
        activities = torch.rand(n_domains) * activity_scale
        cd.step(activities)

        # Check bounds
        assert (cd.ca >= 0).all(), f"Calcium below 0: {cd.ca.min():.4f}"
        assert (cd.ca <= config.ca_max).all(), f"Calcium above ca_max: {cd.ca.max():.4f}"
        assert (cd.h >= 0).all(), f"h below 0: {cd.h.min():.4f}"
        assert (cd.h <= 1).all(), f"h above 1: {cd.h.max():.4f}"
