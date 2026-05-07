"""Hypothesis strategies for BCM-directed rule property tests."""

import sys
from pathlib import Path

import torch
import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.strategies import composite

# Ensure step 12b code is importable
_step12b_dir = str(Path(__file__).parent.parent.parent)
if _step12b_dir not in sys.path:
    sys.path.insert(0, _step12b_dir)

from code.step_imports import LayerState, DomainAssignment, CalciumDynamics, CalciumConfig, DomainConfig
from code.bcm_config import BCMConfig
from code.bcm_rule import BCMDirectedRule


# --- Strategies ---

@composite
def layer_states(draw, out_features=None, in_features=None, batch_size=None):
    """Generate valid LayerState objects."""
    out_f = out_features or draw(st.integers(min_value=8, max_value=64))
    in_f = in_features or draw(st.integers(min_value=8, max_value=64))
    batch = batch_size or draw(st.integers(min_value=4, max_value=32))
    layer_idx = draw(st.integers(min_value=0, max_value=4))

    pre = torch.randn(batch, in_f)
    # Use abs to simulate ReLU output (non-negative post-activations)
    post = torch.randn(batch, out_f).abs()
    weights = torch.randn(out_f, in_f) * 0.1
    bias = torch.randn(out_f) * 0.01

    return LayerState(
        pre_activation=pre,
        post_activation=post,
        weights=weights,
        bias=bias,
        layer_index=layer_idx,
    )


@composite
def domain_configs(draw):
    """Generate valid DomainConfig."""
    domain_size = draw(st.sampled_from([4, 8, 16, 32]))
    mode = draw(st.sampled_from(["spatial", "random"]))
    seed = draw(st.integers(min_value=0, max_value=1000))
    return DomainConfig(domain_size=domain_size, mode=mode, seed=seed)


@composite
def bcm_configs(draw):
    """Generate valid BCMConfig with reasonable parameter ranges."""
    lr = draw(st.floats(min_value=0.001, max_value=0.1))
    theta_decay = draw(st.floats(min_value=0.9, max_value=0.999))
    theta_init = draw(st.floats(min_value=0.01, max_value=1.0))
    d_serine_boost = draw(st.floats(min_value=0.1, max_value=5.0))
    competition_strength = draw(st.floats(min_value=0.0, max_value=1.0))
    clip_delta = draw(st.floats(min_value=0.1, max_value=10.0))
    use_d_serine = draw(st.booleans())
    use_competition = draw(st.booleans())

    return BCMConfig(
        lr=lr,
        theta_decay=theta_decay,
        theta_init=theta_init,
        d_serine_boost=d_serine_boost,
        competition_strength=competition_strength,
        clip_delta=clip_delta,
        use_d_serine=use_d_serine,
        use_competition=use_competition,
    )


@composite
def activity_tensors(draw, size=None):
    """Generate non-negative activity tensors."""
    n = size or draw(st.integers(min_value=4, max_value=64))
    # Non-negative activities (like ReLU outputs)
    values = torch.rand(n) * draw(st.floats(min_value=0.1, max_value=5.0))
    return values


def make_rule_and_state(out_features=32, in_features=16, batch_size=8, domain_size=8, layer_index=0, config=None):
    """Helper to create a BCMDirectedRule and matching LayerState for testing."""
    config = config or BCMConfig()
    layer_sizes = [(in_features, out_features)]
    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    calcium_dynamics = {}
    for idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
        calcium_dynamics[idx] = CalciumDynamics(n_domains=n_domains)

    rule = BCMDirectedRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        config=config,
    )

    pre = torch.randn(batch_size, in_features)
    post = torch.randn(batch_size, out_features).abs() + 0.01  # Ensure non-zero
    weights = torch.randn(out_features, in_features) * 0.1

    state = LayerState(
        pre_activation=pre,
        post_activation=post,
        weights=weights,
        bias=None,
        layer_index=layer_index,
    )

    return rule, state, domain_assignment
