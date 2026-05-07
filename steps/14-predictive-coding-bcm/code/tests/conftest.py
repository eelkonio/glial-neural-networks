"""Hypothesis strategies for Predictive Coding + BCM property tests."""

import sys
from pathlib import Path

import torch
import hypothesis.strategies as st
from hypothesis.strategies import composite

_step14_dir = str(Path(__file__).parent.parent.parent)
if _step14_dir not in sys.path:
    sys.path.insert(0, _step14_dir)

from code.step_imports import LayerState, DomainAssignment, CalciumDynamics, CalciumConfig, DomainConfig
from code.predictive_bcm_config import PredictiveBCMConfig
from code.predictive_bcm_rule import PredictiveBCMRule


def make_rule_and_states(
    out_features=128, in_features=128, batch_size=16, domain_size=16,
    n_layers=5, config=None, device="cpu",
):
    """Create a PredictiveBCMRule and matching list of LayerStates for testing."""
    config = config or PredictiveBCMConfig()

    # Build layer sizes for a simple MLP
    layer_sizes = [(784, out_features)]  # First layer
    for _ in range(n_layers - 2):
        layer_sizes.append((out_features, out_features))
    layer_sizes.append((out_features, 10))  # Last layer

    domain_config = DomainConfig(domain_size=domain_size, mode="random", seed=42)
    domain_assignment = DomainAssignment(layer_sizes, domain_config)

    calcium_dynamics = {}
    for idx, n_domains in enumerate(domain_assignment.n_domains_per_layer):
        calcium_dynamics[idx] = CalciumDynamics(n_domains=n_domains, device=device)

    rule = PredictiveBCMRule(
        domain_assignment=domain_assignment,
        calcium_dynamics=calcium_dynamics,
        layer_sizes=layer_sizes,
        config=config,
        device=device,
    )

    # Create matching LayerStates
    states = []
    for i, (in_f, out_f) in enumerate(layer_sizes):
        pre = torch.randn(batch_size, in_f).abs()  # Non-negative (post-ReLU from prev layer)
        post = torch.randn(batch_size, out_f).abs() + 0.01  # Non-negative, non-zero
        weights = torch.randn(out_f, in_f) * 0.1
        states.append(LayerState(
            pre_activation=pre,
            post_activation=post,
            weights=weights,
            bias=None,
            layer_index=i,
        ))

    return rule, states, domain_assignment


@composite
def domain_activity_pairs(draw, n_domains=None):
    """Generate pairs of domain activity vectors (current, next)."""
    n = n_domains or draw(st.integers(min_value=4, max_value=16))
    current = torch.rand(n) * draw(st.floats(min_value=0.1, max_value=2.0))
    next_act = torch.rand(n) * draw(st.floats(min_value=0.1, max_value=2.0))
    return current, next_act, n


@composite
def prediction_matrices(draw, rows=None, cols=None):
    """Generate random prediction weight matrices."""
    r = rows or draw(st.integers(min_value=4, max_value=16))
    c = cols or draw(st.integers(min_value=4, max_value=16))
    P = torch.randn(r, c) * 0.5
    return P, r, c
