"""Tests for Hebbian and Oja learning rules."""

import torch
import pytest

from code.rules.base import LayerState
from code.rules.hebbian import HebbianRule
from code.rules.oja import OjaRule


def make_layer_state(
    batch_size: int = 8,
    in_features: int = 64,
    out_features: int = 32,
    seed: int = 42,
) -> LayerState:
    """Create a LayerState with random activations for testing."""
    torch.manual_seed(seed)
    return LayerState(
        pre_activation=torch.rand(batch_size, in_features),
        post_activation=torch.rand(batch_size, out_features),
        weights=torch.randn(out_features, in_features) * 0.1,
        bias=torch.zeros(out_features),
        layer_index=0,
    )


class TestHebbianRule:
    """Tests for HebbianRule."""

    def test_update_shape(self):
        """Update should match weight shape."""
        rule = HebbianRule()
        state = make_layer_state()
        delta = rule.compute_update(state)
        assert delta.shape == state.weights.shape

    def test_update_formula(self):
        """Update should equal η·mean(outer(post, pre)) - λ·weights."""
        lr, decay = 0.01, 0.001
        rule = HebbianRule(lr=lr, weight_decay=decay)
        state = make_layer_state()

        delta = rule.compute_update(state)

        # Compute expected
        hebbian = torch.einsum(
            "bo,bi->oi", state.post_activation, state.pre_activation
        )
        hebbian /= state.pre_activation.size(0)
        expected = lr * hebbian - decay * state.weights

        # Check if norm guard was triggered
        new_weights = state.weights + expected
        if torch.norm(new_weights) <= rule.max_norm:
            assert torch.allclose(delta, expected, atol=1e-6)

    def test_weight_decay_effect(self):
        """With large weights, decay should pull update toward zero."""
        rule = HebbianRule(lr=0.01, weight_decay=0.1)
        state = make_layer_state()
        # Make weights large
        state.weights = torch.ones_like(state.weights) * 10.0
        delta = rule.compute_update(state)

        # Decay term dominates: delta should be mostly negative
        # (pulling weights toward zero)
        assert delta.mean() < 0

    def test_explosion_guard(self):
        """Weights should not exceed max_norm after update."""
        rule = HebbianRule(lr=1.0, weight_decay=0.0, max_norm=100.0)
        state = make_layer_state()
        # Make weights already near the limit
        state.weights = torch.randn_like(state.weights) * 90.0

        delta = rule.compute_update(state)
        new_weights = state.weights + delta
        assert torch.norm(new_weights) <= rule.max_norm + 1e-4

    def test_reset_is_noop(self):
        """Reset should not raise."""
        rule = HebbianRule()
        rule.reset()  # Should not raise

    def test_name(self):
        """Rule should have correct name."""
        rule = HebbianRule()
        assert rule.name == "hebbian"


class TestOjaRule:
    """Tests for OjaRule."""

    def test_update_shape(self):
        """Update should match weight shape."""
        rule = OjaRule()
        state = make_layer_state()
        delta = rule.compute_update(state)
        assert delta.shape == state.weights.shape

    def test_update_formula(self):
        """Update should equal η·mean(outer(post, pre - post·w))."""
        lr = 0.01
        rule = OjaRule(lr=lr)
        state = make_layer_state()

        delta = rule.compute_update(state)

        # Compute expected
        reconstruction = state.post_activation @ state.weights
        residual = state.pre_activation - reconstruction
        expected = torch.einsum("bo,bi->oi", state.post_activation, residual)
        expected /= state.pre_activation.size(0)
        expected *= lr

        assert torch.allclose(delta, expected, atol=1e-6)

    def test_self_normalizing(self):
        """After many updates, weight norm should stay bounded."""
        rule = OjaRule(lr=0.001)
        torch.manual_seed(123)

        # Start with random weights
        weights = torch.randn(1, 64) * 0.5

        # Apply 1000 random updates
        for _ in range(1000):
            pre = torch.randn(1, 64)
            post = pre @ weights.T  # Simple linear response
            post = torch.relu(post)
            if post.sum() == 0:
                continue

            state = LayerState(
                pre_activation=pre,
                post_activation=post,
                weights=weights,
                bias=None,
                layer_index=0,
            )
            delta = rule.compute_update(state)
            weights = weights + delta

        # Norm should be bounded (Oja converges to unit norm)
        assert torch.norm(weights).item() < 2.0

    def test_reset_is_noop(self):
        """Reset should not raise."""
        rule = OjaRule()
        rule.reset()

    def test_name(self):
        """Rule should have correct name."""
        rule = OjaRule()
        assert rule.name == "oja"

    def test_zero_post_gives_zero_update(self):
        """If post-activation is zero, update should be zero."""
        rule = OjaRule()
        state = make_layer_state()
        state.post_activation = torch.zeros_like(state.post_activation)
        delta = rule.compute_update(state)
        assert torch.allclose(delta, torch.zeros_like(delta))
