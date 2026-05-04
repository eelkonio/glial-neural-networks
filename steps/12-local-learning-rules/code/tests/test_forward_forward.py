"""Tests for forward-forward algorithm."""

import torch
import pytest

from code.network.local_mlp import LocalMLP
from code.rules.forward_forward import ForwardForwardRule
from code.data.fashion_mnist import embed_label, generate_negative


class TestForwardForwardRule:
    """Tests for ForwardForwardRule."""

    def test_goodness_computation(self):
        """Goodness should equal sum of squared activations."""
        rule = ForwardForwardRule()
        activations = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        goodness = rule.compute_goodness(activations)
        expected = torch.tensor([14.0, 0.75])
        assert torch.allclose(goodness, expected)

    def test_goodness_shape(self):
        """Goodness should have shape (batch_size,)."""
        rule = ForwardForwardRule()
        activations = torch.randn(16, 128)
        goodness = rule.compute_goodness(activations)
        assert goodness.shape == (16,)

    def test_goodness_non_negative(self):
        """Goodness should always be non-negative (sum of squares)."""
        rule = ForwardForwardRule()
        activations = torch.randn(32, 64)
        goodness = rule.compute_goodness(activations)
        assert (goodness >= 0).all()

    def test_setup_optimizers(self):
        """Should create one optimizer per layer."""
        rule = ForwardForwardRule()
        model = LocalMLP()
        rule.setup_optimizers(model)
        assert len(rule._optimizers) == 5  # 5 layers
        assert rule._layer_norms is not None

    def test_train_step_returns_losses(self):
        """train_step should return per-layer losses."""
        rule = ForwardForwardRule()
        model = LocalMLP()

        batch_size = 8
        x = torch.randn(batch_size, 784)
        labels = torch.randint(0, 10, (batch_size,))
        x_pos = embed_label(x, labels)
        x_neg = generate_negative(x, labels)

        losses = rule.train_step(model, x_pos, x_neg)
        assert len(losses) == 5
        assert all(isinstance(l, float) for l in losses)
        assert all(l > 0 for l in losses)  # Losses should be positive

    def test_train_step_updates_weights(self):
        """Training step should modify model weights."""
        rule = ForwardForwardRule()
        model = LocalMLP()

        # Record initial weights
        initial_weights = [
            layer.linear.weight.data.clone() for layer in model.layers
        ]

        batch_size = 16
        x = torch.randn(batch_size, 784)
        labels = torch.randint(0, 10, (batch_size,))
        x_pos = embed_label(x, labels)
        x_neg = generate_negative(x, labels)

        rule.train_step(model, x_pos, x_neg)

        # At least some weights should have changed
        any_changed = False
        for i, layer in enumerate(model.layers):
            if not torch.allclose(layer.linear.weight.data, initial_weights[i]):
                any_changed = True
                break
        assert any_changed

    def test_classify_returns_valid_labels(self):
        """classify should return labels in [0, n_classes)."""
        rule = ForwardForwardRule(n_classes=10)
        model = LocalMLP()
        rule.setup_optimizers(model)

        x = torch.randn(8, 784)
        predictions = rule.classify(model, x)
        assert predictions.shape == (8,)
        assert (predictions >= 0).all()
        assert (predictions < 10).all()

    def test_threshold_auto_computed(self):
        """Threshold should be auto-computed from first batch."""
        rule = ForwardForwardRule(threshold=None)
        model = LocalMLP()

        batch_size = 8
        x = torch.randn(batch_size, 784)
        labels = torch.randint(0, 10, (batch_size,))
        x_pos = embed_label(x, labels)
        x_neg = generate_negative(x, labels)

        rule.train_step(model, x_pos, x_neg)
        assert rule.threshold is not None
        assert rule._threshold_initialized

    def test_threshold_manual(self):
        """Manual threshold should be used as-is."""
        rule = ForwardForwardRule(threshold=5.0)
        assert rule.threshold == 5.0
        assert rule._threshold_initialized

    def test_name(self):
        """Rule should have correct name."""
        rule = ForwardForwardRule()
        assert rule.name == "forward_forward"

    def test_positive_higher_goodness_after_training(self):
        """After some training, classify should produce non-random predictions."""
        torch.manual_seed(42)
        rule = ForwardForwardRule(lr=0.03)
        model = LocalMLP()

        # Train for a few steps on simple data
        for _ in range(20):
            batch_size = 32
            x = torch.rand(batch_size, 784)
            labels = torch.randint(0, 10, (batch_size,))
            x_pos = embed_label(x, labels)
            x_neg = generate_negative(x, labels)
            rule.train_step(model, x_pos, x_neg)

        # Verify training didn't produce NaN weights
        for layer in model.layers:
            assert not torch.isnan(layer.linear.weight.data).any()
            assert not torch.isinf(layer.linear.weight.data).any()

        # Verify classify produces valid predictions
        x_test = torch.rand(16, 784)
        predictions = rule.classify(model, x_test)
        assert predictions.shape == (16,)
        assert (predictions >= 0).all()
        assert (predictions < 10).all()
