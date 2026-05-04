"""Tests for LocalLayer and LocalMLP network architecture."""

import torch
import pytest

from code.network.local_layer import LocalLayer
from code.network.local_mlp import LocalMLP
from code.rules.base import LayerState


class TestLocalLayer:
    """Tests for LocalLayer."""

    def test_forward_shape(self):
        """Output shape should be (batch, out_features)."""
        layer = LocalLayer(784, 128)
        x = torch.randn(16, 784)
        out = layer(x)
        assert out.shape == (16, 128)

    def test_detach_mode(self):
        """In detach mode, output should not require grad."""
        layer = LocalLayer(128, 128)
        x = torch.randn(8, 128, requires_grad=True)
        out = layer(x, detach=True)
        assert not out.requires_grad

    def test_no_detach_mode(self):
        """Without detach, output should maintain grad connection."""
        layer = LocalLayer(128, 128)
        x = torch.randn(8, 128, requires_grad=True)
        out = layer(x, detach=False)
        assert out.requires_grad

    def test_stores_activations(self):
        """Layer should store pre and post activations."""
        layer = LocalLayer(64, 32)
        x = torch.randn(4, 64)
        layer(x)
        assert layer.last_pre is not None
        assert layer.last_post is not None
        assert layer.last_pre.shape == (4, 64)
        assert layer.last_post.shape == (4, 32)

    def test_relu_activation(self):
        """Hidden layers should apply ReLU."""
        layer = LocalLayer(10, 10, use_activation=True)
        x = torch.randn(4, 10)
        out = layer(x, detach=False)
        # All outputs should be >= 0 (ReLU)
        assert (out >= 0).all()

    def test_no_activation_output_layer(self):
        """Output layer should not apply activation."""
        layer = LocalLayer(128, 10, use_activation=False)
        # Use input that would produce negative outputs
        torch.manual_seed(42)
        layer.linear.weight.data = torch.randn(10, 128)
        layer.linear.bias.data = torch.zeros(10)
        x = torch.randn(4, 128)
        out = layer(x, detach=False)
        # Should have some negative values (no ReLU)
        assert (out < 0).any()


class TestLocalMLP:
    """Tests for LocalMLP."""

    def test_forward_shape(self):
        """Forward pass should produce (batch, 10) logits."""
        model = LocalMLP()
        x = torch.randn(16, 784)
        out = model(x)
        assert out.shape == (16, 10)

    def test_forward_with_image_shape(self):
        """Should handle (batch, 1, 28, 28) input."""
        model = LocalMLP()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_forward_with_states_count(self):
        """forward_with_states should return 5 LayerState objects."""
        model = LocalMLP()
        x = torch.randn(4, 784)
        states = model.forward_with_states(x)
        assert len(states) == 5

    def test_forward_with_states_shapes(self):
        """Each LayerState should have correct pre/post shapes."""
        model = LocalMLP()
        x = torch.randn(8, 784)
        states = model.forward_with_states(x)

        expected_shapes = [
            ((8, 784), (8, 128)),   # Layer 0: 784 -> 128
            ((8, 128), (8, 128)),   # Layer 1: 128 -> 128
            ((8, 128), (8, 128)),   # Layer 2: 128 -> 128
            ((8, 128), (8, 128)),   # Layer 3: 128 -> 128
            ((8, 128), (8, 10)),    # Layer 4: 128 -> 10
        ]

        for i, (state, (pre_shape, post_shape)) in enumerate(zip(states, expected_shapes)):
            assert state.pre_activation.shape == pre_shape, f"Layer {i} pre shape mismatch"
            assert state.post_activation.shape == post_shape, f"Layer {i} post shape mismatch"

    def test_forward_with_states_weight_shapes(self):
        """Weight matrices should have correct shapes."""
        model = LocalMLP()
        x = torch.randn(4, 784)
        states = model.forward_with_states(x)

        expected_weight_shapes = [
            (128, 784),  # Layer 0
            (128, 128),  # Layer 1
            (128, 128),  # Layer 2
            (128, 128),  # Layer 3
            (10, 128),   # Layer 4
        ]

        for i, (state, shape) in enumerate(zip(states, expected_weight_shapes)):
            assert state.weights.shape == shape, f"Layer {i} weight shape mismatch"

    def test_gradient_locality(self):
        """With detach=True, gradients should not flow to earlier layers.

        We test this by checking that the output of each intermediate layer
        is detached (no grad_fn), which means backward from any downstream
        computation cannot reach earlier layers.
        """
        model = LocalMLP()
        x = torch.randn(4, 784)

        # Forward with detach — each layer's output is detached
        current = x.view(x.size(0), -1)
        for i, layer in enumerate(model.layers):
            current = layer(current, detach=True)
            if i < len(model.layers) - 1:
                # Intermediate outputs should be detached (no grad_fn)
                assert not current.requires_grad, (
                    f"Layer {i} output should be detached"
                )
                assert current.grad_fn is None, (
                    f"Layer {i} output should have no grad_fn"
                )

    def test_no_detach_allows_gradients(self):
        """With detach=False, gradients should flow to all layers."""
        model = LocalMLP()
        x = torch.randn(4, 784)

        out = model(x, detach=False)
        loss = out.sum()
        loss.backward()

        # All layers should have non-zero gradients
        for i, layer in enumerate(model.layers):
            assert layer.linear.weight.grad is not None, f"Layer {i} missing gradient"
            assert (layer.linear.weight.grad != 0).any(), f"Layer {i} has zero gradient"

    def test_get_layer_activations(self):
        """get_layer_activations should return 5 tensors."""
        model = LocalMLP()
        x = torch.randn(4, 784)
        activations = model.get_layer_activations(x)

        assert len(activations) == 5
        assert activations[0].shape == (4, 128)
        assert activations[-1].shape == (4, 10)

    def test_layer_indices(self):
        """LayerState should have correct layer_index values."""
        model = LocalMLP()
        x = torch.randn(4, 784)
        states = model.forward_with_states(x)

        for i, state in enumerate(states):
            assert state.layer_index == i
