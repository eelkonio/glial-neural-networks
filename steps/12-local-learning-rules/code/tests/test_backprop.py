"""Tests for backpropagation baseline training."""

import torch
import pytest

from code.network.local_mlp import LocalMLP
from code.experiment.runner import (
    train_backprop,
    evaluate_accuracy,
    set_seed,
    get_device,
)


class TestBackpropBaseline:
    """Tests for backprop training infrastructure."""

    def test_evaluate_accuracy_random_model(self):
        """Random model should have ~10% accuracy (10 classes)."""
        from code.data.fashion_mnist import get_fashion_mnist_loaders

        model = LocalMLP()
        _, test_loader = get_fashion_mnist_loaders(batch_size=128)
        device = torch.device("cpu")
        model = model.to(device)

        acc = evaluate_accuracy(model, test_loader, device, detach=False)
        # Random chance is 10%, allow some variance
        assert 0.05 < acc < 0.20

    def test_train_backprop_short(self):
        """Short training should improve over random."""
        result = train_backprop(
            epochs=2,
            batch_size=256,
            lr=1e-3,
            seed=42,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert "model" in result
        assert "history" in result
        assert "final_accuracy" in result
        assert result["rule_name"] == "backprop"
        # After 2 epochs, should be better than random (10%)
        assert result["final_accuracy"] > 0.3

    def test_set_seed_determinism(self):
        """Same seed should produce same initial weights."""
        set_seed(42)
        model1 = LocalMLP()
        w1 = model1.layers[0].linear.weight.data.clone()

        set_seed(42)
        model2 = LocalMLP()
        w2 = model2.layers[0].linear.weight.data.clone()

        assert torch.allclose(w1, w2)

    def test_model_uses_no_detach(self):
        """Backprop should use detach=False for full gradient flow."""
        model = LocalMLP()
        x = torch.randn(4, 784)

        # With detach=False, we should be able to compute gradients
        out = model(x, detach=False)
        loss = out.sum()
        loss.backward()

        # All layers should have gradients
        for layer in model.layers:
            assert layer.linear.weight.grad is not None
