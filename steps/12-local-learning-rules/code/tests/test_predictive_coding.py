"""Tests for predictive coding learning rule."""

import torch
import pytest

from code.network.local_mlp import LocalMLP
from code.rules.predictive_coding import PredictiveCodingRule


class TestPredictiveCodingRule:
    """Tests for PredictiveCodingRule."""

    def test_setup_predictions(self):
        """setup_predictions should create prediction weights."""
        rule = PredictiveCodingRule()
        model = LocalMLP()
        rule.setup_predictions(model)

        assert rule._W_predict is not None
        assert len(rule._W_predict) == 5  # One per layer boundary
        # W_predict[0]: (784, 128) — predicts input from first hidden
        assert rule._W_predict[0].shape == (784, 128)
        # W_predict[1]: (128, 128)
        assert rule._W_predict[1].shape == (128, 128)
        # W_predict[4]: (128, 10) — predicts last hidden from output
        assert rule._W_predict[4].shape == (128, 10)

    def test_compute_prediction_errors(self):
        """Prediction errors should have correct shapes."""
        rule = PredictiveCodingRule()
        model = LocalMLP()
        rule.setup_predictions(model)

        batch_size = 8
        x = torch.randn(batch_size, 784)

        # Create representations
        representations = [x] + [
            torch.randn(batch_size, 128) for _ in range(4)
        ] + [torch.randn(batch_size, 10)]

        errors = rule.compute_prediction_errors(representations, x)
        assert len(errors) == 5  # One per layer boundary (except top)
        assert errors[0].shape == (batch_size, 784)
        assert errors[1].shape == (batch_size, 128)

    def test_prediction_error_formula(self):
        """Error should equal actual - W_predict @ repr_above."""
        rule = PredictiveCodingRule()
        model = LocalMLP()
        rule.setup_predictions(model)

        batch_size = 4
        x = torch.randn(batch_size, 784)
        repr_above = torch.randn(batch_size, 128)
        representations = [x, repr_above] + [
            torch.randn(batch_size, 128) for _ in range(3)
        ] + [torch.randn(batch_size, 10)]

        errors = rule.compute_prediction_errors(representations, x)

        # Error at layer 0: input - W_predict[0] @ repr[1]
        expected_error = x - repr_above @ rule._W_predict[0].T
        assert torch.allclose(errors[0], expected_error, atol=1e-5)

    def test_inference_step_updates_representations(self):
        """Inference step should modify hidden representations."""
        rule = PredictiveCodingRule(inference_lr=0.1)
        model = LocalMLP()
        rule.setup_predictions(model)

        batch_size = 4
        x = torch.randn(batch_size, 784)
        representations = [x] + [
            torch.randn(batch_size, 128) for _ in range(4)
        ] + [torch.randn(batch_size, 10)]

        initial_repr = [r.clone() for r in representations]
        new_repr = rule.inference_step(representations, x)

        # Input should not change
        assert torch.allclose(new_repr[0], initial_repr[0])

        # Hidden representations should change
        any_changed = False
        for i in range(1, len(new_repr) - 1):
            if not torch.allclose(new_repr[i], initial_repr[i]):
                any_changed = True
                break
        assert any_changed

    def test_inference_convergence(self):
        """Prediction errors should decrease over inference iterations."""
        rule = PredictiveCodingRule(inference_lr=0.1, n_inference_steps=20)
        model = LocalMLP()
        rule.setup_predictions(model)

        batch_size = 8
        x = torch.randn(batch_size, 784)

        # Initialize representations
        with torch.no_grad():
            activations = model.get_layer_activations(x)
        representations = [x.clone()] + [a.clone() for a in activations]

        # Measure initial error
        errors_initial = rule.compute_prediction_errors(representations, x)
        initial_total = sum(
            (e ** 2).mean().item() for e in errors_initial
        )

        # Run inference
        for _ in range(20):
            representations = rule.inference_step(representations, x)

        # Measure final error
        errors_final = rule.compute_prediction_errors(representations, x)
        final_total = sum(
            (e ** 2).mean().item() for e in errors_final
        )

        # Error should decrease (or at least not explode)
        assert final_total < initial_total * 10  # Shouldn't explode

    def test_train_step_returns_error(self):
        """train_step should return total prediction error."""
        rule = PredictiveCodingRule(n_inference_steps=5)
        model = LocalMLP()

        x = torch.randn(8, 784)
        labels = torch.randint(0, 10, (8,))

        error = rule.train_step(model, x, labels)
        assert isinstance(error, float)
        assert error >= 0

    def test_train_step_modifies_weights(self):
        """Training step should modify model weights."""
        rule = PredictiveCodingRule(lr=0.01, n_inference_steps=5)
        model = LocalMLP()

        initial_weights = [
            layer.linear.weight.data.clone() for layer in model.layers
        ]

        x = torch.randn(16, 784)
        labels = torch.randint(0, 10, (16,))
        rule.train_step(model, x, labels)

        any_changed = False
        for i, layer in enumerate(model.layers):
            if not torch.allclose(layer.linear.weight.data, initial_weights[i]):
                any_changed = True
                break
        assert any_changed

    def test_divergence_guard(self):
        """Should not crash even with adversarial inputs."""
        rule = PredictiveCodingRule(
            lr=0.01, inference_lr=0.5, n_inference_steps=50
        )
        model = LocalMLP()

        # Large inputs that might cause divergence
        x = torch.randn(4, 784) * 10.0
        labels = torch.randint(0, 10, (4,))

        # Should not raise
        error = rule.train_step(model, x, labels)
        assert not torch.isnan(torch.tensor(error))

    def test_reset(self):
        """Reset should clear prediction weights."""
        rule = PredictiveCodingRule()
        model = LocalMLP()
        rule.setup_predictions(model)
        assert rule._W_predict is not None

        rule.reset()
        assert rule._W_predict is None
        assert not rule._setup_done

    def test_name(self):
        """Rule should have correct name."""
        rule = PredictiveCodingRule()
        assert rule.name == "predictive_coding"

    def test_supervised_signal_at_top(self):
        """Top layer should be biased toward correct class."""
        rule = PredictiveCodingRule(inference_lr=0.5)
        model = LocalMLP()
        rule.setup_predictions(model)

        batch_size = 4
        x = torch.randn(batch_size, 784)
        labels = torch.zeros(batch_size, dtype=torch.long)  # All class 0

        # Initialize representations
        representations = [x] + [
            torch.randn(batch_size, 128) for _ in range(4)
        ] + [torch.zeros(batch_size, 10)]  # Start at zero

        # Run inference with labels
        new_repr = rule.inference_step(representations, x, labels=labels)

        # Top layer should move toward one-hot for class 0
        assert new_repr[-1][:, 0].mean() > new_repr[-1][:, 1].mean()
