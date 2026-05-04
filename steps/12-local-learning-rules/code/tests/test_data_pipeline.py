"""Tests for the FashionMNIST data pipeline and ForwardForward adapter."""

import torch
import pytest

from code.data.fashion_mnist import (
    embed_label,
    generate_negative,
    ForwardForwardDataAdapter,
    get_fashion_mnist_loaders,
)


class TestEmbedLabel:
    """Tests for embed_label function."""

    def test_correct_pixel_set(self):
        """Label pixel should be set to 1.0."""
        x = torch.zeros(4, 784)
        labels = torch.tensor([0, 3, 7, 9])
        result = embed_label(x, labels)

        for i, label in enumerate(labels):
            assert result[i, label.item()] == 1.0

    def test_other_label_pixels_zeroed(self):
        """Non-label pixels in the label region should be 0."""
        x = torch.ones(4, 784)  # Start with all ones
        labels = torch.tensor([2, 5, 8, 1])
        result = embed_label(x, labels)

        for i, label in enumerate(labels):
            for j in range(10):
                if j == label.item():
                    assert result[i, j] == 1.0
                else:
                    assert result[i, j] == 0.0

    def test_image_pixels_unchanged(self):
        """Pixels beyond the label region should be unchanged."""
        x = torch.rand(4, 784)
        labels = torch.tensor([0, 1, 2, 3])
        result = embed_label(x, labels)

        assert torch.allclose(result[:, 10:], x[:, 10:])

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.rand(16, 784)
        labels = torch.randint(0, 10, (16,))
        result = embed_label(x, labels)
        assert result.shape == (16, 784)


class TestGenerateNegative:
    """Tests for generate_negative function."""

    def test_wrong_label_embedded(self):
        """Negative samples should have a different label than the correct one."""
        x = torch.zeros(100, 784)
        labels = torch.randint(0, 10, (100,))
        result = generate_negative(x, labels)

        for i, correct_label in enumerate(labels):
            # The embedded label should NOT be the correct one
            embedded_label = result[i, :10].argmax().item()
            assert embedded_label != correct_label.item()

    def test_image_pixels_unchanged(self):
        """Image pixels beyond label region should be unchanged."""
        x = torch.rand(16, 784)
        labels = torch.randint(0, 10, (16,))
        result = generate_negative(x, labels)

        # After embed_label zeros out first 10 pixels, compare from pixel 10 onward
        # generate_negative calls embed_label internally, so first 10 pixels change
        assert torch.allclose(result[:, 10:], x[:, 10:])

    def test_exactly_one_label_pixel_set(self):
        """Exactly one pixel in the label region should be 1.0."""
        x = torch.rand(32, 784)
        labels = torch.randint(0, 10, (32,))
        result = generate_negative(x, labels)

        for i in range(32):
            label_region = result[i, :10]
            assert label_region.sum().item() == pytest.approx(1.0)
            assert label_region.max().item() == 1.0

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.rand(8, 784)
        labels = torch.randint(0, 10, (8,))
        result = generate_negative(x, labels)
        assert result.shape == (8, 784)


class TestForwardForwardDataAdapter:
    """Tests for the ForwardForwardDataAdapter."""

    def test_yields_correct_tuple(self):
        """Adapter should yield (x_pos, x_neg, labels) tuples."""
        train_loader, _ = get_fashion_mnist_loaders(batch_size=32)
        adapter = ForwardForwardDataAdapter(train_loader)

        x_pos, x_neg, labels = next(iter(adapter))
        assert x_pos.shape == (32, 784)
        assert x_neg.shape == (32, 784)
        assert labels.shape == (32,)

    def test_positive_has_correct_label(self):
        """Positive samples should have the correct label embedded."""
        train_loader, _ = get_fashion_mnist_loaders(batch_size=16)
        adapter = ForwardForwardDataAdapter(train_loader)

        x_pos, _, labels = next(iter(adapter))
        for i, label in enumerate(labels):
            assert x_pos[i, label.item()] == 1.0

    def test_negative_has_wrong_label(self):
        """Negative samples should have an incorrect label embedded."""
        train_loader, _ = get_fashion_mnist_loaders(batch_size=64)
        adapter = ForwardForwardDataAdapter(train_loader)

        _, x_neg, labels = next(iter(adapter))
        for i, correct_label in enumerate(labels):
            embedded_label = x_neg[i, :10].argmax().item()
            assert embedded_label != correct_label.item()

    def test_pixel_values_in_range(self):
        """All pixel values should be in [0, 1]."""
        train_loader, _ = get_fashion_mnist_loaders(batch_size=32)
        adapter = ForwardForwardDataAdapter(train_loader)

        x_pos, x_neg, _ = next(iter(adapter))
        assert x_pos.min() >= 0.0
        assert x_pos.max() <= 1.0
        assert x_neg.min() >= 0.0
        assert x_neg.max() <= 1.0
