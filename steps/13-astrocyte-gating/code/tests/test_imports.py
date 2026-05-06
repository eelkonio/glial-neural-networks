"""Smoke test: verify imports from Step 12 and Step 01 work."""


def test_step12_imports():
    """Verify we can import key classes from Step 12."""
    from code.step12_imports import (
        ThreeFactorRule,
        ThirdFactorInterface,
        LayerState,
        LocalMLP,
        get_fashion_mnist_loaders,
    )

    # Instantiate to verify they work
    model = LocalMLP()
    assert model.hidden_size == 128
    assert model.n_classes == 10

    rule = ThreeFactorRule(lr=0.01, tau=100.0)
    assert rule.name == "three_factor"


def test_step01_imports():
    """Verify we can import SpectralEmbedding from Step 01."""
    from code.step12_imports import SpectralEmbedding

    se = SpectralEmbedding()
    assert se.name == "spectral"
