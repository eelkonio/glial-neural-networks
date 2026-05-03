"""Property-based tests for embedding strategies.

Tests the correctness properties defined in the design document:
- Property 1: Embedding output contract (shape and range)
- Property 2: Embedding determinism
- Property 3: Linear embedding formula
- Property 4: Layered clustered x-coordinate preserves layer structure

Uses Hypothesis for property-based testing.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from code.embeddings.layered_clustered import LayeredClusteredEmbedding
from code.embeddings.linear import LinearEmbedding
from code.embeddings.random import RandomEmbedding
from code.embeddings.spectral import SpectralEmbedding
from code.model import BaselineMLP


# --- Strategies ---

embedding_strategies = st.sampled_from([
    LinearEmbedding(),
    RandomEmbedding(seed=42),
    RandomEmbedding(seed=123),
    SpectralEmbedding(),
    LayeredClusteredEmbedding(),
])


# --- Property 1: Embedding output contract (shape and range) ---
# Feature: spatial-embedding-experiments, Property 1: Embedding output contract
# **Validates: Requirements 3.1, 3.2, 4.1, 4.2**


@given(strategy=embedding_strategies)
@settings(max_examples=20, deadline=None)
def test_embedding_output_shape_and_range(strategy):
    """For any embedding strategy and any model, embed() produces
    (N_weights, 3) with all values in [0, 1].
    """
    model = BaselineMLP()
    positions = strategy.embed(model)

    n_weights = model.get_weight_count()

    # Shape check
    assert positions.shape == (n_weights, 3), (
        f"Expected shape ({n_weights}, 3), got {positions.shape}"
    )

    # Range check: all values in [0, 1]
    assert np.all(positions >= 0.0), (
        f"Found values below 0: min={positions.min()}"
    )
    assert np.all(positions <= 1.0), (
        f"Found values above 1: max={positions.max()}"
    )


# --- Property 2: Embedding determinism ---
# Feature: spatial-embedding-experiments, Property 2: Embedding determinism
# **Validates: Requirements 4.3, 14.2**


@given(strategy=embedding_strategies)
@settings(max_examples=20, deadline=None)
def test_embedding_determinism(strategy):
    """For fixed inputs, embed() twice produces identical arrays."""
    model = BaselineMLP()

    positions_1 = strategy.embed(model)
    positions_2 = strategy.embed(model)

    np.testing.assert_array_equal(
        positions_1, positions_2,
        err_msg="Embedding is not deterministic: two calls with same inputs differ"
    )


# --- Property 3: Linear embedding formula ---
# Feature: spatial-embedding-experiments, Property 3: Linear embedding formula
# **Validates: Requirements 3.1**


@given(weight_idx=st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_linear_embedding_formula(weight_idx):
    """For any weight, the linear embedding produces coordinates matching
    (layer_idx/total_layers, source/max_neurons, target/max_neurons).
    """
    model = BaselineMLP()
    embedding = LinearEmbedding()
    positions = embedding.embed(model)

    metadata = model.get_weight_metadata()
    layer_info = model.get_layer_info()
    total_layers = len(layer_info)
    max_neurons = max(
        max(in_feat, out_feat) for _, in_feat, out_feat in layer_info
    )

    # Clamp index to valid range
    idx = weight_idx % len(metadata)
    w = metadata[idx]

    expected_x = w.layer_idx / total_layers
    expected_y = w.source_neuron / max_neurons
    expected_z = w.target_neuron / max_neurons

    np.testing.assert_allclose(
        positions[idx],
        [expected_x, expected_y, expected_z],
        rtol=1e-10,
        err_msg=f"Linear embedding formula mismatch at index {idx}"
    )


# --- Property 4: Layered clustered x-coordinate preserves layer structure ---
# Feature: spatial-embedding-experiments, Property 4: Layered clustered x-coordinate preserves layer structure
# **Validates: Requirements 7.1**


@given(weight_idx=st.integers(min_value=0, max_value=268799))
@settings(max_examples=100)
def test_layered_clustered_x_coordinate(weight_idx):
    """For any model and any weight, the x-coordinate from layered-clustered
    equals layer_idx/total_layers. All weights in the same layer have
    identical x-coordinates.
    """
    model = BaselineMLP()
    embedding = LayeredClusteredEmbedding()
    positions = embedding.embed(model)

    metadata = model.get_weight_metadata()
    layer_info = model.get_layer_info()
    total_layers = len(layer_info)

    # Clamp index to valid range
    idx = weight_idx % len(metadata)
    w = metadata[idx]

    expected_x = w.layer_idx / total_layers
    actual_x = positions[idx, 0]

    np.testing.assert_allclose(
        actual_x,
        expected_x,
        rtol=1e-10,
        err_msg=(
            f"Layered-clustered x-coordinate mismatch at index {idx}: "
            f"layer_idx={w.layer_idx}, total_layers={total_layers}, "
            f"expected={expected_x}, got={actual_x}"
        ),
    )
