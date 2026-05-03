"""Property-based test for developmental force direction.

Property 5: Developmental force direction
- For any pair of weight positions and any gradient correlation value:
  - If correlation > 0: force is attractive (moves positions closer)
  - If correlation <= 0: force is repulsive (moves positions apart)

Tests the force computation function in isolation without needing data.

**Validates: Requirements 8.2**
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from code.embeddings.developmental import compute_force


# --- Strategies ---

# 3D position vectors in a reasonable range
position_strategy = arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)

# Correlation values in [-1, 1]
positive_correlation = st.floats(
    min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
)

non_positive_correlation = st.floats(
    min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False
)


# --- Property 5: Developmental force direction ---
# Feature: spatial-embedding-experiments, Property 5: Developmental force direction
# **Validates: Requirements 8.2**


@given(
    pos_i=position_strategy,
    pos_j=position_strategy,
    correlation=positive_correlation,
)
@settings(max_examples=200, deadline=None)
def test_positive_correlation_produces_attractive_force(pos_i, pos_j, correlation):
    """For positive correlation, force is attractive (moves positions closer).

    After applying the force, the distance between the two positions
    should decrease (or stay the same if positions are identical).
    """
    # Ensure positions are not too close (degenerate case)
    distance = np.linalg.norm(pos_j - pos_i)
    assume(distance > 1e-6)

    force_on_i, force_on_j = compute_force(pos_i, pos_j, correlation)

    # Apply forces to get new positions
    new_pos_i = pos_i + force_on_i
    new_pos_j = pos_j + force_on_j

    # New distance should be less than or equal to original distance
    new_distance = np.linalg.norm(new_pos_j - new_pos_i)

    assert new_distance < distance, (
        f"Positive correlation ({correlation}) should produce attractive force. "
        f"Original distance: {distance}, new distance: {new_distance}, "
        f"force_on_i: {force_on_i}, force_on_j: {force_on_j}"
    )


@given(
    pos_i=position_strategy,
    pos_j=position_strategy,
    correlation=non_positive_correlation,
)
@settings(max_examples=200, deadline=None)
def test_non_positive_correlation_produces_repulsive_force(pos_i, pos_j, correlation):
    """For zero or negative correlation, force is repulsive (moves positions apart).

    After applying the force, the distance between the two positions
    should increase (or stay the same if correlation is exactly zero and
    positions are identical).
    """
    # Ensure positions are not too close (degenerate case)
    distance = np.linalg.norm(pos_j - pos_i)
    assume(distance > 1e-6)

    # Skip exact zero correlation (produces zero force magnitude)
    assume(abs(correlation) > 1e-10)

    force_on_i, force_on_j = compute_force(pos_i, pos_j, correlation)

    # Apply forces to get new positions
    new_pos_i = pos_i + force_on_i
    new_pos_j = pos_j + force_on_j

    # New distance should be greater than original distance
    new_distance = np.linalg.norm(new_pos_j - new_pos_i)

    assert new_distance > distance, (
        f"Non-positive correlation ({correlation}) should produce repulsive force. "
        f"Original distance: {distance}, new distance: {new_distance}, "
        f"force_on_i: {force_on_i}, force_on_j: {force_on_j}"
    )
