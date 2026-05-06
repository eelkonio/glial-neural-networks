"""Property tests for stability fix (Properties 1, 2).

Property 1: Sign-Preserving Error Clipping
Property 2: Eligibility Trace Norm Bounding

Validates: Requirements 2.1, 2.2, 2.4
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

from code.stability import clip_error_signal, normalize_eligibility


# --- Custom strategies ---

def error_tensors(min_size=1, max_size=50):
    """Generate error tensors with values spanning [-1e6, 1e6]."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.builds(
            lambda arr: torch.tensor(arr, dtype=torch.float32),
            arrays(
                dtype=np.float32,
                shape=(n,),
                elements=st.floats(
                    min_value=-1e6, max_value=1e6,
                    allow_nan=False, allow_infinity=False,
                ),
            ),
        )
    )


def matrix_tensors(min_rows=1, max_rows=20, min_cols=1, max_cols=20):
    """Generate matrix tensors for eligibility traces."""
    return st.tuples(
        st.integers(min_value=min_rows, max_value=max_rows),
        st.integers(min_value=min_cols, max_value=max_cols),
    ).flatmap(
        lambda shape: st.builds(
            lambda arr: torch.tensor(arr, dtype=torch.float32),
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(
                    min_value=-1e4, max_value=1e4,
                    allow_nan=False, allow_infinity=False,
                ),
            ),
        )
    )


# --- Property 1: Sign-Preserving Error Clipping ---

@pytest.mark.property
class TestSignPreservingClipping:
    """Property 1: Sign-Preserving Error Clipping.

    **Validates: Requirements 2.1, 2.4**

    For any error signal tensor with arbitrary values, clipping to a
    threshold T shall produce an output where:
    (a) no element has absolute value exceeding T
    (b) the sign of every non-zero element is preserved from the original
    """

    @given(
        error=error_tensors(),
        threshold=st.floats(min_value=0.01, max_value=100.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_no_element_exceeds_threshold(self, error, threshold):
        """(a) No element has absolute value exceeding threshold."""
        clipped = clip_error_signal(error, threshold=threshold)
        assert (clipped.abs() <= threshold + 1e-6).all()

    @given(error=error_tensors())
    @settings(max_examples=200, deadline=None)
    def test_sign_preserved(self, error):
        """(b) Sign of every non-zero element is preserved."""
        threshold = 10.0
        clipped = clip_error_signal(error, threshold=threshold)

        # For non-zero elements, sign must be preserved
        nonzero_mask = error != 0
        if nonzero_mask.any():
            original_signs = torch.sign(error[nonzero_mask])
            clipped_signs = torch.sign(clipped[nonzero_mask])
            assert (original_signs == clipped_signs).all()


# --- Property 2: Eligibility Trace Norm Bounding ---

@pytest.mark.property
class TestEligibilityNormBounding:
    """Property 2: Eligibility Trace Norm Bounding.

    **Validates: Requirements 2.2**

    For any eligibility trace tensor whose Frobenius norm exceeds a
    threshold N, normalization shall produce a tensor with norm equal
    to the safe constant S, and the direction (unit vector) of the
    trace shall be preserved.
    """

    @given(trace=matrix_tensors())
    @settings(max_examples=200, deadline=None)
    def test_norm_bounded_to_safe_constant(self, trace):
        """When norm exceeds threshold, output norm equals safe_constant."""
        norm_threshold = 100.0
        safe_constant = 1.0

        result = normalize_eligibility(
            trace, norm_threshold=norm_threshold, safe_constant=safe_constant
        )

        if trace.norm() > norm_threshold:
            assert abs(result.norm().item() - safe_constant) < 1e-5
        else:
            # Unchanged when below threshold
            assert torch.allclose(result, trace)

    @given(trace=matrix_tensors())
    @settings(max_examples=200, deadline=None)
    def test_direction_preserved(self, trace):
        """Direction (unit vector) is preserved after normalization."""
        norm_threshold = 100.0
        safe_constant = 1.0

        original_norm = trace.norm()
        assume(original_norm > norm_threshold)  # Only test when normalization applies
        assume(original_norm > 1e-8)  # Avoid division by zero

        result = normalize_eligibility(
            trace, norm_threshold=norm_threshold, safe_constant=safe_constant
        )

        # Direction = unit vector should be the same
        original_direction = trace / original_norm
        result_direction = result / result.norm()

        assert torch.allclose(original_direction, result_direction, atol=1e-5)
