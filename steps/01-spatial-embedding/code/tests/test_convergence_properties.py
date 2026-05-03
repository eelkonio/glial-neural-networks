"""Property-based tests for convergence detection and temporal degradation.

Properties 10, 13, 14, and 15 from the design document.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from code.experiment.convergence import detect_convergence
from code.spatial.temporal_tracking import TemporalQualityTracker


# --- Property 10: Convergence detection ---
# converged=True iff max relative change in final 20% < 5%
# Feature: spatial-embedding-experiments, Property 10: Convergence detection
# **Validates: Requirements 13.3**


@settings(max_examples=100)
@given(
    trajectory=st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    )
)
def test_convergence_detection_property(trajectory: list[float]):
    """Property 10: Convergence detection.

    converged=True iff max relative change in final 20% < 5%.

    **Validates: Requirements 13.3**
    """
    converged, step_index = detect_convergence(trajectory, threshold=0.05)

    # Compute expected result manually
    final_start = int(len(trajectory) * 0.8)
    final_segment = trajectory[final_start:]

    if len(final_segment) < 2:
        # Cannot determine convergence with < 2 points in final segment
        assert not converged
        return

    max_rel_change = 0.0
    for i in range(1, len(final_segment)):
        prev = final_segment[i - 1]
        curr = final_segment[i]

        if abs(prev) < 1e-12:
            rel_change = abs(curr - prev)
        else:
            rel_change = abs(curr - prev) / abs(prev)

        max_rel_change = max(max_rel_change, rel_change)

    expected_converged = max_rel_change < 0.05

    assert converged == expected_converged, (
        f"Expected converged={expected_converged} but got {converged}. "
        f"max_rel_change={max_rel_change:.6f}, threshold=0.05"
    )

    # If converged, step_index should be the start of the final 20%
    if converged:
        assert step_index == final_start


@settings(max_examples=50)
@given(
    base_value=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    noise_scale=st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False),
    n_points=st.integers(min_value=10, max_value=50),
)
def test_convergence_stable_trajectory_converges(
    base_value: float, noise_scale: float, n_points: int
):
    """A nearly constant trajectory should always converge.

    **Validates: Requirements 13.3**
    """
    # Create a trajectory that's nearly constant (tiny noise)
    rng = np.random.default_rng(42)
    trajectory = [base_value + rng.uniform(-noise_scale, noise_scale) for _ in range(n_points)]

    converged, _ = detect_convergence(trajectory, threshold=0.05)

    # With noise_scale <= 0.01 and base_value >= 0.1,
    # max relative change <= 0.02/0.1 = 0.2, but with very small noise
    # it should typically converge. We check the specific case.
    # Actually, max change is 2*noise_scale, relative is 2*noise_scale/base_value
    max_possible_rel_change = 2 * noise_scale / base_value
    if max_possible_rel_change < 0.05:
        assert converged, (
            f"Expected convergence with max_possible_rel_change={max_possible_rel_change:.6f}"
        )


# --- Property 15: Temporal quality degradation detection ---
# degraded=True iff min < 50% of initial
# Feature: spatial-embedding-experiments, Property 15: Temporal quality degradation detection
# **Validates: Requirements 17.3**


@settings(max_examples=100)
@given(
    initial_score=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    scores=st.lists(
        st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    ),
)
def test_temporal_degradation_detection_property(
    initial_score: float, scores: list[float]
):
    """Property 15: Temporal quality degradation detection.

    degraded=True iff min score < 50% of initial score.

    **Validates: Requirements 17.3**
    """
    # Build a tracker with the given trajectory
    tracker = TemporalQualityTracker(record_interval_epochs=1)

    # Manually populate the history
    tracker._history = [(0, 0, initial_score)]
    for i, score in enumerate(scores):
        tracker._history.append((i + 1, (i + 1) * 100, score))

    degraded = tracker.detect_degradation(threshold=0.5)

    # Expected: degraded if min of ALL scores (including initial) < 50% of initial
    all_scores = [initial_score] + scores
    min_score = min(all_scores)

    # For positive initial scores: degraded if min < initial * 0.5
    expected_degraded = min_score < initial_score * 0.5

    assert degraded == expected_degraded, (
        f"Expected degraded={expected_degraded} but got {degraded}. "
        f"initial={initial_score:.4f}, min={min_score:.4f}, "
        f"threshold_value={initial_score * 0.5:.4f}"
    )


@settings(max_examples=50)
@given(
    initial_score=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_temporal_no_degradation_when_scores_stay_high(initial_score: float):
    """If all scores stay above 50% of initial, degraded should be False.

    **Validates: Requirements 17.3**
    """
    tracker = TemporalQualityTracker(record_interval_epochs=1)

    # All scores are at least 60% of initial (above 50% threshold)
    tracker._history = [
        (0, 0, initial_score),
        (1, 100, initial_score * 0.9),
        (2, 200, initial_score * 0.8),
        (3, 300, initial_score * 0.7),
        (4, 400, initial_score * 0.6),
    ]

    degraded = tracker.detect_degradation(threshold=0.5)
    assert not degraded, "Should not detect degradation when min is 60% of initial"


@settings(max_examples=50)
@given(
    initial_score=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_temporal_degradation_when_score_drops_below_half(initial_score: float):
    """If any score drops below 50% of initial, degraded should be True.

    **Validates: Requirements 17.3**
    """
    tracker = TemporalQualityTracker(record_interval_epochs=1)

    # One score drops to 40% of initial (below 50% threshold)
    tracker._history = [
        (0, 0, initial_score),
        (1, 100, initial_score * 0.9),
        (2, 200, initial_score * 0.4),  # Below 50%
        (3, 300, initial_score * 0.8),
    ]

    degraded = tracker.detect_degradation(threshold=0.5)
    assert degraded, "Should detect degradation when min is 40% of initial"
