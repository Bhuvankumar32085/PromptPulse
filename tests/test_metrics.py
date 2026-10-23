"""Unit tests verifying exact variability metrics against hand-calculated mathematical fixtures."""

import math
import pytest
from promptpulse.metrics import compute_exact_metrics, normalize_response


def test_all_identical_responses():
    """Fixture where every single output is identical."""
    responses = ["Paris"] * 10
    latencies = [10.0, 12.0, 14.0, 11.0, 13.0, 10.0, 15.0, 12.0, 11.0, 12.0]
    m = compute_exact_metrics(responses, latencies_ms=latencies, failed_attempts_count=0)

    assert m["successful_responses"] == 10
    assert m["failed_attempts"] == 0
    assert m["unique_responses_count"] == 1
    assert m["unique_response_ratio"] == 0.1
    assert m["duplicate_rate"] == 0.9
    assert m["dominant_response_share"] == 1.0
    assert m["shannon_entropy"] == 0.0
    assert m["normalized_entropy"] == 0.0
    assert m["recurrence_observed"] is True
    assert m["first_recurrence_run_index"] == 2
    assert m["unique_responses_before_first_recurrence"] == 1
    assert m["longest_consecutive_streak"] == 10
    assert len(m["recurrence_gaps"]) == 9
    assert all(g == 1 for g in m["recurrence_gaps"])
    assert m["latency"]["min_ms"] == 10.0
    assert m["latency"]["max_ms"] == 15.0


def test_all_unique_responses():
    """Fixture where all responses are distinct (maximum entropy)."""
    responses = [f"Unique Answer {i}" for i in range(1, 9)]
    m = compute_exact_metrics(responses)

    assert m["successful_responses"] == 8
    assert m["unique_responses_count"] == 8
    assert m["unique_response_ratio"] == 1.0
    assert m["duplicate_rate"] == 0.0
    assert m["dominant_response_share"] == 0.125
    # H = log2(8) = 3.0
    assert pytest.approx(m["shannon_entropy"], 0.001) == 3.0
    assert pytest.approx(m["normalized_entropy"], 0.001) == 1.0
    assert m["recurrence_observed"] is False
    assert m["first_recurrence_run_index"] is None
    assert m["unique_responses_before_first_recurrence"] == 8
    assert m["longest_consecutive_streak"] == 1
    assert m["recurrence_gaps"] == []
    assert m["mean_recurrence_gap"] is None


def test_mixed_frequency_analytical_fixture():
    """Analytically calculated fixture:

    Sequence: ['A', 'B', 'A', 'C', 'A', 'B', 'A', 'D']
    N = 8
    Counts: A: 4 (p=0.5), B: 2 (p=0.25), C: 1 (p=0.125), D: 1 (p=0.125)
    H = -(0.5*(-1) + 0.25*(-2) + 0.125*(-3) + 0.125*(-3)) = 1.75
    K = 4 -> log2(4) = 2.0 -> Normalized H = 1.75 / 2.0 = 0.875
    """
    responses = ["A", "B", "A", "C", "A", "B", "A", "D"]
    m = compute_exact_metrics(responses)

    assert m["successful_responses"] == 8
    assert m["unique_responses_count"] == 4
    assert pytest.approx(m["unique_response_ratio"], 0.001) == 0.5
    assert pytest.approx(m["duplicate_rate"], 0.001) == 0.5
    assert pytest.approx(m["dominant_response_share"], 0.001) == 0.5
    assert pytest.approx(m["shannon_entropy"], 0.001) == 1.75
    assert pytest.approx(m["normalized_entropy"], 0.001) == 0.875

    # First recurrence: 'A' repeats at step 3
    assert m["recurrence_observed"] is True
    assert m["first_recurrence_run_index"] == 3
    assert m["unique_responses_before_first_recurrence"] == 2

    # Gaps for A: steps 1, 3, 5, 7 -> gaps 2, 2, 2
    # Gaps for B: steps 2, 6 -> gap 4
    assert sorted(m["recurrence_gaps"]) == [2, 2, 2, 4]
    assert pytest.approx(m["mean_recurrence_gap"], 0.001) == 2.5
    assert m["longest_consecutive_streak"] == 1


def test_consecutive_streak():
    responses = ["A", "B", "B", "B", "B", "C", "C"]
    m = compute_exact_metrics(responses)
    assert m["longest_consecutive_streak"] == 4


def test_conservative_normalization():
    raw_a = "Hello World  \r\n"
    raw_b = "Hello World\n"
    assert normalize_response(raw_a, "conservative") == "Hello World"
    assert normalize_response(raw_b, "conservative") == "Hello World"
    assert normalize_response(raw_a, "conservative") == normalize_response(raw_b, "conservative")


def test_empty_metrics():
    m = compute_exact_metrics([], failed_attempts_count=2)
    assert m["successful_responses"] == 0
    assert m["failed_attempts"] == 2
    assert m["shannon_entropy"] == 0.0
    assert m["recurrence_observed"] is False
    assert m["first_recurrence_run_index"] is None
