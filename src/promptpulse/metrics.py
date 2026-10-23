"""Exact-response variability, entropy, recurrence, and latency metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np


def normalize_response(text: str, policy: str = "conservative") -> str:
    """Normalize text according to specified policy without modifying raw response storage.

    - "conservative": Standardizes newline characters (\r\n -> \n), strips trailing
      whitespace on individual lines, and strips trailing newlines.
    - "raw": Leaves text completely unmodified.
    """
    if not text:
        return ""
    if policy == "conservative":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()
    return text


def compute_exact_metrics(
    raw_responses: List[str],
    latencies_ms: Optional[List[float]] = None,
    failed_attempts_count: int = 0,
    normalization_policy: str = "conservative",
) -> Dict[str, Any]:
    """Compute mathematical variability and consistency metrics for a sequence of responses.

    Args:
        raw_responses: Chronologically ordered list of raw model response texts.
        latencies_ms: Chronologically ordered list of measured latencies in milliseconds.
        failed_attempts_count: Total failed attempts recorded for this prompt.
        normalization_policy: Text normalization policy ('conservative' or 'raw').

    Returns:
        Structured dictionary containing exact variability metrics, entropy, streaks,
        recurrence statistics, and latency summaries.
    """
    n_successful = len(raw_responses)
    latencies = latencies_ms or []

    # Safe handling of empty result set
    if n_successful == 0:
        return {
            "successful_responses": 0,
            "failed_attempts": failed_attempts_count,
            "unique_responses_count": 0,
            "unique_response_ratio": 0.0,
            "duplicate_rate": 0.0,
            "dominant_response_share": 0.0,
            "shannon_entropy": 0.0,
            "normalized_entropy": 0.0,
            "recurrence_observed": False,
            "first_recurrence_run_index": None,
            "unique_responses_before_first_recurrence": 0,
            "recurrence_gaps": [],
            "mean_recurrence_gap": None,
            "longest_consecutive_streak": 0,
            "frequency_distribution": [],
            "latency": {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            },
        }

    # Normalize responses
    normalized = [normalize_response(r, policy=normalization_policy) for r in raw_responses]

    # Frequency calculation
    counts = Counter(normalized)
    k_unique = len(counts)

    # Core variability ratios
    unique_ratio = k_unique / n_successful
    duplicate_rate = 1.0 - unique_ratio

    dominant_count = max(counts.values())
    dominant_share = dominant_count / n_successful

    # Shannon Entropy H = - sum(p_i * log2(p_i))
    shannon_entropy = 0.0
    for count in counts.values():
        p_i = count / n_successful
        if p_i > 0:
            shannon_entropy -= p_i * math.log2(p_i)

    # Normalized Entropy = H / log2(K) when K > 1, else 0.0
    if k_unique > 1:
        max_possible_entropy = math.log2(k_unique)
        normalized_entropy = shannon_entropy / max_possible_entropy
    else:
        normalized_entropy = 0.0

    # Sequential recurrence analysis
    first_recurrence_idx: Optional[int] = None
    recurrence_observed = False
    unique_before_recurrence = 0

    seen: set[str] = set()
    for idx, resp in enumerate(normalized, start=1):
        if resp in seen:
            first_recurrence_idx = idx
            recurrence_observed = True
            unique_before_recurrence = len(seen)
            break
        seen.add(resp)

    if not recurrence_observed:
        unique_before_recurrence = k_unique

    # Recurrence gaps calculation
    # Track positions of each response
    positions_map: Dict[str, List[int]] = {}
    for idx, resp in enumerate(normalized, start=1):
        positions_map.setdefault(resp, []).append(idx)

    all_recurrence_gaps: List[int] = []
    for positions in positions_map.values():
        if len(positions) > 1:
            for j in range(1, len(positions)):
                gap = positions[j] - positions[j - 1]
                all_recurrence_gaps.append(gap)

    mean_recurrence_gap = (
        float(np.mean(all_recurrence_gaps)) if all_recurrence_gaps else None
    )

    # Longest consecutive streak of identical responses
    longest_streak = 1
    current_streak = 1
    for i in range(1, n_successful):
        if normalized[i] == normalized[i - 1]:
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            current_streak = 1

    # Frequency distribution
    freq_distribution = [
        {
            "normalized_response": resp,
            "count": count,
            "share": round(count / n_successful, 4),
        }
        for resp, count in counts.most_common()
    ]

    # Latency statistics
    latency_stats = {
        "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "median_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
        "min_ms": round(float(np.min(latencies)), 2) if latencies else 0.0,
        "max_ms": round(float(np.max(latencies)), 2) if latencies else 0.0,
    }

    return {
        "successful_responses": n_successful,
        "failed_attempts": failed_attempts_count,
        "unique_responses_count": k_unique,
        "unique_response_ratio": round(unique_ratio, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "dominant_response_share": round(dominant_share, 4),
        "shannon_entropy": round(shannon_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "recurrence_observed": recurrence_observed,
        "first_recurrence_run_index": first_recurrence_idx,
        "unique_responses_before_first_recurrence": unique_before_recurrence,
        "recurrence_gaps": all_recurrence_gaps,
        "mean_recurrence_gap": round(mean_recurrence_gap, 2) if mean_recurrence_gap is not None else None,
        "longest_consecutive_streak": longest_streak,
        "frequency_distribution": freq_distribution,
        "latency": latency_stats,
    }
