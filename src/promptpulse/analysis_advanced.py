"""Advanced analysis module for PromptPulse: Semantic Diversity, Response Clustering, Correctness Evaluation, and Latency Correlation."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from promptpulse.metrics import normalize_response


def compute_semantic_clusters(
    responses: List[str],
    distance_threshold: float = 0.35,
) -> Tuple[int, List[int], Dict[int, List[str]]]:
    """Group responses into semantic meaning clusters using TF-IDF and Cosine Agglomerative Clustering.

    Args:
        responses: List of raw response texts.
        distance_threshold: Maximum cosine distance to merge two responses into the same cluster.

    Returns:
        (num_clusters, cluster_labels_per_response, cluster_to_samples_map)
    """
    if not responses:
        return 0, [], {}

    cleaned = [normalize_response(r) for r in responses]
    if len(set(cleaned)) <= 1:
        return 1, [0] * len(responses), {0: [responses[0]]}

    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(cleaned).toarray()
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X).tolist()
    except Exception:
        # Fallback to exact match grouping if vectorization encounters single-token/degenerate cases
        unique_map: Dict[str, int] = {}
        labels = []
        for text in cleaned:
            if text not in unique_map:
                unique_map[text] = len(unique_map)
            labels.append(unique_map[text])

    num_clusters = len(set(labels))
    cluster_samples: Dict[int, List[str]] = {}
    for label, text in zip(labels, responses):
        cluster_samples.setdefault(label, []).append(text)

    return num_clusters, labels, cluster_samples


def evaluate_correctness(
    response_text: str,
    category: str,
    expected_answer: Optional[str],
    evaluation_type: str,
) -> Optional[bool]:
    """Evaluate whether a model response matches the verified ground truth reference answer."""
    if expected_answer is None or not expected_answer.strip():
        return None

    norm_resp = response_text.lower().strip()
    norm_exp = expected_answer.lower().strip()

    # 1. Maths evaluation (GSM8K)
    if category == "maths" or evaluation_type == "exact_numeric":
        # Check explicit markers like #### <number> or The answer is <number>
        num_match = re.search(r"####\s*([-\d.,]+)", norm_resp)
        if num_match:
            candidate = num_match.group(1).replace(",", "").strip()
            if candidate == norm_exp.replace(",", ""):
                return True

        # Extract last isolated numeric token
        all_numbers = re.findall(r"[-+]?\d*\.?\d+", norm_resp)
        if all_numbers:
            last_number = all_numbers[-1].replace(",", "")
            if last_number == norm_exp.replace(",", ""):
                return True

        # Plain substring check as fallback
        clean_exp_num = re.sub(r"[^\d.]", "", norm_exp)
        if clean_exp_num and clean_exp_num in re.findall(r"\b\d+\b", norm_resp):
            return True
        return False

    # 2. Factual evaluation
    elif category == "factual" or evaluation_type == "fact_check":
        # Token overlap of expected answer tokens
        exp_tokens = [tok.strip(",.()\"'") for tok in norm_exp.split() if len(tok.strip(",.()\"'")) > 1]
        if not exp_tokens:
            return norm_exp in norm_resp
        matches = sum(1 for tok in exp_tokens if tok in norm_resp)
        # If at least 70% of factual tokens present, mark correct
        return (matches / len(exp_tokens)) >= 0.70

    # 3. Coding evaluation (MBPP)
    elif category == "coding" or evaluation_type == "code_execution":
        # Extract function name from expected answer, e.g. "def min_cost"
        fn_match = re.search(r"def\s+([a-zA-Z0-9_]+)", norm_exp)
        if fn_match:
            fn_name = fn_match.group(1)
            # Check if function name and return statement exist in output
            has_fn = f"def {fn_name}" in norm_resp or fn_name in norm_resp
            has_return = "return" in norm_resp
            return has_fn and has_return
        return "def " in norm_resp and "return" in norm_resp

    return None


def run_advanced_evaluation_on_db(
    db_path: str | Path,
    experiment_id: str,
) -> pd.DataFrame:
    """Run semantic clustering, correctness check, and latency correlation across all experiment runs."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Fetch prompts with metadata
    prompts_cursor = conn.execute(
        "SELECT id, category, prompt_text, evaluation_type, expected_answer, metadata_json FROM prompts"
    )
    prompts = {row["id"]: dict(row) for row in prompts_cursor.fetchall()}

    # 2. Fetch runs for this experiment
    runs_cursor = conn.execute(
        """
        SELECT prompt_id, run_index, status, raw_response, latency_ms
        FROM runs
        WHERE experiment_id = ? AND status = 'success'
        ORDER BY prompt_id, run_index
        """,
        (experiment_id,),
    )
    all_runs = runs_cursor.fetchall()

    runs_by_prompt: Dict[str, List[sqlite3.Row]] = {}
    for r in all_runs:
        runs_by_prompt.setdefault(r["prompt_id"], []).append(r)

    results: List[Dict[str, Any]] = []

    for prompt_id, run_list in runs_by_prompt.items():
        p_info = prompts.get(prompt_id, {})
        category = p_info.get("category", "unknown")
        expected_answer = p_info.get("expected_answer")
        eval_type = p_info.get("evaluation_type", "")

        responses = [r["raw_response"] for r in run_list if r["raw_response"] is not None]
        latencies = [r["latency_ms"] for r in run_list if r["latency_ms"] is not None]
        n_total = len(responses)

        # Exact unique strings count
        cleaned_responses = [normalize_response(r) for r in responses]
        k_exact = len(set(cleaned_responses))

        # Semantic clusters
        k_sem, labels, clusters = compute_semantic_clusters(responses)

        # Correctness evaluation
        correct_count = 0
        evaluable = expected_answer is not None and expected_answer.strip() != ""
        if evaluable:
            for resp in responses:
                is_correct = evaluate_correctness(resp, category, expected_answer, eval_type)
                if is_correct:
                    correct_count += 1
            correctness_rate = round((correct_count / n_total) * 100, 2)
        else:
            correctness_rate = None

        # Wording-to-Meaning Diversity Ratio (How many ways did the AI phrase each meaning?)
        wording_multiplier = round(k_exact / max(k_sem, 1), 2)

        results.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "total_runs": n_total,
                "exact_unique_count": k_exact,
                "semantic_clusters_count": k_sem,
                "exact_unique_ratio": round(k_exact / n_total, 4),
                "semantic_diversity_ratio": round(k_sem / n_total, 4),
                "wording_variety_multiplier": wording_multiplier,
                "correctness_rate_pct": correctness_rate,
                "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
                "std_latency_ms": round(float(np.std(latencies)), 2) if latencies else 0.0,
            }
        )

    conn.close()
    df = pd.DataFrame(results)
    return df
