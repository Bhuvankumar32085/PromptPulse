"""Reporting and export utilities for PromptPulse benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from promptpulse.storage import Storage


def export_experiment_reports(
    storage: Storage,
    experiment_id: str,
    output_dir: str | Path = "reports",
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Export experiment results, prompt metrics, and raw runs into JSON and CSV files.

    Args:
        storage: Initialized Storage instance.
        experiment_id: Unique experiment ID.
        output_dir: Destination folder.
        formats: Whitelist of formats, e.g. ["json", "csv"].

    Returns:
        Mapping from report name to written file Path.
    """
    out_path = Path(output_dir) / experiment_id
    out_path.mkdir(parents=True, exist_ok=True)

    exp_info = storage.get_experiment(experiment_id)
    if not exp_info:
        raise ValueError(f"Experiment '{experiment_id}' not found in database.")

    metrics_records = storage.get_experiment_metrics(experiment_id)
    all_runs = storage.get_all_experiment_runs(experiment_id)

    target_formats = set(formats or ["json", "csv"])
    exported_files: Dict[str, Path] = {}

    # 1. Summary JSON
    if "json" in target_formats:
        summary_payload = {
            "experiment": exp_info,
            "total_prompts_evaluated": len(metrics_records),
            "total_runs_recorded": len(all_runs),
            "metrics": metrics_records,
        }
        json_file = out_path / "summary.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        exported_files["summary_json"] = json_file

    # 2. Metrics CSV & Runs CSV
    if "csv" in target_formats:
        flat_metrics: List[Dict[str, Any]] = []
        for item in metrics_records:
            m = item["metrics"]
            flat = {
                "prompt_id": item["prompt_id"],
                "category": item["category"],
                "successful_responses": m.get("successful_responses", 0),
                "failed_attempts": m.get("failed_attempts", 0),
                "unique_responses_count": m.get("unique_responses_count", 0),
                "unique_response_ratio": m.get("unique_response_ratio", 0.0),
                "duplicate_rate": m.get("duplicate_rate", 0.0),
                "dominant_response_share": m.get("dominant_response_share", 0.0),
                "shannon_entropy": m.get("shannon_entropy", 0.0),
                "normalized_entropy": m.get("normalized_entropy", 0.0),
                "recurrence_observed": m.get("recurrence_observed", False),
                "first_recurrence_run_index": m.get("first_recurrence_run_index"),
                "longest_consecutive_streak": m.get("longest_consecutive_streak", 0),
                "mean_latency_ms": m.get("latency", {}).get("mean_ms", 0.0),
                "median_latency_ms": m.get("latency", {}).get("median_ms", 0.0),
            }
            flat_metrics.append(flat)

        metrics_csv = out_path / "metrics.csv"
        df_metrics = pd.DataFrame(flat_metrics)
        df_metrics.to_csv(metrics_csv, index=False)
        exported_files["metrics_csv"] = metrics_csv

        runs_csv = out_path / "runs.csv"
        df_runs = pd.DataFrame(all_runs)
        df_runs.to_csv(runs_csv, index=False)
        exported_files["runs_csv"] = runs_csv

        # 3. Category-wise subdirectories
        categories = sorted(list(set(r.get("category", "unknown") for r in all_runs)))
        for cat in categories:
            cat_dir = out_path / "categories" / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

            cat_metrics = [m for m in flat_metrics if m.get("category") == cat]
            if cat_metrics:
                pd.DataFrame(cat_metrics).to_csv(cat_dir / f"{cat}_metrics.csv", index=False)

            cat_runs = [r for r in all_runs if r.get("category") == cat]
            if cat_runs:
                pd.DataFrame(cat_runs).to_csv(cat_dir / f"{cat}_runs.csv", index=False)

            if "json" in target_formats:
                cat_summary = {
                    "experiment_id": experiment_id,
                    "category": cat,
                    "total_prompts": len(cat_metrics),
                    "total_runs": len(cat_runs),
                    "metrics": [m for m in metrics_records if m.get("category") == cat],
                }
                with open(cat_dir / f"{cat}_summary.json", "w", encoding="utf-8") as f:
                    json.dump(cat_summary, f, indent=2)

    return exported_files


def format_cli_summary(metrics_records: List[Dict[str, Any]]) -> str:
    """Format prompt variability metrics into a clean, readable text table."""
    if not metrics_records:
        return "No evaluated metrics found for this experiment."

    header = (
        f"{'Prompt ID':<18} {'Category':<12} {'N':<5} {'K (Uniq)':<9} "
        f"{'Uniq %':<8} {'Dup %':<8} {'Max %':<8} {'Entropy':<8} {'Streak':<7} {'Recur':<6}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for item in metrics_records:
        m = item["metrics"]
        p_id = item["prompt_id"][:16]
        cat = item["category"][:10]
        n_succ = m.get("successful_responses", 0)
        k_uniq = m.get("unique_responses_count", 0)
        u_ratio = f"{m.get('unique_response_ratio', 0.0) * 100:.1f}%"
        d_rate = f"{m.get('duplicate_rate', 0.0) * 100:.1f}%"
        dom_share = f"{m.get('dominant_response_share', 0.0) * 100:.1f}%"
        entropy = f"{m.get('shannon_entropy', 0.0):.2f}"
        streak = str(m.get("longest_consecutive_streak", 0))
        recur = (
            f"Run {m.get('first_recurrence_run_index')}"
            if m.get("recurrence_observed")
            else "None"
        )

        lines.append(
            f"{p_id:<18} {cat:<12} {n_succ:<5} {k_uniq:<9} "
            f"{u_ratio:<8} {d_rate:<8} {dom_share:<8} {entropy:<8} {streak:<7} {recur:<6}"
        )

    lines.append(separator)
    return "\n".join(lines)
