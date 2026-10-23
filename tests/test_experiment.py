"""Integration tests for experiment orchestration, resumption, retry limits, and database persistence."""

from pathlib import Path
import pytest
from promptpulse.config import AppConfig
from promptpulse.dataset import PromptItem
from promptpulse.experiment import ExperimentRunner
from promptpulse.model import MockModelProvider
from promptpulse.storage import Storage


def create_test_runner(
    tmp_path: Path,
    mock_provider: MockModelProvider,
    n_required: int = 10,
    max_retries: int = 3,
) -> tuple[ExperimentRunner, Storage, AppConfig]:
    dataset_file = tmp_path / "test_prompts.jsonl"
    dataset_file.write_text(
        '{"id": "p1", "category": "maths", "prompt": "Solve 5 + 5"}\n'
        '{"id": "p2", "category": "factual", "prompt": "Capital of Japan?"}\n',
        encoding="utf-8",
    )

    db_file = tmp_path / "test.db"
    cfg = AppConfig(
        dataset={"file_path": str(dataset_file), "categories": ["maths", "factual"]},
        experiment={
            "successful_responses_required": n_required,
            "max_retries": max_retries,
            "reuse_completed_runs": True,
        },
        storage={"db_path": str(db_file)},
    )

    storage = Storage(db_file)
    runner = ExperimentRunner(config=cfg, storage=storage, model_provider=mock_provider)
    return runner, storage, cfg


def test_experiment_runs_to_configured_success_count(tmp_path: Path):
    mock = MockModelProvider(mode="identical")
    runner, storage, cfg = create_test_runner(tmp_path, mock, n_required=10)

    exp_id = runner.run()

    exp = storage.get_experiment(exp_id)
    assert exp["status"] == "completed"

    # Check both prompts completed 10 runs
    runs_p1 = storage.get_successful_runs(exp_id, "p1")
    runs_p2 = storage.get_successful_runs(exp_id, "p2")
    assert len(runs_p1) == 10
    assert len(runs_p2) == 10
    assert [r["run_index"] for r in runs_p1] == list(range(1, 11))

    # Raw responses remain immutable
    assert all(r["raw_response"] == "42" for r in runs_p1)

    # Check metrics
    metrics = storage.get_experiment_metrics(exp_id)
    assert len(metrics) == 2
    m1 = next(m["metrics"] for m in metrics if m["prompt_id"] == "p1")
    assert m1["successful_responses"] == 10
    assert m1["dominant_response_share"] == 1.0


def test_failed_calls_do_not_count_as_successes_and_retries_respected(tmp_path: Path):
    # Simulate a provider that fails the first 2 calls with a retryable timeout
    mock = MockModelProvider(mode="identical", fail_first_n_calls=2, fail_error_type="timeout")
    runner, storage, cfg = create_test_runner(tmp_path, mock, n_required=5, max_retries=3)

    exp_id = runner.run(prompt_id_filter="p1")

    # Prompt p1 should still eventually reach 5 successes because retries were permitted
    runs = storage.get_successful_runs(exp_id, "p1")
    assert len(runs) == 5

    # But 2 failure records must be persisted and counted separately
    failed_count = storage.get_failed_attempts_count(exp_id, "p1")
    assert failed_count == 2

    metrics = storage.get_experiment_metrics(exp_id)
    m = metrics[0]["metrics"]
    assert m["successful_responses"] == 5
    assert m["failed_attempts"] == 2


def test_safe_resumption_does_not_repeat_completed_runs(tmp_path: Path):
    mock = MockModelProvider(mode="unique")
    runner, storage, cfg = create_test_runner(tmp_path, mock, n_required=10)

    exp_id = "test_resume_exp"

    # Save initial experiment and insert 4 existing successful runs for p1
    storage.save_experiment(exp_id, "ResumeTest", cfg.to_snapshot_dict())
    storage.save_prompt(PromptItem(id="p1", category="maths", prompt="Solve 5 + 5"))
    for idx in range(1, 5):
        storage.record_run(
            experiment_id=exp_id,
            prompt_id="p1",
            run_index=idx,
            status="success",
            raw_response=f"Pre-existing run {idx}",
            latency_ms=10.0,
        )

    assert len(storage.get_completed_run_indices(exp_id, "p1")) == 4

    # Run experiment runner with resumption enabled
    runner.run(experiment_id=exp_id, prompt_id_filter="p1")

    # Completed runs must total exactly 10, without repeating 1..4
    all_runs = storage.get_successful_runs(exp_id, "p1")
    assert len(all_runs) == 10
    assert [r["run_index"] for r in all_runs] == list(range(1, 11))
    assert all_runs[0]["raw_response"] == "Pre-existing run 1"
    assert all_runs[3]["raw_response"] == "Pre-existing run 4"
    assert "Unique answer" in all_runs[4]["raw_response"]


def test_reanalysis_produces_identical_metrics(tmp_path: Path):
    mock = MockModelProvider(custom_responses=["A", "B", "A", "C"])
    runner, storage, cfg = create_test_runner(tmp_path, mock, n_required=8)

    exp_id = runner.run(prompt_id_filter="p1")
    m_initial = storage.get_experiment_metrics(exp_id)[0]["metrics"]

    # Re-run metrics computation from raw stored responses
    m_recalculated = runner._recalculate_and_save_metrics(exp_id, "p1")

    assert m_initial["shannon_entropy"] == m_recalculated["shannon_entropy"]
    assert m_initial["unique_response_ratio"] == m_recalculated["unique_response_ratio"]
    assert m_initial["longest_consecutive_streak"] == m_recalculated["longest_consecutive_streak"]
