"""End-to-end tests for PromptPulse CLI commands using Typer's CliRunner."""

from pathlib import Path
from typer.testing import CliRunner
from promptpulse.cli import app

runner = CliRunner()


def test_cli_validate_success():
    result = runner.invoke(app, ["validate", "--config", "config.yaml"])
    assert result.exit_code == 0
    assert "All validation checks PASSED" in result.stdout


def test_cli_validate_missing_file():
    result = runner.invoke(app, ["validate", "--config", "nonexistent_config.yaml"])
    assert result.exit_code != 0


def test_cli_run_dry_run_and_export(tmp_path: Path):
    cfg_file = tmp_path / "cli_test_config.yaml"
    db_file = tmp_path / "cli_test.db"
    report_dir = tmp_path / "cli_reports"

    db_posix = db_file.as_posix()
    report_posix = report_dir.as_posix()
    cfg_content = f"""
project:
  name: "CliTest"
  random_seed: 42
model:
  loading_source: "mock"
  model_id: "mock"
experiment:
  successful_responses_required: 5
  max_retries: 2
dataset:
  file_path: "datasets/prompts.example.jsonl"
  categories: ["maths"]
  max_prompts_per_category: 1
storage:
  db_path: "{db_posix}"
reporting:
  report_dir: "{report_posix}"
  output_formats: ["json", "csv"]
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")

    # Run with dry run
    run_res = runner.invoke(app, ["run", "--config", str(cfg_file), "--dry-run"])
    assert run_res.exit_code == 0
    assert "completed!" in run_res.stdout
    assert "math_gsm8k_001" in run_res.stdout

    # Analyze
    analyze_res = runner.invoke(app, ["analyze", "--config", str(cfg_file)])
    assert analyze_res.exit_code == 0
    assert "math_gsm8k_001" in analyze_res.stdout

    # Export
    export_res = runner.invoke(app, ["export", "--config", str(cfg_file), "--output-dir", str(report_dir)])
    assert export_res.exit_code == 0
    assert "Successfully exported" in export_res.stdout

    # Verify category-wise directories exist
    categories_dir = report_dir / run_res.stdout.split("Experiment '")[1].split("'")[0] / "categories" / "maths"
    assert categories_dir.exists()

    # Plot
    plot_res = runner.invoke(app, ["plot", "--config", str(cfg_file), "--no-sync-assets"])
    assert plot_res.exit_code == 0
    assert "Generated 5 publication-quality plot(s)" in plot_res.stdout


def test_cli_direct_custom_prompt(tmp_path: Path):
    cfg_file = tmp_path / "custom_test_config.yaml"
    db_file = tmp_path / "custom_test.db"
    report_dir = tmp_path / "custom_reports"

    cfg_content = f"""
project:
  name: "CustomPromptTest"
model:
  loading_source: "mock"
storage:
  db_path: "{db_file.as_posix()}"
reporting:
  report_dir: "{report_dir.as_posix()}"
experiment:
  successful_responses_required: 5
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")

    run_res = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(cfg_file),
            "--prompt",
            "What is 20 + 20?",
            "--category",
            "maths",
            "--dry-run",
        ],
    )
    assert run_res.exit_code == 0
    assert "completed!" in run_res.stdout
    assert "maths" in run_res.stdout


def test_cli_add_prompt_command(tmp_path: Path):
    dataset_file = tmp_path / "test_add.jsonl"
    cfg_file = tmp_path / "add_config.yaml"
    cfg_file.write_text(f'dataset:\n  file_path: "{dataset_file.as_posix()}"\n', encoding="utf-8")

    res = runner.invoke(
        app,
        [
            "add-prompt",
            "--config",
            str(cfg_file),
            "--prompt",
            "Solve for x: 2x = 10",
            "--category",
            "maths",
            "--expected-answer",
            "5",
        ],
    )
    assert res.exit_code == 0
    assert "[OK] Custom prompt added" in res.stdout
    assert dataset_file.exists()
    content = dataset_file.read_text(encoding="utf-8")
    assert "Solve for x: 2x = 10" in content
    assert "5" in content

