"""Command-line interface (CLI) for PromptPulse."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
import typer

from promptpulse.config import load_config
from promptpulse.dataset import load_dataset
from promptpulse.experiment import ExperimentRunner
from promptpulse.model import MockModelProvider, get_model_provider
from promptpulse.reporting import export_experiment_reports, format_cli_summary
from promptpulse.storage import Storage

app = typer.Typer(
    name="promptpulse",
    help="PromptPulse: LLM Response Variability & Consistency Tracker CLI",
    add_completion=False,
)


def setup_logging(level_name: str) -> None:
    """Configure console logging level and format."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def validate(
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Validate configuration file, benchmark dataset, and storage setup."""
    typer.echo(f"Validating configuration from: {config_path}")
    try:
        config = load_config(config_path)
        setup_logging(config.project.logging_level)
        typer.secho("[OK] Configuration syntax and schema are valid.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"[ERROR] Configuration error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Validating dataset from: {config.dataset.file_path}")
    try:
        prompts = load_dataset(
            file_path=config.dataset.file_path,
            categories=config.dataset.categories,
            max_prompts_per_category=config.dataset.max_prompts_per_category,
        )
        typer.secho(
            f"[OK] Dataset is valid. Found {len(prompts)} prompt(s) across categories: "
            f"{sorted(list(set(p.category for p in prompts)))}",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        typer.secho(f"[ERROR] Dataset error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Checking database accessibility: {config.storage.db_path}")
    try:
        storage = Storage(config.storage.db_path)
        typer.secho("[OK] SQLite database initialized successfully.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"[ERROR] Database initialization error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("\nAll validation checks PASSED successfully!", fg=typer.colors.GREEN, bold=True)


@app.command("download-dataset")
def download_dataset_cmd(
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
    prompts_per_category: Optional[int] = typer.Option(
        None, "--per-category", "-n", help="Number of prompts per category (overrides config)"
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Custom destination JSONL file"
    ),
) -> None:
    """Download real benchmark prompts (GSM8K, MBPP, Factual, Creative, Ambiguous)."""
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    n_per_cat = prompts_per_category or config.dataset.prompts_per_category
    dest_path = output_path or Path(config.dataset.file_path)

    typer.echo(f"Downloading real benchmark datasets ({n_per_cat} prompts per category across {config.dataset.categories})...")
    from promptpulse.downloader import download_and_build_dataset
    try:
        saved_file = download_and_build_dataset(
            output_path=dest_path,
            prompts_per_category=n_per_cat,
            categories=config.dataset.categories,
        )
        typer.secho(f"[OK] Real benchmark dataset saved at: {saved_file.resolve()}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"[ERROR] Failed to download dataset: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("add-prompt")
def add_prompt_cmd(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Custom prompt text"),
    category: str = typer.Option("maths", "--category", "-cat", help="Category: maths, coding, factual, creative, ambiguous"),
    expected_answer: Optional[str] = typer.Option(None, "--expected-answer", "-a", help="Optional reference answer"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
) -> None:
    """Add a user's own custom prompt to the benchmark dataset."""
    import json
    import uuid
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    clean_cat = category.strip().lower()
    from promptpulse.config import VALID_CATEGORIES
    if clean_cat not in VALID_CATEGORIES:
        typer.secho(f"Invalid category '{category}'. Allowed: {sorted(list(VALID_CATEGORIES))}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    item = {
        "id": f"custom_{clean_cat}_{uuid.uuid4().hex[:6]}",
        "category": clean_cat,
        "prompt": prompt.strip(),
        "evaluation_type": "user_custom",
        "expected_answer": expected_answer,
        "reference_source": "User-supplied custom prompt",
        "metadata": {"user_defined": True},
    }

    target_file = Path(config.dataset.file_path)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")

    typer.secho(f"[OK] Custom prompt added with ID: '{item['id']}' to {target_file.resolve()}", fg=typer.colors.GREEN)


@app.command()
def run(
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment-id", "-e", help="Specific or existing experiment ID to resume"
    ),
    prompt_id: Optional[str] = typer.Option(
        None, "--prompt-id", "-p", help="Filter to run only a single prompt ID from dataset"
    ),
    custom_prompt: Optional[str] = typer.Option(
        None, "--prompt", help="Direct custom prompt text to test on the fly"
    ),
    category: str = typer.Option(
        "maths", "--category", help="Category for custom prompt (maths, coding, factual, creative, ambiguous)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run with mock provider to test pipeline without loading HF model"
    ),
) -> None:
    """Execute response variability benchmark runs across prompts."""
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    setup_logging(config.project.logging_level)

    # If dataset file does not exist and no custom prompt given, auto-download
    if not custom_prompt and not Path(config.dataset.file_path).is_file():
        typer.echo(f"Dataset file '{config.dataset.file_path}' not found. Downloading real benchmark prompts...")
        from promptpulse.downloader import download_and_build_dataset
        download_and_build_dataset(
            output_path=config.dataset.file_path,
            prompts_per_category=config.dataset.prompts_per_category,
            categories=config.dataset.categories,
        )

    storage = Storage(config.storage.db_path)

    if dry_run or config.model.loading_source == "mock":
        typer.secho("Notice: Running in DRY-RUN mode using MockModelProvider.", fg=typer.colors.YELLOW)
        provider = MockModelProvider(mode="pattern", custom_responses=["Ans A", "Ans A", "Ans B", "Ans C"])
    else:
        try:
            provider = get_model_provider(config.model)
        except Exception as e:
            typer.secho(f"Model initialization error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    runner = ExperimentRunner(config=config, storage=storage, model_provider=provider)

    # Handle direct on-the-fly custom prompt
    direct_items = None
    if custom_prompt:
        import uuid
        from promptpulse.dataset import PromptItem
        custom_id = f"custom_run_{uuid.uuid4().hex[:6]}"
        p_item = PromptItem(
            id=custom_id,
            category=category.strip().lower(),
            prompt=custom_prompt.strip(),
            evaluation_type="custom_direct",
            metadata={"on_the_fly": True},
        )
        direct_items = [p_item]
        prompt_id = custom_id

    def progress_callback(pid: str, current: int, total: int) -> None:
        if current % 10 == 0 or current == total:
            typer.echo(f"  -> Prompt '{pid}': {current}/{total} successful runs completed.")

    typer.echo(f"Starting experiment execution...")
    exp_id = runner.run(
        experiment_id=experiment_id,
        prompt_id_filter=prompt_id,
        direct_prompts=direct_items,
        progress_cb=progress_callback,
    )

    typer.secho(f"\nExperiment '{exp_id}' completed!", fg=typer.colors.GREEN, bold=True)

    metrics = storage.get_experiment_metrics(exp_id)
    summary_text = format_cli_summary(metrics)
    typer.echo("\n" + summary_text)

    # Automatically generate default reports
    export_experiment_reports(
        storage=storage,
        experiment_id=exp_id,
        output_dir=config.reporting.report_dir,
        formats=config.reporting.output_formats,
    )
    typer.echo(f"\nReports saved in: {Path(config.reporting.report_dir) / exp_id}")


@app.command()
def analyze(
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment-id", "-e", help="Experiment ID to analyze (defaults to latest)"
    ),
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
) -> None:
    """Analyze and display variability metrics for an experiment."""
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    storage = Storage(config.storage.db_path)

    if not experiment_id:
        # Discover latest experiment
        with storage._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1;"
            ).fetchone()
            if not row:
                typer.secho("No experiments found in database.", fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)
            experiment_id = row["id"]

    typer.echo(f"Analyzing experiment: {experiment_id}")
    metrics = storage.get_experiment_metrics(experiment_id)
    if not metrics:
        typer.secho("No metrics found for this experiment.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    summary_text = format_cli_summary(metrics)
    typer.echo("\n" + summary_text)


@app.command()
def export(
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment-id", "-e", help="Experiment ID to export (defaults to latest)"
    ),
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Custom export directory"
    ),
    format_type: str = typer.Option(
        "all", "--format", "-f", help="Export format: json, csv, or all"
    ),
) -> None:
    """Export summary metrics and raw runs to JSON and CSV files."""
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    storage = Storage(config.storage.db_path)

    if not experiment_id:
        with storage._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1;"
            ).fetchone()
            if not row:
                typer.secho("No experiments found in database.", fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)
            experiment_id = row["id"]

    target_dir = output_dir or Path(config.reporting.report_dir)
    formats = ["json", "csv"] if format_type.lower() == "all" else [format_type.lower()]

    try:
        exported = export_experiment_reports(
            storage=storage,
            experiment_id=experiment_id,
            output_dir=target_dir,
            formats=formats,
        )
        typer.secho(f"Successfully exported {len(exported)} file(s) for '{experiment_id}':", fg=typer.colors.GREEN)
        for k, p in exported.items():
            typer.echo(f"  [{k}] {p.resolve()}")
    except Exception as e:
        typer.secho(f"Export error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("plot")
def plot_cmd(
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment-id", "-e", help="Experiment ID to plot (defaults to latest completed)"
    ),
    config_path: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to config.yaml"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Custom directory to store generated plots"
    ),
    sync_assets: bool = typer.Option(
        True, "--sync-assets/--no-sync-assets", help="Also sync generated plots to assets/plots for README display"
    ),
) -> None:
    """Generate publication-grade PNG and PDF visualization plots for an experiment."""
    import shutil
    try:
        config = load_config(config_path)
    except Exception as e:
        typer.secho(f"Error loading configuration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    storage = Storage(config.storage.db_path)

    if not experiment_id:
        with storage._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM experiments WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1;"
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1;"
                ).fetchone()
            if not row:
                typer.secho("No experiments found in database to plot.", fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)
            experiment_id = row["id"]

    typer.echo(f"Generating benchmark visualization plots for experiment: '{experiment_id}'...")
    from promptpulse.visualization import generate_all_plots

    try:
        plots = generate_all_plots(
            reports_dir=config.reporting.report_dir,
            experiment_id=experiment_id,
            output_plots_dir=output_dir,
            db_path=config.storage.db_path,
        )
        typer.secho(f"[OK] Generated {len(plots)} publication-quality plot(s):", fg=typer.colors.GREEN, bold=True)
        for name, path in plots.items():
            typer.echo(f"  -> [{name}] {path.resolve()}")

        if sync_assets:
            assets_dir = Path("assets/plots")
            assets_dir.mkdir(parents=True, exist_ok=True)
            for path in plots.values():
                shutil.copy2(path, assets_dir / path.name)
            typer.secho(f"[OK] Synced plots to: {assets_dir.resolve()} (Ready for README.md)", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"[ERROR] Plot generation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
