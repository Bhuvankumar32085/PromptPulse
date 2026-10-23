"""Experiment runner orchestration: independent prompt execution, safe resumption, and tracking."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from promptpulse.config import AppConfig
from promptpulse.dataset import PromptItem, load_dataset
from promptpulse.metrics import compute_exact_metrics
from promptpulse.model import ModelProvider, get_model_provider
from promptpulse.storage import Storage

logger = logging.getLogger("promptpulse.experiment")


def generate_experiment_id() -> str:
    """Generate a clean timestamped experiment identifier."""
    return f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def is_error_retryable(error: Exception, retryable_categories: List[str]) -> bool:
    """Check if an exception matches configured retryable categories."""
    err_str = f"{type(error).__name__} {str(error)}".lower()
    for cat in retryable_categories:
        clean_cat = cat.lower().replace("_", "")
        if clean_cat in err_str.replace("_", ""):
            return True
    return False


class ExperimentRunner:
    """Orchestrates variability experiments across benchmark prompts."""

    def __init__(
        self,
        config: AppConfig,
        storage: Optional[Storage] = None,
        model_provider: Optional[ModelProvider] = None,
    ):
        self.config = config
        self.storage = storage or Storage(config.storage.db_path)
        self.model_provider = model_provider or get_model_provider(config.model)

    def run(
        self,
        experiment_id: Optional[str] = None,
        prompt_id_filter: Optional[str] = None,
        direct_prompts: Optional[List[PromptItem]] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> str:
        """Run or resume an experiment across benchmark prompts.

        Args:
            experiment_id: Specific experiment ID to create or resume.
            prompt_id_filter: Optional single prompt ID to filter the run.
            direct_prompts: Optional in-memory prompt items (e.g. for user custom prompt testing).
            progress_cb: Optional callback(prompt_id, current_successes, total_required).

        Returns:
            The experiment ID.
        """
        exp_id = experiment_id or generate_experiment_id()
        existing_exp = self.storage.get_experiment(exp_id)

        # Check existing or create new experiment with configuration snapshot
        if existing_exp is None:
            self.storage.save_experiment(
                experiment_id=exp_id,
                name=self.config.project.name,
                config_snapshot=self.config.to_snapshot_dict(),
                status="running",
            )
            logger.info(f"Created new experiment record '{exp_id}' with configuration snapshot.")
        else:
            self.storage.update_experiment_status(exp_id, "running")
            logger.info(f"Resuming existing experiment '{exp_id}'.")

        # Load prompts
        if direct_prompts:
            prompts = direct_prompts
        else:
            prompts = load_dataset(
                file_path=self.config.dataset.file_path,
                categories=self.config.dataset.categories,
                max_prompts_per_category=self.config.dataset.max_prompts_per_category,
                shuffle=self.config.dataset.shuffle,
                seed=self.config.project.random_seed,
            )

        if prompt_id_filter:
            prompts = [p for p in prompts if p.id == prompt_id_filter]
            if not prompts:
                raise ValueError(f"Prompt with ID '{prompt_id_filter}' not found in loaded dataset.")

        if self.config.experiment.shuffle_prompts:
            rng = random.Random(self.config.project.random_seed)
            rng.shuffle(prompts)

        # Upsert prompt records in database
        for p in prompts:
            self.storage.save_prompt(p)

        n_required = self.config.experiment.successful_responses_required
        max_retries = self.config.experiment.max_retries
        all_completed = True

        logger.info(
            f"Starting execution: {len(prompts)} prompt(s), "
            f"{n_required} successful response(s) required per prompt."
        )

        for p_idx, prompt in enumerate(prompts, start=1):
            logger.info(f"[{p_idx}/{len(prompts)}] Evaluating prompt '{prompt.id}' ({prompt.category})")

            # Determine completed successful runs for safe resumption
            if self.config.experiment.reuse_completed_runs:
                completed_indices = self.storage.get_completed_run_indices(exp_id, prompt.id)
            else:
                completed_indices = set()

            success_count = len(completed_indices)
            if success_count >= n_required:
                logger.info(
                    f"Prompt '{prompt.id}' already completed ({success_count}/{n_required}). Skipping."
                )
                self._recalculate_and_save_metrics(exp_id, prompt.id)
                continue

            # Compute remaining run indices needed: 1 to n_required
            needed_indices = [idx for idx in range(1, n_required + 1) if idx not in completed_indices]
            prompt_completed_all = True
            batch_size = max(1, getattr(self.config.generation, "batch_size", 1))

            i = 0
            while i < len(needed_indices):
                chunk_indices = needed_indices[i : i + batch_size]
                chunk_size = len(chunk_indices)
                attempts_for_chunk = 0
                chunk_succeeded = False

                while not chunk_succeeded:
                    attempts_for_chunk += 1
                    try:
                        # Concurrent batch inference without appending earlier outputs
                        results = self.model_provider.generate_batch(
                            prompt=prompt.prompt,
                            count=chunk_size,
                            generation_cfg=self.config.generation,
                        )

                        # Record each successful result
                        for run_index, gen_result in zip(chunk_indices, results):
                            self.storage.record_run(
                                experiment_id=exp_id,
                                prompt_id=prompt.id,
                                run_index=run_index,
                                status="success",
                                raw_response=gen_result.text,
                                latency_ms=gen_result.latency_ms,
                                prompt_tokens=gen_result.prompt_tokens,
                                completion_tokens=gen_result.completion_tokens,
                            )
                            success_count += 1
                            if progress_cb:
                                progress_cb(prompt.id, success_count, n_required)

                        chunk_succeeded = True
                        i += chunk_size

                    except Exception as err:
                        # Record failure entry for this attempt
                        self.storage.record_run(
                            experiment_id=exp_id,
                            prompt_id=prompt.id,
                            run_index=chunk_indices[0],
                            status="failure",
                            error_message=f"{type(err).__name__}: {str(err)}",
                        )

                        retryable = is_error_retryable(err, self.config.experiment.retryable_errors)
                        if retryable and attempts_for_chunk <= max_retries:
                            logger.warning(
                                f"Retryable error on batch {chunk_indices} for '{prompt.id}' "
                                f"(attempt {attempts_for_chunk}/{max_retries + 1}): {err}"
                            )
                        else:
                            logger.error(
                                f"Failed batch {chunk_indices} for '{prompt.id}' "
                                f"(attempts exhausted or non-retryable): {err}"
                            )
                            prompt_completed_all = False
                            all_completed = False
                            break

                if not chunk_succeeded:
                    break

            # Compute and persist exact metrics for the prompt
            self._recalculate_and_save_metrics(exp_id, prompt.id)

            if not prompt_completed_all:
                logger.warning(
                    f"Prompt '{prompt.id}' ended prematurely with {success_count}/{n_required} successes."
                )

        final_status = "completed" if all_completed else "incomplete"
        self.storage.update_experiment_status(exp_id, final_status)
        logger.info(f"Experiment '{exp_id}' finished with status: {final_status}")
        return exp_id

    def _recalculate_and_save_metrics(self, experiment_id: str, prompt_id: str) -> Dict[str, Any]:
        """Compute metrics from saved runs and update metrics table."""
        successful_runs = self.storage.get_successful_runs(experiment_id, prompt_id)
        failed_count = self.storage.get_failed_attempts_count(experiment_id, prompt_id)

        raw_responses = [r["raw_response"] for r in successful_runs if r["raw_response"] is not None]
        latencies = [r["latency_ms"] for r in successful_runs if r["latency_ms"] is not None]

        metrics = compute_exact_metrics(
            raw_responses=raw_responses,
            latencies_ms=latencies,
            failed_attempts_count=failed_count,
            normalization_policy=self.config.analysis.normalization_policy,
        )

        self.storage.save_metrics(experiment_id, prompt_id, metrics)
        return metrics
