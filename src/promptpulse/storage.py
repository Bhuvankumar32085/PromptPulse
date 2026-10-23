"""SQLite database persistence layer for PromptPulse experiments and runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from promptpulse.dataset import PromptItem


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hex digest for response text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Storage:
    """Dedicated SQLite storage manager for experiments, prompts, runs, and metrics."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tables and unique indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_snapshot TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    prompt_text TEXT NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    expected_answer TEXT,
                    reference_source TEXT,
                    test_case_file TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    prompt_id TEXT NOT NULL REFERENCES prompts(id),
                    run_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    raw_response TEXT,
                    response_hash TEXT,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );

                -- Enforce that successful run indices for the same experiment and prompt are strictly unique
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_successful_run 
                ON runs (experiment_id, prompt_id, run_index) 
                WHERE status = 'success';

                CREATE INDEX IF NOT EXISTS idx_runs_lookup 
                ON runs (experiment_id, prompt_id, status);

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    prompt_id TEXT NOT NULL REFERENCES prompts(id),
                    computed_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    UNIQUE(experiment_id, prompt_id)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    evaluator_name TEXT NOT NULL,
                    score REAL,
                    label TEXT,
                    details_json TEXT,
                    evaluated_at TEXT NOT NULL
                );
                """
            )

    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        config_snapshot: Dict[str, Any],
        status: str = "running",
    ) -> None:
        """Create or update an experiment record with an immutable configuration snapshot."""
        now = datetime.now(timezone.utc).isoformat()
        snapshot_json = json.dumps(config_snapshot, sort_keys=True, indent=2)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (id, name, created_at, config_snapshot, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status = excluded.status;
                """,
                (experiment_id, name, now, snapshot_json, status),
            )

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """Update experiment execution status."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE experiments SET status = ? WHERE id = ?;",
                (status, experiment_id),
            )

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment metadata and configuration snapshot."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?;", (experiment_id,)
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "status": row["status"],
                "config_snapshot": json.loads(row["config_snapshot"]),
            }

    def save_prompt(self, item: PromptItem) -> None:
        """Upsert prompt item."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO prompts (
                    id, category, prompt_text, evaluation_type,
                    expected_answer, reference_source, test_case_file, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    prompt_text = excluded.prompt_text,
                    evaluation_type = excluded.evaluation_type,
                    expected_answer = excluded.expected_answer,
                    reference_source = excluded.reference_source,
                    test_case_file = excluded.test_case_file,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    item.id,
                    item.category,
                    item.prompt,
                    item.evaluation_type,
                    item.expected_answer,
                    item.reference_source,
                    item.test_case_file,
                    json.dumps(item.metadata),
                ),
            )

    def record_run(
        self,
        experiment_id: str,
        prompt_id: str,
        run_index: int,
        status: str,
        raw_response: Optional[str] = None,
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> int:
        """Record an attempt (success or failure) in the database."""
        now = datetime.now(timezone.utc).isoformat()
        resp_hash = compute_sha256(raw_response) if raw_response is not None else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    experiment_id, prompt_id, run_index, status,
                    raw_response, response_hash, latency_ms,
                    prompt_tokens, completion_tokens, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    experiment_id,
                    prompt_id,
                    run_index,
                    status,
                    raw_response,
                    resp_hash,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    error_message,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_completed_run_indices(self, experiment_id: str, prompt_id: str) -> Set[int]:
        """Fetch indices of already completed successful runs for a prompt to allow safe resumption."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT run_index FROM runs
                WHERE experiment_id = ? AND prompt_id = ? AND status = 'success'
                ORDER BY run_index ASC;
                """,
                (experiment_id, prompt_id),
            ).fetchall()
            return {row["run_index"] for row in rows}

    def get_successful_runs(self, experiment_id: str, prompt_id: str) -> List[Dict[str, Any]]:
        """Retrieve all successful runs for a prompt ordered by run_index."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE experiment_id = ? AND prompt_id = ? AND status = 'success'
                ORDER BY run_index ASC;
                """,
                (experiment_id, prompt_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_failed_attempts_count(self, experiment_id: str, prompt_id: str) -> int:
        """Count failed attempts for a given prompt in an experiment."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count FROM runs
                WHERE experiment_id = ? AND prompt_id = ? AND status != 'success';
                """,
                (experiment_id, prompt_id),
            ).fetchone()
            return row["count"] if row else 0

    def save_metrics(
        self, experiment_id: str, prompt_id: str, metrics: Dict[str, Any]
    ) -> None:
        """Persist calculated metrics dictionary for a prompt."""
        now = datetime.now(timezone.utc).isoformat()
        metrics_json = json.dumps(metrics, sort_keys=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics (experiment_id, prompt_id, computed_at, metrics_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(experiment_id, prompt_id) DO UPDATE SET
                    computed_at = excluded.computed_at,
                    metrics_json = excluded.metrics_json;
                """,
                (experiment_id, prompt_id, now, metrics_json),
            )

    def get_experiment_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Retrieve metrics for all evaluated prompts in an experiment."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.prompt_id, p.category, p.prompt_text, m.computed_at, m.metrics_json
                FROM metrics m
                JOIN prompts p ON m.prompt_id = p.id
                WHERE m.experiment_id = ?
                ORDER BY p.category, m.prompt_id;
                """,
                (experiment_id,),
            ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["metrics"] = json.loads(item.pop("metrics_json"))
                results.append(item)
            return results

    def get_all_experiment_runs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Retrieve all runs (successful and failed) for an experiment with prompt details."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT r.*, p.category, p.prompt_text
                FROM runs r
                JOIN prompts p ON r.prompt_id = p.id
                WHERE r.experiment_id = ?
                ORDER BY r.prompt_id, r.run_index ASC, r.id ASC;
                """,
                (experiment_id,),
            ).fetchall()
            return [dict(r) for r in rows]
