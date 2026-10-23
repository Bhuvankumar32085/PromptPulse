"""Dataset loader and schema validation for PromptPulse benchmark prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from promptpulse.config import VALID_CATEGORIES


class PromptItem(BaseModel):
    """Schema for an individual benchmark prompt record."""
    id: str = Field(min_length=1, description="Unique prompt identifier")
    category: str = Field(description="One of maths, coding, factual, creative, ambiguous")
    prompt: str = Field(min_length=1, description="Exact prompt string sent to the model")
    evaluation_type: str = Field(default="exact_match", description="Evaluation method or rubric identifier")
    expected_answer: Optional[str] = None
    reference_source: Optional[str] = None
    test_case_file: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in VALID_CATEGORIES:
            raise ValueError(
                f"Unknown category '{v}'. Allowed categories: {sorted(list(VALID_CATEGORIES))}"
            )
        return clean

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Prompt id cannot be empty or whitespace.")
        return clean

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt text cannot be empty or whitespace.")
        return v


def load_dataset(
    file_path: str | Path,
    categories: Optional[List[str]] = None,
    max_prompts_per_category: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[PromptItem]:
    """Load and strictly validate prompts from a JSONL file.

    Args:
        file_path: Path to the JSONL dataset file.
        categories: Optional category whitelist.
        max_prompts_per_category: Optional cap per category.
        shuffle: Whether to shuffle prompts.
        seed: Random seed for deterministic shuffling.

    Returns:
        List of validated PromptItem instances.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If duplicate prompt IDs exist, or records fail validation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    prompts: List[PromptItem] = []
    seen_ids: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            try:
                raw_record = json.loads(line_str)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {err.msg}"
                ) from err

            if not isinstance(raw_record, dict):
                raise ValueError(
                    f"Record at {path}:{line_no} must be a JSON object, got {type(raw_record).__name__}"
                )

            try:
                item = PromptItem.model_validate(raw_record)
            except Exception as err:
                raise ValueError(
                    f"Schema validation error at {path}:{line_no}:\n{err}"
                ) from err

            if item.id in seen_ids:
                raise ValueError(
                    f"Duplicate prompt ID detected: '{item.id}' at {path}:{line_no}. "
                    f"Prompt IDs must be globally unique across the dataset."
                )
            seen_ids.add(item.id)
            prompts.append(item)

    if not prompts:
        raise ValueError(f"No prompt records found in {path}")

    # Whitelist filtering
    if categories:
        target_cats = set(c.strip().lower() for c in categories)
        prompts = [p for p in prompts if p.category in target_cats]

    # Deterministic shuffle if requested
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    # Per-category capping
    if max_prompts_per_category is not None and max_prompts_per_category > 0:
        counts_by_cat: Dict[str, int] = {}
        filtered: List[PromptItem] = []
        for p in prompts:
            c = p.category
            counts_by_cat[c] = counts_by_cat.get(c, 0)
            if counts_by_cat[c] < max_prompts_per_category:
                filtered.append(p)
                counts_by_cat[c] += 1
        prompts = filtered

    return prompts
