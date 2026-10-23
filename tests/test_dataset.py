"""Unit tests for dataset loading, schema validation, and duplicate rejection."""

import pytest
from pathlib import Path
from promptpulse.dataset import PromptItem, load_dataset


def test_valid_dataset_loading(tmp_path: Path):
    jsonl_content = (
        '{"id": "m1", "category": "maths", "prompt": "What is 2+2?", "expected_answer": "4"}\n'
        '{"id": "c1", "category": "coding", "prompt": "Write hello world."}\n'
        '{"id": "f1", "category": "factual", "prompt": "Capital of France?", "expected_answer": "Paris"}\n'
    )
    dataset_file = tmp_path / "prompts.jsonl"
    dataset_file.write_text(jsonl_content, encoding="utf-8")

    prompts = load_dataset(dataset_file)
    assert len(prompts) == 3
    assert prompts[0].id == "m1"
    assert prompts[0].category == "maths"
    assert prompts[0].expected_answer == "4"


def test_duplicate_prompt_id_rejected(tmp_path: Path):
    jsonl_content = (
        '{"id": "duplicate_id", "category": "maths", "prompt": "Prompt 1"}\n'
        '{"id": "duplicate_id", "category": "coding", "prompt": "Prompt 2"}\n'
    )
    dataset_file = tmp_path / "dup.jsonl"
    dataset_file.write_text(jsonl_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate prompt ID detected"):
        load_dataset(dataset_file)


def test_invalid_category_rejected(tmp_path: Path):
    jsonl_content = '{"id": "p1", "category": "unsupported_cat", "prompt": "Hello"}\n'
    dataset_file = tmp_path / "bad_cat.jsonl"
    dataset_file.write_text(jsonl_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown category"):
        load_dataset(dataset_file)


def test_empty_prompt_rejected(tmp_path: Path):
    jsonl_content = '{"id": "p1", "category": "maths", "prompt": "   "}\n'
    dataset_file = tmp_path / "empty_p.jsonl"
    dataset_file.write_text(jsonl_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Prompt text cannot be empty"):
        load_dataset(dataset_file)


def test_category_filtering_and_capping(tmp_path: Path):
    jsonl_content = (
        '{"id": "m1", "category": "maths", "prompt": "P1"}\n'
        '{"id": "m2", "category": "maths", "prompt": "P2"}\n'
        '{"id": "m3", "category": "maths", "prompt": "P3"}\n'
        '{"id": "c1", "category": "coding", "prompt": "P4"}\n'
    )
    dataset_file = tmp_path / "filter.jsonl"
    dataset_file.write_text(jsonl_content, encoding="utf-8")

    # Filter only maths and cap at 2
    prompts = load_dataset(dataset_file, categories=["maths"], max_prompts_per_category=2)
    assert len(prompts) == 2
    assert all(p.category == "maths" for p in prompts)
    assert [p.id for p in prompts] == ["m1", "m2"]
