"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path
from promptpulse.config import AppConfig, load_config


def test_default_config_validity(tmp_path: Path):
    cfg = AppConfig()
    assert cfg.project.name == "PromptPulse"
    assert cfg.experiment.successful_responses_required == 100
    assert cfg.generation.temperature == 0.7
    assert cfg.generation.top_p == 0.9
    assert cfg.analysis.normalization_policy == "conservative"

    snapshot = cfg.to_snapshot_dict()
    assert isinstance(snapshot, dict)
    assert snapshot["project"]["name"] == "PromptPulse"


def test_load_config_from_yaml(tmp_path: Path):
    yaml_content = """
project:
  name: "CustomPulse"
  random_seed: 123
  logging_level: "DEBUG"
model:
  loading_source: "mock"
  model_id: "test-model"
experiment:
  successful_responses_required: 50
  max_retries: 2
dataset:
  categories: ["maths", "coding"]
  max_prompts_per_category: 2
storage:
  db_path: "custom.db"
"""
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    loaded = load_config(cfg_file)
    assert loaded.project.name == "CustomPulse"
    assert loaded.project.random_seed == 123
    assert loaded.model.loading_source == "mock"
    assert loaded.experiment.successful_responses_required == 50
    assert loaded.dataset.categories == ["maths", "coding"]


def test_invalid_category_fails(tmp_path: Path):
    yaml_content = """
dataset:
  categories: ["invalid_category_xyz"]
"""
    cfg_file = tmp_path / "bad_config.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid categories"):
        load_config(cfg_file)


def test_nonexistent_config_file_raises_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_path_file.yaml")


def test_zero_temperature_adjusts_do_sample():
    cfg = AppConfig(generation={"temperature": 0.0, "do_sample": True})
    assert cfg.generation.do_sample is False
