"""Configuration schema validation and loading for PromptPulse."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectConfig(BaseModel):
    """General project runtime parameters."""
    name: str = Field(default="PromptPulse", min_length=1)
    random_seed: int = Field(default=42)
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class ModelConfig(BaseModel):
    """Model backend and loading parameters."""
    loading_source: Literal["huggingface", "mock"] = "huggingface"
    model_id: str = Field(default="HuggingFaceTB/SmolLM2-1.7B-Instruct", min_length=1)
    model_revision: Optional[str] = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id cannot be an empty or whitespace string.")
        return v.strip()


class ExperimentConfig(BaseModel):
    """Experiment sampling constraints and resumption controls."""
    successful_responses_required: int = Field(default=100, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retryable_errors: List[str] = Field(
        default_factory=lambda: [
            "rate_limit",
            "timeout",
            "connection_error",
            "out_of_memory",
            "generation_exception",
        ]
    )
    shuffle_prompts: bool = False
    reuse_completed_runs: bool = True


class GenerationConfig(BaseModel):
    """LLM decoding and generation hyper-parameters."""
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=256, ge=1)
    do_sample: bool = True
    batch_size: int = Field(default=5, ge=1, description="Number of parallel generations per batch")

    @model_validator(mode="after")
    def validate_sampling_consistency(self) -> GenerationConfig:
        if self.temperature == 0.0 and self.do_sample:
            # Temperature 0.0 mathematically represents greedy decoding
            logging.getLogger("promptpulse").warning(
                "Temperature is 0.0; forcing do_sample=False for deterministic greedy decoding."
            )
            self.do_sample = False
        return self


VALID_CATEGORIES = {"maths", "coding", "factual", "creative", "ambiguous"}


class DatasetConfig(BaseModel):
    """Prompt benchmark dataset loading settings."""
    source: Literal["local_jsonl"] = "local_jsonl"
    file_path: str = Field(default="datasets/prompts.jsonl")
    categories: List[str] = Field(
        default_factory=lambda: ["maths", "coding", "factual", "creative", "ambiguous"]
    )
    max_prompts_per_category: Optional[int] = Field(default=20, ge=1)
    prompts_per_category: int = Field(default=20, ge=1, description="Target prompts to download/test per category")
    subset: Optional[str] = None
    shuffle: bool = False

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, categories: List[str]) -> List[str]:
        if not categories:
            raise ValueError("categories cannot be empty. Specify at least one category.")
        invalid = [c for c in categories if c not in VALID_CATEGORIES]
        if invalid:
            raise ValueError(
                f"Invalid categories: {invalid}. Supported categories are: {sorted(list(VALID_CATEGORIES))}"
            )
        return categories


class AnalysisConfig(BaseModel):
    """Variability and evaluation analysis settings."""
    exact_response_analysis: bool = True
    normalization_policy: Literal["conservative", "raw"] = "conservative"
    semantic_analysis: bool = False
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_clustering_threshold: float = Field(default=0.85, gt=0.0, lt=1.0)
    correctness_evaluation: bool = False


class StorageConfig(BaseModel):
    """Persistence and database settings."""
    db_path: str = "data/promptpulse.db"
    retain_raw_responses: bool = True
    retain_failed_attempts: bool = True


class ReportingConfig(BaseModel):
    """Output formats and destination settings."""
    output_formats: List[Literal["json", "csv"]] = Field(default_factory=lambda: ["json", "csv"])
    report_dir: str = "reports"

    @field_validator("output_formats")
    @classmethod
    def validate_output_formats(cls, formats: List[str]) -> List[str]:
        if not formats:
            raise ValueError("At least one output format ('json' or 'csv') must be specified.")
        return formats


class AppConfig(BaseModel):
    """Root configuration container combining all component settings."""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    def to_snapshot_dict(self) -> dict[str, Any]:
        """Generate a serializable snapshot of the validated configuration."""
        return self.model_dump(mode="json")


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.
    
    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If YAML syntax is invalid or configuration schema validation fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed YAML in {path}: {e}") from e

    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration in {path} must be a YAML dictionary mapping.")

    try:
        config = AppConfig.model_validate(raw_data)
    except Exception as e:
        raise ValueError(f"Configuration validation error in {path}:\n{e}") from e

    return config
