"""Model provider interface, Mock provider for testing, and Hugging Face local loader."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from promptpulse.config import GenerationConfig, ModelConfig

logger = logging.getLogger("promptpulse.model")


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result of a single model inference attempt."""
    text: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Abstract base protocol for LLM response generation backends."""

    @abstractmethod
    def generate(self, prompt: str, generation_cfg: GenerationConfig) -> GenerationResult:
        """Generate a response given a prompt and sampling configuration.

        Args:
            prompt: User prompt text.
            generation_cfg: Sampling and length limits.

        Returns:
            GenerationResult containing the raw text and execution metadata.
        """
        raise NotImplementedError

    def generate_batch(self, prompt: str, count: int, generation_cfg: GenerationConfig) -> List[GenerationResult]:
        """Generate multiple responses in parallel or sequential batches."""
        return [self.generate(prompt, generation_cfg) for _ in range(count)]


class MockModelProvider(ModelProvider):
    """Controlled mock provider for fast, deterministic, non-downloading tests and verification.

    Simulates various real-world scenarios: identical outputs, unique sequences,
    custom repetition patterns, simulated latencies, and intermittent failures.
    """

    def __init__(
        self,
        mode: str = "identical",
        custom_responses: Optional[List[str]] = None,
        fail_first_n_calls: int = 0,
        fail_error_type: str = "timeout",
        base_latency_ms: float = 15.0,
    ):
        self.mode = mode
        self.custom_responses = custom_responses or []
        self.fail_first_n_calls = fail_first_n_calls
        self.fail_error_type = fail_error_type
        self.base_latency_ms = base_latency_ms
        self.call_count = 0

    def generate(self, prompt: str, generation_cfg: GenerationConfig) -> GenerationResult:
        self.call_count += 1
        t_start = time.perf_counter()

        # Simulate intermittent or transient failure if configured
        if self.call_count <= self.fail_first_n_calls:
            raise RuntimeError(f"Mock failure triggered: {self.fail_error_type}")

        if self.custom_responses:
            idx = (self.call_count - 1 - self.fail_first_n_calls) % len(self.custom_responses)
            text = self.custom_responses[idx]
        elif self.mode == "identical":
            text = "42"
        elif self.mode == "unique":
            text = f"Unique answer {self.call_count} to prompt: {prompt[:20]}"
        elif self.mode == "binary_alternating":
            text = "Option A" if (self.call_count % 2 == 1) else "Option B"
        else:
            text = f"Mock response {self.call_count}"

        latency = (time.perf_counter() - t_start) * 1000.0 + self.base_latency_ms
        tokens_est = max(1, len(text.split()))

        return GenerationResult(
            text=text,
            latency_ms=round(latency, 2),
            prompt_tokens=len(prompt.split()),
            completion_tokens=tokens_est,
            model_id="mock-model",
            metadata={"call_index": self.call_count, "mode": self.mode},
        )


class HuggingFaceModelProvider(ModelProvider):
    """Local Hugging Face Transformers model loader and inference engine.

    Loads model and tokenizer once, operates in evaluation mode with torch.inference_mode(),
    handles chat templates, and measures latency precisely.
    """

    def __init__(self, model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        self.model = None
        self.tokenizer = None
        self.resolved_device = None
        self._load_model()

    def _resolve_device_and_dtype(self) -> tuple[Any, Any]:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for HuggingFace local inference. "
                "Install torch or use model.loading_source='mock'."
            ) from e

        # Resolve device
        requested_dev = self.model_cfg.device
        if requested_dev == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_dev == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested in config (device='cuda') but no CUDA-enabled GPU is available."
                )
            device = torch.device("cuda")
        elif requested_dev == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device setting: {requested_dev}")

        # Resolve dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        requested_dtype = self.model_cfg.dtype
        if requested_dtype == "auto":
            # On CUDA, bfloat16 or float16 is preferred; on CPU, float32 is default
            dtype = torch.float16 if (device.type == "cuda" and torch.cuda.is_bf16_supported() is False) else (
                torch.bfloat16 if (device.type == "cuda") else torch.float32
            )
        else:
            if requested_dtype not in dtype_map:
                raise ValueError(f"Unsupported dtype '{requested_dtype}'. Choose from {list(dtype_map.keys())}")
            dtype = dtype_map[requested_dtype]

        return device, dtype

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Hugging Face transformers and torch must be installed to load models. "
                f"Underlying error: {e}"
            ) from e

        device, dtype = self._resolve_device_and_dtype()
        self.resolved_device = device
        logger.info(
            f"Loading Hugging Face model '{self.model_cfg.model_id}' "
            f"on device='{device}' with dtype='{dtype}'..."
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.model_revision,
                cache_dir=self.model_cfg.cache_dir,
                trust_remote_code=self.model_cfg.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg.model_id,
                revision=self.model_cfg.model_revision,
                cache_dir=self.model_cfg.cache_dir,
                torch_dtype=dtype,
                trust_remote_code=self.model_cfg.trust_remote_code,
            )
            self.model.to(device)
            self.model.eval()
            logger.info("Model and tokenizer successfully initialized and placed in eval mode.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face model '{self.model_cfg.model_id}': {e}"
            ) from e

    def generate(self, prompt: str, generation_cfg: GenerationConfig) -> GenerationResult:
        import torch

        # Format prompt with chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_input = prompt

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.resolved_device)
        input_length = inputs.input_ids.shape[1]

        # Configure generation parameters
        gen_kwargs = {
            "max_new_tokens": generation_cfg.max_new_tokens,
            "do_sample": generation_cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_cfg.do_sample:
            gen_kwargs["temperature"] = max(0.01, generation_cfg.temperature)
            gen_kwargs["top_p"] = generation_cfg.top_p

        # Precise latency measurement around model generation
        t_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        generated_ids = outputs[0][input_length:]
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            text=decoded_text,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=int(input_length),
            completion_tokens=int(generated_ids.shape[0]),
            model_id=self.model_cfg.model_id,
            metadata={
                "device": str(self.resolved_device),
                "temperature": generation_cfg.temperature,
                "top_p": generation_cfg.top_p,
            },
        )

    def generate_batch(self, prompt: str, count: int, generation_cfg: GenerationConfig) -> List[GenerationResult]:
        import torch

        if count <= 0:
            return []
        if count == 1:
            return [self.generate(prompt, generation_cfg)]

        # Format prompt with chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_input = prompt

        batch_texts = [formatted_input] * count
        inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.resolved_device)
        input_length = inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": generation_cfg.max_new_tokens,
            "do_sample": generation_cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_cfg.do_sample:
            gen_kwargs["temperature"] = max(0.01, generation_cfg.temperature)
            gen_kwargs["top_p"] = generation_cfg.top_p

        t_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        batch_latency_ms = (time.perf_counter() - t_start) * 1000.0
        avg_latency_ms = round(batch_latency_ms / count, 2)

        results: List[GenerationResult] = []
        for i in range(count):
            generated_ids = outputs[i][input_length:]
            decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=decoded_text,
                    latency_ms=avg_latency_ms,
                    prompt_tokens=int(input_length),
                    completion_tokens=int(generated_ids.shape[0]),
                    model_id=self.model_cfg.model_id,
                    metadata={
                        "device": str(self.resolved_device),
                        "temperature": generation_cfg.temperature,
                        "top_p": generation_cfg.top_p,
                        "batch_size": count,
                    },
                )
            )
        return results


def get_model_provider(model_cfg: ModelConfig) -> ModelProvider:
    """Factory creating the appropriate ModelProvider based on configuration."""
    if model_cfg.loading_source == "mock":
        return MockModelProvider()
    elif model_cfg.loading_source == "huggingface":
        return HuggingFaceModelProvider(model_cfg)
    else:
        raise ValueError(f"Unsupported model loading_source: {model_cfg.loading_source}")
