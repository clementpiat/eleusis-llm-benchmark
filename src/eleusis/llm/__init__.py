"""LLM client and player components for Eleusis."""

import logging
import os
from pathlib import Path

import yaml

from eleusis.llm.anthropic import AnthropicClient
from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    TruncationError,
)
from eleusis.llm.google import GoogleClient
from eleusis.llm.huggingface import HuggingFaceClient
from eleusis.llm.openai_client import OpenAIClient
from eleusis.llm.xai import XAIClient
from eleusis.player import LLMScientist  # Re-export for backward compat

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLLMClient",
    "AnthropicClient",
    "GoogleClient",
    "HuggingFaceClient",
    "OpenAIClient",
    "XAIClient",
    "LLMCallMetrics",
    "GenerateMetrics",
    "TruncationError",
    "LLMScientist",
    "create_client",
    "create_client_from_config",
    "load_model_config",
]


def find_models_yaml() -> Path:
    """Find models.yaml in project root."""
    # Try common locations
    candidates = [
        Path("models.yaml"),
        Path(__file__).parent.parent.parent.parent / "models.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("models.yaml not found")


def load_model_config(model_key: str) -> dict:
    """Load model configuration from models.yaml."""
    models_path = find_models_yaml()
    with open(models_path) as f:
        all_models = yaml.safe_load(f)

    if model_key not in all_models:
        available = list(all_models.keys())
        raise ValueError(f"Model '{model_key}' not found in models.yaml. Available: {available}")

    return all_models[model_key]


def create_client(
    model_key: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    seed: int | None = None,
) -> BaseLLMClient:
    """Create an LLM client based on model key from models.yaml.

    Args:
        model_key: Key referencing a model in models.yaml (e.g., "claude-opus", "gpt-5.2")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        role: Role identifier for metrics
        seed: Random seed for reproducibility

    Returns:
        Configured LLM client instance
    """
    config = load_model_config(model_key)
    provider = config["provider"]
    model_id = config["model_id"]

    common_kwargs = {
        "model_name": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "role": role,
        "seed": seed,
    }

    if provider == "anthropic":
        reasoning_budget = config.get("reasoning_budget", 8192)
        return AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            reasoning_budget=reasoning_budget,
            **common_kwargs,
        )

    elif provider == "openai":
        reasoning_effort = config.get("reasoning_effort", "medium")
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort=reasoning_effort,
            **common_kwargs,
        )

    elif provider == "google":
        thinking_level = config.get("thinking_level", "high")
        return GoogleClient(
            api_key=os.getenv("GOOGLE_API_KEY"),
            thinking_level=thinking_level,
            **common_kwargs,
        )

    elif provider == "xai":
        return XAIClient(
            api_key=os.getenv("XAI_API_KEY"),
            **common_kwargs,
        )

    elif provider == "huggingface":
        hf_provider = config.get("hf_provider")
        reasoning_format = config.get("reasoning_format", "separate_field")
        return HuggingFaceClient(
            api_key=os.getenv("HF_TOKEN"),
            hf_provider=hf_provider,
            reasoning_format=reasoning_format,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_client_from_config(
    config: dict,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    seed: int | None = None,
) -> BaseLLMClient:
    """Create an LLM client from inline config dict.

    Args:
        config: Dict with provider, model_id, and provider-specific options
        temperature: Sampling temperature (can be overridden by config)
        max_tokens: Maximum tokens to generate
        role: Role identifier for metrics
        seed: Random seed for reproducibility

    Returns:
        Configured LLM client instance
    """
    provider = config["provider"]
    model_id = config["model_id"]
    temperature = config.get("temperature", temperature)

    common_kwargs = {
        "model_name": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "role": role,
        "seed": seed,
    }

    if provider == "huggingface":
        hf_provider = config.get("hf_provider")  # None if not specified
        reasoning_format = config.get("reasoning_format", "separate_field")
        return HuggingFaceClient(
            api_key=os.getenv("HF_TOKEN"),
            hf_provider=hf_provider,
            reasoning_format=reasoning_format,
            **common_kwargs,
        )
    elif provider == "google":
        return GoogleClient(
            api_key=os.getenv("GOOGLE_API_KEY"),
            **common_kwargs,
        )

    # Could extend to other providers if needed
    raise ValueError(f"create_client_from_config only supports huggingface, got: {provider}")
