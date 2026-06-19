from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a complete lab config.

    All API-related settings are optional because the lab has a deterministic
    offline path for tests and benchmarking.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env_path = root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    judge_provider = normalize_provider(os.getenv("JUDGE_LLM_PROVIDER", provider))

    default_models = {
        "openai": "gpt-4o-mini",
        "custom": os.getenv("CUSTOM_MODEL", "gpt-4o-mini"),
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }

    def api_key_for(name: str) -> str | None:
        env_names = {
            "openai": "OPENAI_API_KEY",
            "custom": "CUSTOM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": "",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_name = env_names[name]
        return os.getenv(env_name) if env_name else None

    def base_url_for(name: str) -> str | None:
        env_names = {
            "custom": "CUSTOM_BASE_URL",
            "ollama": "OLLAMA_BASE_URL",
            "openrouter": "OPENROUTER_BASE_URL",
        }
        return os.getenv(env_names[name]) if name in env_names else None

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL", default_models[provider]),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        api_key=api_key_for(provider),
        base_url=base_url_for(provider),
    )
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_LLM_MODEL", os.getenv("LLM_MODEL", default_models[judge_provider])),
        temperature=float(os.getenv("JUDGE_LLM_TEMPERATURE", "0")),
        api_key=api_key_for(judge_provider),
        base_url=base_url_for(judge_provider),
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=judge_model,
    )
