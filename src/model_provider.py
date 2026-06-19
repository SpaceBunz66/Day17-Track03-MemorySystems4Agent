from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and common aliases.

    The lab is expected to support several live providers, but the tests and
    benchmark also run fully offline. Keeping this function dependency-free
    makes provider configuration safe even when SDKs are not installed.
    """

    aliases = {
        "": "openai",
        "open-ai": "openai",
        "gpt": "openai",
        "google": "gemini",
        "google-genai": "gemini",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "local": "ollama",
        "open-router": "openrouter",
        "open_router": "openrouter",
        "openai-compatible": "custom",
        "compatible": "custom",
    }
    normalized = aliases.get((value or "").strip().lower(), (value or "openai").strip().lower())
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported provider '{value}'. Supported providers: {sorted(supported)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate a LangChain chat model for the selected provider.

    This function imports SDK integrations lazily so the deterministic offline
    benchmark can run without API keys or optional provider packages.
    """

    provider = normalize_provider(config.provider)

    def clean_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in raw.items() if value is not None}

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                **clean_kwargs(
                    {
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "api_key": config.api_key,
                    }
                )
            )

        if provider == "custom":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                **clean_kwargs(
                    {
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "api_key": config.api_key,
                        "base_url": config.base_url,
                    }
                )
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                **clean_kwargs(
                    {
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "google_api_key": config.api_key,
                    }
                )
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                **clean_kwargs(
                    {
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "api_key": config.api_key,
                    }
                )
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                **clean_kwargs(
                    {
                        "model": config.model_name,
                        "temperature": config.temperature,
                        "base_url": config.base_url,
                    }
                )
            )

        if provider == "openrouter":
            try:
                from langchain_openrouter import ChatOpenRouter

                return ChatOpenRouter(
                    **clean_kwargs(
                        {
                            "model": config.model_name,
                            "temperature": config.temperature,
                            "api_key": config.api_key,
                        }
                    )
                )
            except ImportError:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    **clean_kwargs(
                        {
                            "model": config.model_name,
                            "temperature": config.temperature,
                            "api_key": config.api_key,
                            "base_url": config.base_url or "https://openrouter.ai/api/v1",
                        }
                    )
                )
    except ImportError as exc:
        raise RuntimeError(
            f"Provider package for '{provider}' is not installed. "
            "Install the optional LangChain provider dependency or run the lab offline."
        ) from exc

    raise ValueError(f"Unsupported provider '{provider}'")
