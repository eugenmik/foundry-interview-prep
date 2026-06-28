"""Thin wrapper around the OpenRouter API.

Handles chat completions, embeddings and live model pricing. All network
access for the app goes through this module so the rest of the code stays
provider-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Chat models the user can pick from (Sprint 1 allow-list).
# label -> model id
CHAT_MODELS = {
    "GPT-5 mini (default)": "openai/gpt-5-mini",
    "GPT-5 nano (cheaper)": "openai/gpt-5-nano",
    "GPT-5 (highest quality)": "openai/gpt-5",
}

# Embedding model used for the RAG / vector-store features.
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Model used for the optional image-generation feature.
IMAGE_MODEL = "google/gemini-2.5-flash-image"

DEFAULT_TIMEOUT = 90


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error or is unreachable."""


@dataclass
class ChatResult:
    """Result of a chat completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def get_api_key() -> str | None:
    """Return the configured OpenRouter API key, if any."""
    return os.environ.get("OPENROUTER_API_KEY")


def _headers() -> dict[str, str]:
    key = get_api_key()
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Optional attribution headers (recommended by OpenRouter).
    if os.environ.get("OPENROUTER_APP_URL"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_APP_URL"]
    if os.environ.get("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
    return headers


def chat(
    messages: list[dict[str, str]],
    model: str = "openai/gpt-5-mini",
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 1200,
    json_mode: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> ChatResult:
    """Call the chat-completions endpoint and return text + token usage.

    When ``json_mode`` is True we request a JSON object response so the
    structured-output features can parse the result reliably.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # network-level failure
        raise OpenRouterError(f"Network error calling OpenRouter: {exc}") from exc

    if resp.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter returned {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"Unexpected response shape: {data}") from exc

    usage = data.get("usage", {}) or {}
    return ChatResult(
        text=text,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        model=data.get("model", model),
        raw=data,
    )


def embed(texts: list[str], model: str = EMBEDDING_MODEL,
          timeout: int = DEFAULT_TIMEOUT) -> list[list[float]]:
    """Return embedding vectors for a list of texts."""
    if not texts:
        return []
    payload = {"model": model, "input": texts}
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE}/embeddings",
            headers=_headers(),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(f"Network error calling embeddings: {exc}") from exc

    if resp.status_code != 200:
        raise OpenRouterError(
            f"Embeddings endpoint returned {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    return [item["embedding"] for item in data.get("data", [])]


# Module-level cache so we only hit the /models endpoint once per process.
_PRICING_CACHE: dict[str, dict[str, float]] = {}


def get_pricing(model: str) -> dict[str, float] | None:
    """Return {'prompt': usd_per_token, 'completion': usd_per_token} for a model.

    Pricing is fetched live from the OpenRouter /models endpoint and cached.
    Returns None if pricing could not be retrieved.
    """
    if not _PRICING_CACHE:
        try:
            resp = requests.get(f"{OPENROUTER_BASE}/models", timeout=30)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    pricing = m.get("pricing", {}) or {}
                    try:
                        _PRICING_CACHE[m["id"]] = {
                            "prompt": float(pricing.get("prompt", 0) or 0),
                            "completion": float(pricing.get("completion", 0) or 0),
                        }
                    except (TypeError, ValueError):
                        continue
        except requests.RequestException:
            return None
    return _PRICING_CACHE.get(model)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate the USD cost of a call given token usage and live pricing."""
    pricing = get_pricing(model)
    if not pricing:
        return None
    return prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
