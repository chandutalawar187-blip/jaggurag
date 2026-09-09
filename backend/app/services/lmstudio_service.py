"""Ollama service — OpenAI-compatible local Qwen integration."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _available_model_ids() -> list[str]:
    """Return Ollama model IDs, preferring generation-capable models."""
    with httpx.Client(timeout=5.0) as client:
        response = client.get(f"{settings.OLLAMA_BASE_URL}/models")
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models if model.get("id")]


def _select_model(model_ids: list[str]) -> str:
    """Select the configured model, then a Qwen model, never an embedding model."""
    configured = settings.OLLAMA_MODEL
    if configured in model_ids:
        return configured
    qwen_model = next((model_id for model_id in model_ids if "qwen" in model_id.lower()), None)
    if qwen_model:
        return qwen_model
    non_embedding = next(
        (model_id for model_id in model_ids if "embed" not in model_id.lower()),
        None,
    )
    return non_embedding or configured


def _resolve_model(model_name: Optional[str]) -> str:
    if not model_name or model_name.lower() == "local_qwen":
        try:
            return _select_model(_available_model_ids())
        except Exception:
            pass
        return settings.OLLAMA_MODEL
    return model_name


def generate(
    prompt: str,
    system_prompt: str = "",
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Generate a response using Ollama's OpenAI-compatible API."""
    model_name = _resolve_model(model_name)
    base_url = settings.OLLAMA_BASE_URL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except httpx.ConnectError:
        raise RuntimeError(
            "Local Qwen is offline. Start Ollama and ensure the Qwen model is installed."
        )
    except Exception as e:
        log.error("LM Studio generation error: %s", e)
        raise


async def stream_generate(
    prompt: str,
    system_prompt: str = "",
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AsyncGenerator[str, None]:
    """Stream a response from Ollama."""
    model_name = _resolve_model(model_name)
    base_url = settings.OLLAMA_BASE_URL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
    except httpx.ConnectError:
        yield "Error: Local Qwen is offline. Start Ollama and ensure the Qwen model is installed."
    except Exception as e:
        log.error("LM Studio stream error: %s", e)
        yield f"Error: {str(e)}"


def health_check() -> bool:
    """Check if LM Studio is accessible."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{settings.OLLAMA_BASE_URL}/models")
            return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> list[dict]:
    """Return installed Ollama models."""
    is_available = health_check()
    model_name = settings.OLLAMA_MODEL
    if is_available:
        try:
            model_name = _select_model(_available_model_ids())
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            log.warning("Unable to discover Ollama model name: %s", exc)

    return [{
        "id": "local_qwen",
        "name": f"Local Qwen ({model_name})",
        "provider": "ollama",
        "available": is_available,
    }]
