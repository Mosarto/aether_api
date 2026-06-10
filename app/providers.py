import time
from typing import Any

import httpx
from qdrant_client import QdrantClient

from app.config import (
    QDRANT_URL, QDRANT_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_ENDPOINT,
    OPENROUTER_CHAT_MODEL,
    OPENROUTER_BACKGROUND_MODEL,
    OPENROUTER_TIMEOUT_SECONDS,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_RETRY_STATUS_CODES,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_APP_TITLE,
    logger,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def _openrouter_headers() -> dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY não configurada")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-OpenRouter-Title"] = OPENROUTER_APP_TITLE
    return headers


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _extract_content(payload: dict[str, Any], model: str) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response missing choices for model {model}")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"OpenRouter response has invalid choice for model {model}")

    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"OpenRouter response missing message for model {model}")

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"OpenRouter response empty content for model {model}")
    return content


def _create_completion(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    attempts = OPENROUTER_MAX_RETRIES + 1
    last_status: int | None = None
    for attempt in range(attempts):
        try:
            response = httpx.post(
                OPENROUTER_ENDPOINT,
                headers=_openrouter_headers(),
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"OpenRouter request timed out for model {model}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"OpenRouter request failed for model {model}: {exc.__class__.__name__}") from exc

        last_status = response.status_code
        if response.status_code in OPENROUTER_RETRY_STATUS_CODES and attempt < attempts - 1:
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                logger.warning(
                    "OpenRouter %s returned HTTP %s; retrying after %.2fs",
                    model,
                    response.status_code,
                    retry_after,
                )
                time.sleep(retry_after)
                continue

        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"OpenRouter request failed with HTTP {response.status_code} for model {model}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"OpenRouter returned non-JSON response for model {model}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"OpenRouter returned invalid JSON payload for model {model}")

        return _extract_content(data, model), f"openrouter-{model}"

    raise RuntimeError(f"OpenRouter request exhausted retries for model {model}; last_status={last_status}")


def create_chat_completion(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> tuple[str, str]:
    return _create_completion(OPENROUTER_CHAT_MODEL, messages, temperature, max_tokens)


def create_background_completion(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> tuple[str, str]:
    return _create_completion(OPENROUTER_BACKGROUND_MODEL, messages, temperature, max_tokens)
