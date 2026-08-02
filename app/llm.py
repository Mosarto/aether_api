"""Central Agnes LLM layer.

Every LLM call in the API goes through this module. Routes and jobs never
touch HTTP details: they pick a use case, hand over messages, and get back
content (or a typed, secret-free error).

Responsibilities:
- authentication, endpoint building and AGNES_BASE_URL normalization;
- single shared httpx.AsyncClient (created/closed by the lifespan);
- per-use-case policies (temperature, max_tokens, timeout, retries);
- retry with exponential backoff, honoring capped Retry-After;
- transient vs permanent HTTP status classification;
- content extraction and strict structured-output validation with a single
  corrective retry;
- metrics and logs that never include prompts, responses or credentials.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app import config
from app.config import logger

# ---------------------------------------------------------------------------
# Typed errors — messages must stay safe to surface in logs and HTTP bodies:
# never include the API key, Authorization header, prompts or model output.
# ---------------------------------------------------------------------------


class AgnesError(Exception):
    """Base error for the Agnes layer."""

    retryable = False


class AgnesConfigError(AgnesError):
    """Required AGNES_* configuration is missing."""


class AgnesTimeoutError(AgnesError):
    retryable = True


class AgnesConnectionError(AgnesError):
    retryable = True


class AgnesHTTPError(AgnesError):
    def __init__(self, status_code: int, use_case: str, retryable: bool):
        super().__init__(f"Agnes HTTP {status_code} ({use_case})")
        self.status_code = status_code
        self.retryable = retryable


class AgnesResponseError(AgnesError):
    """2xx response whose body is not a usable completion."""


class AgnesInvalidOutputError(AgnesError):
    """Structured output still invalid after the single corrective retry."""


# Consistent 503 body routes return when Agnes is unavailable.
LLM_UNAVAILABLE_DETAIL = {"error": "llm_unavailable"}


def was_billed(error: BaseException) -> bool:
    """True when the provider already charged for the failed call.

    Callers use this to decide whether a reserved quota slot may be refunded:
    refunding a billed failure would let a user obtain unlimited paid
    completions by steering the model into invalid output.
    """
    return isinstance(error, (AgnesResponseError, AgnesInvalidOutputError))

# ---------------------------------------------------------------------------
# Per-use-case policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMPolicy:
    use_case: str
    temperature: float
    max_tokens: int
    timeout_seconds: float | None = None  # None → AGNES_TIMEOUT_SECONDS
    max_retries: int | None = None  # None → AGNES_MAX_RETRIES
    json_output: bool = False
    corrective_temperature: float = 0.2  # used by the single corrective retry


POLICIES: dict[str, LLMPolicy] = {
    p.use_case: p
    for p in (
        # Nyx main chat
        LLMPolicy("chat", temperature=0.7, max_tokens=600),
        # Session title from the first exchange
        LLMPolicy("session_title", temperature=0.3, max_tokens=24, timeout_seconds=15.0, max_retries=1),
        # History compression for long sessions
        LLMPolicy("history_compression", temperature=0.3, max_tokens=300),
        # Profile extraction (structured)
        LLMPolicy("profile_extraction", temperature=0.3, max_tokens=500, json_output=True),
        # Akashic metadata (structured)
        LLMPolicy("akashic_metadata", temperature=0.3, max_tokens=250, json_output=True),
        # Gender inference from display name
        LLMPolicy("gender_inference", temperature=0.0, max_tokens=8, timeout_seconds=10.0, max_retries=1),
        # Daily verse/reflection
        LLMPolicy("daily_verse", temperature=0.7, max_tokens=300),
        # Structured reflection prompt generation
        LLMPolicy("prompt_generation", temperature=0.7, max_tokens=1500, json_output=True),
        # AI tools (structured)
        LLMPolicy("dream", temperature=config.AI_TOOL_LLM_TEMPERATURE, max_tokens=config.AI_TOOL_LLM_MAX_TOKENS, json_output=True),
        LLMPolicy("aura", temperature=config.AI_TOOL_LLM_TEMPERATURE, max_tokens=config.AI_TOOL_LLM_MAX_TOKENS, json_output=True),
        LLMPolicy("stoic", temperature=config.AI_TOOL_LLM_TEMPERATURE, max_tokens=config.AI_TOOL_LLM_MAX_TOKENS, json_output=True),
        LLMPolicy("sync", temperature=config.AI_TOOL_LLM_TEMPERATURE, max_tokens=config.AI_TOOL_LLM_MAX_TOKENS, json_output=True),
    )
}

# ---------------------------------------------------------------------------
# Retry classification (per Agnes error-code docs + product policy)
# ---------------------------------------------------------------------------

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 520, 522, 524})
RETRY_BACKOFF_BASE_SECONDS = 0.5
RETRY_BACKOFF_MAX_SECONDS = 8.0
RETRY_AFTER_MAX_SECONDS = 15.0

# ---------------------------------------------------------------------------
# Configuration / endpoint
# ---------------------------------------------------------------------------

_REQUIRED_ENV = ("AGNES_API_KEY", "AGNES_BASE_URL", "AGNES_MODEL")


def missing_agnes_config() -> list[str]:
    """Names of required AGNES_* variables that are absent. Never logs values."""
    return [name for name in _REQUIRED_ENV if not str(getattr(config, name, "") or "").strip()]


def agnes_endpoint() -> str:
    """Build the chat-completions endpoint exactly once from AGNES_BASE_URL.

    The base URL already carries the API version (e.g. https://host/v1) and may
    end with a trailing slash; normalize it and append /chat/completions unless
    the base already points at it.
    """
    base = str(config.AGNES_BASE_URL or "").strip().rstrip("/")
    if not base:
        raise AgnesConfigError("AGNES_BASE_URL não configurada")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


# ---------------------------------------------------------------------------
# Shared async client (owned by the FastAPI lifespan)
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def init_llm_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=config.AGNES_TIMEOUT_SECONDS)
    return _client


async def close_llm_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def set_llm_client(client: httpx.AsyncClient | None) -> None:
    """Test hook: inject a client (e.g. httpx.MockTransport-backed)."""
    global _client
    _client = client


def _get_client() -> httpx.AsyncClient:
    return init_llm_client()


# ---------------------------------------------------------------------------
# Metrics (in-process, content-free)
# ---------------------------------------------------------------------------

_metrics: dict[str, dict[str, int]] = {}


def _metric(use_case: str) -> dict[str, int]:
    return _metrics.setdefault(use_case, {"calls": 0, "successes": 0, "retries": 0, "failures": 0})


def get_llm_metrics() -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in _metrics.items()}


# ---------------------------------------------------------------------------
# Untrusted-content helpers
# ---------------------------------------------------------------------------

# Tag names used by prompts to wrap untrusted content. Content embedded inside
# these tags must be neutralized first so it cannot close/reopen the wrapper.
_DATA_TAG_NAMES = (
    "dados_usuario", "historico", "perfil_atual", "resumo_conversa", "conteudo",
    "perfil_usuario", "conversas_recentes", "sinais", "memorias", "sugestoes",
)
_TAG_BREAKOUT_RE = re.compile(rf"<(?=\s*/?\s*(?:{'|'.join(_DATA_TAG_NAMES)})\b)", re.IGNORECASE)


def neutralize_delimiters(text: str) -> str:
    """Prevent untrusted text from escaping its data tag."""
    return _TAG_BREAKOUT_RE.sub("‹", text or "")


def wrap_untrusted(tag: str, text: str) -> str:
    """Wrap untrusted content in a named data tag, neutralizing breakouts."""
    return f"<{tag}>\n{neutralize_delimiters(text)}\n</{tag}>"


_FENCE_RE = re.compile(r"^```[\w-]*\s*\n?|\n?```\s*$")


def strip_markdown_fences(raw: str) -> str:
    """Remove a single wrapping Markdown code fence, if present. Nothing else."""
    clean = (raw or "").strip()
    if clean.startswith("```"):
        clean = _FENCE_RE.sub("", clean).strip()
    return clean


# ---------------------------------------------------------------------------
# Core completion call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, RETRY_AFTER_MAX_SECONDS)


def _backoff_seconds(attempt: int) -> float:
    return min(RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt), RETRY_BACKOFF_MAX_SECONDS)


def _extract_content(payload: dict[str, Any], use_case: str) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise AgnesResponseError(f"resposta Agnes sem choices ({use_case})")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise AgnesResponseError(f"resposta Agnes sem message ({use_case})")

    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise AgnesResponseError(f"resposta Agnes com conteúdo vazio ({use_case})")
    return content


def get_policy(use_case: str) -> LLMPolicy:
    try:
        return POLICIES[use_case]
    except KeyError:
        raise AgnesConfigError(f"caso de uso LLM desconhecido: {use_case}") from None


async def complete(
    use_case: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Run one chat completion for a use case, with retry on transient failures.

    Raises AgnesConfigError / AgnesTimeoutError / AgnesConnectionError /
    AgnesHTTPError / AgnesResponseError. Error messages are secret-free.
    """
    policy = get_policy(use_case)

    missing = missing_agnes_config()
    if missing:
        raise AgnesConfigError(f"configuração ausente: {', '.join(missing)}")

    endpoint = agnes_endpoint()
    timeout = policy.timeout_seconds if policy.timeout_seconds is not None else config.AGNES_TIMEOUT_SECONDS
    max_retries = policy.max_retries if policy.max_retries is not None else config.AGNES_MAX_RETRIES
    attempts = max(1, max_retries + 1)

    payload: dict[str, Any] = {
        "model": config.AGNES_MODEL,
        "messages": messages,
        "temperature": policy.temperature if temperature is None else temperature,
        "max_tokens": policy.max_tokens if max_tokens is None else max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {config.AGNES_API_KEY}",
        "Content-Type": "application/json",
    }

    metric = _metric(use_case)
    metric["calls"] += 1
    client = _get_client()
    started = time.perf_counter()
    last_error: AgnesError | None = None

    for attempt in range(attempts):
        try:
            response = await client.post(endpoint, headers=headers, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            last_error = AgnesTimeoutError(f"Agnes timeout após {timeout:.0f}s ({use_case})")
            if attempt < attempts - 1:
                metric["retries"] += 1
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            metric["failures"] += 1
            raise last_error
        except httpx.RequestError as exc:
            last_error = AgnesConnectionError(f"Agnes falha de conexão ({use_case}): {exc.__class__.__name__}")
            if attempt < attempts - 1:
                metric["retries"] += 1
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            metric["failures"] += 1
            raise last_error

        status = response.status_code
        if status in RETRYABLE_STATUS_CODES:
            if attempt < attempts - 1:
                delay = _retry_after_seconds(response)
                if delay is None:
                    delay = _backoff_seconds(attempt)
                logger.warning(
                    "agnes %s: HTTP %s transitório, retry em %.2fs (tentativa %d/%d)",
                    use_case, status, delay, attempt + 1, attempts,
                )
                metric["retries"] += 1
                await asyncio.sleep(delay)
                continue
            metric["failures"] += 1
            raise AgnesHTTPError(status, use_case, retryable=True)

        if status < 200 or status >= 300:
            # Permanent (400/401/402/403/404/405/413/415/422/431...): never retry.
            metric["failures"] += 1
            raise AgnesHTTPError(status, use_case, retryable=False)

        try:
            data = response.json()
        except ValueError:
            metric["failures"] += 1
            raise AgnesResponseError(f"resposta Agnes não-JSON ({use_case})") from None
        if not isinstance(data, dict):
            metric["failures"] += 1
            raise AgnesResponseError(f"payload Agnes inválido ({use_case})")

        content = _extract_content(data, use_case)
        metric["successes"] += 1
        logger.debug(
            "agnes %s: ok em %.0fms (%d chars, tentativa %d/%d)",
            use_case, (time.perf_counter() - started) * 1000, len(content), attempt + 1, attempts,
        )
        return LLMResult(content=content, model=config.AGNES_MODEL)

    # Unreachable: every loop path returns or raises.
    metric["failures"] += 1
    raise last_error or AgnesResponseError(f"Agnes sem resposta ({use_case})")


# ---------------------------------------------------------------------------
# Structured (JSON) completions
# ---------------------------------------------------------------------------

TModel = TypeVar("TModel", bound=BaseModel)

_CORRECTIVE_INSTRUCTION = (
    "Sua resposta anterior não estava no formato exigido. Responda novamente "
    "APENAS com o objeto JSON válido pedido no início — sem markdown, sem "
    "comentários, sem nenhum texto fora do JSON, com todos os campos "
    "obrigatórios e enums exatamente como especificados."
)


def _parse_json_dict(raw: str) -> dict[str, Any]:
    data = json.loads(strip_markdown_fences(raw))
    if not isinstance(data, dict):
        raise json.JSONDecodeError("objeto JSON esperado", raw[:1], 0)
    return data


async def complete_json_dict(use_case: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Completion that must return a JSON object; one corrective retry only."""
    policy = get_policy(use_case)
    result = await complete(use_case, messages)
    try:
        return _parse_json_dict(result.content)
    except (json.JSONDecodeError, ValueError):
        logger.warning("agnes %s: saída não-JSON, retry corretivo único", use_case)

    retry_messages = messages + [
        {"role": "assistant", "content": result.content[:4000]},
        {"role": "user", "content": _CORRECTIVE_INSTRUCTION},
    ]
    retry = await complete(use_case, retry_messages, temperature=policy.corrective_temperature)
    try:
        return _parse_json_dict(retry.content)
    except (json.JSONDecodeError, ValueError):
        raise AgnesInvalidOutputError(f"saída estruturada inválida após retry ({use_case})") from None


async def complete_json(
    use_case: str,
    messages: list[dict[str, Any]],
    schema: type[TModel],
) -> TModel:
    """Completion validated against a Pydantic schema; one corrective retry only."""
    policy = get_policy(use_case)
    result = await complete(use_case, messages)
    try:
        return schema.model_validate(_parse_json_dict(result.content))
    except (json.JSONDecodeError, ValueError, ValidationError):
        logger.warning("agnes %s: saída fora do schema, retry corretivo único", use_case)

    retry_messages = messages + [
        {"role": "assistant", "content": result.content[:4000]},
        {"role": "user", "content": _CORRECTIVE_INSTRUCTION},
    ]
    retry = await complete(use_case, retry_messages, temperature=policy.corrective_temperature)
    try:
        return schema.model_validate(_parse_json_dict(retry.content))
    except (json.JSONDecodeError, ValueError, ValidationError):
        raise AgnesInvalidOutputError(f"saída estruturada inválida após retry ({use_case})") from None
