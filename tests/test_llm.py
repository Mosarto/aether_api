"""Unit tests for the central Agnes layer (app/llm.py)."""

import asyncio

import httpx
import pytest

from app import config, llm
from app.llm import (
    AgnesConfigError,
    AgnesConnectionError,
    AgnesHTTPError,
    AgnesResponseError,
    AgnesTimeoutError,
    agnes_endpoint,
    complete,
    missing_agnes_config,
)


def run(coro):
    return asyncio.run(coro)


def _messages():
    return [{"role": "user", "content": "ping"}]


# --- Configuration -----------------------------------------------------------


def test_missing_env_vars_fail_before_network(monkeypatch, agnes):
    monkeypatch.setattr(config, "AGNES_API_KEY", "")
    monkeypatch.setattr(config, "AGNES_MODEL", "")

    missing = missing_agnes_config()
    assert missing == ["AGNES_API_KEY", "AGNES_MODEL"]

    with pytest.raises(AgnesConfigError) as err:
        run(complete("chat", _messages()))

    assert "AGNES_API_KEY" in str(err.value)
    assert agnes.call_count == 0, "Config ausente não pode gerar chamada de rede"


def test_all_env_present_reports_nothing_missing():
    assert missing_agnes_config() == []


def test_base_url_normalization_trailing_slash(monkeypatch):
    monkeypatch.setattr(config, "AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1/")
    assert agnes_endpoint() == "https://apihub.agnes-ai.com/v1/chat/completions"


def test_base_url_normalization_no_duplicate_path(monkeypatch):
    monkeypatch.setattr(config, "AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1/chat/completions")
    endpoint = agnes_endpoint()
    assert endpoint == "https://apihub.agnes-ai.com/v1/chat/completions"
    assert endpoint.count("/chat/completions") == 1
    assert endpoint.count("/v1") == 1


def test_base_url_empty_raises_config_error(monkeypatch):
    monkeypatch.setattr(config, "AGNES_BASE_URL", "  ")
    with pytest.raises(AgnesConfigError):
        agnes_endpoint()


# --- Request construction ----------------------------------------------------


def test_request_hits_normalized_endpoint(agnes):
    run(complete("chat", _messages()))
    assert str(agnes.requests[0].url) == "https://agnes.test/v1/chat/completions"
    assert agnes.requests[0].method == "POST"


def test_request_uses_model_from_env(agnes):
    run(complete("chat", _messages()))
    assert agnes.bodies[0]["model"] == config.AGNES_MODEL
    assert agnes.bodies[0]["model"] == "agnes-2.5-flash"


def test_request_sends_bearer_auth(agnes):
    run(complete("chat", _messages()))
    assert agnes.requests[0].headers["Authorization"] == f"Bearer {config.AGNES_API_KEY}"


def test_request_uses_max_tokens_not_advanced_params(agnes):
    run(complete("chat", _messages()))
    body = agnes.bodies[0]
    assert "max_tokens" in body
    assert body["max_tokens"] == llm.POLICIES["chat"].max_tokens
    assert "max_completion_tokens" not in body
    assert "response_format" not in body, "response_format não é documentado pela Agnes"
    assert "chat_template_kwargs" not in body, "Thinking Mode não deve ser habilitado por padrão"


def test_policy_overrides_apply(agnes):
    run(complete("chat", _messages(), temperature=0.1, max_tokens=42))
    assert agnes.bodies[0]["temperature"] == 0.1
    assert agnes.bodies[0]["max_tokens"] == 42


def test_unknown_use_case_rejected(agnes):
    with pytest.raises(AgnesConfigError):
        run(complete("nao_existe", _messages()))
    assert agnes.call_count == 0


# --- Response handling -------------------------------------------------------


def test_valid_response_extracted(agnes):
    agnes.enqueue(agnes.ok("olá mundo"))
    result = run(complete("chat", _messages()))
    assert result.content == "olá mundo"
    assert result.model == "agnes-2.5-flash"


def test_content_parts_list_joined(agnes):
    agnes.enqueue(httpx.Response(200, json={
        "choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]
    }))
    assert run(complete("chat", _messages())).content == "ab"


def test_empty_content_raises(agnes):
    agnes.enqueue(agnes.ok("   "))
    with pytest.raises(AgnesResponseError):
        run(complete("chat", _messages()))


def test_missing_choices_raises(agnes):
    agnes.enqueue(httpx.Response(200, json={"choices": []}))
    with pytest.raises(AgnesResponseError):
        run(complete("chat", _messages()))


def test_non_json_body_raises(agnes):
    agnes.enqueue(httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(AgnesResponseError):
        run(complete("chat", _messages()))


# --- Transport failures ------------------------------------------------------


def test_timeout_retries_then_raises_typed_error(agnes):
    def _timeout(request):
        raise httpx.ReadTimeout("boom", request=request)

    agnes.handle(_timeout)
    with pytest.raises(AgnesTimeoutError):
        run(complete("chat", _messages()))
    assert agnes.call_count == config.AGNES_MAX_RETRIES + 1


def test_connection_error_retries_then_raises_typed_error(agnes):
    def _refused(request):
        raise httpx.ConnectError("refused", request=request)

    agnes.handle(_refused)
    with pytest.raises(AgnesConnectionError):
        run(complete("chat", _messages()))
    assert agnes.call_count == config.AGNES_MAX_RETRIES + 1


def test_transport_error_recovers_on_retry(agnes):
    agnes.enqueue(httpx.ConnectError("refused"), agnes.ok("recuperado"))
    result = run(complete("chat", _messages()))
    assert result.content == "recuperado"
    assert agnes.call_count == 2


# --- HTTP status retry policy ------------------------------------------------


def test_retry_on_429_honors_retry_after(agnes):
    agnes.enqueue(
        httpx.Response(429, json={"error": "rate"}, headers={"Retry-After": "2"}),
        agnes.ok("depois"),
    )
    result = run(complete("chat", _messages()))
    assert result.content == "depois"
    assert agnes.call_count == 2
    assert agnes.sleep_delays == [2.0]


def test_retry_after_is_capped(agnes):
    agnes.enqueue(
        httpx.Response(429, json={}, headers={"Retry-After": "99999"}),
        agnes.ok(),
    )
    run(complete("chat", _messages()))
    assert agnes.sleep_delays == [llm.RETRY_AFTER_MAX_SECONDS]


def test_retry_without_retry_after_uses_exponential_backoff(agnes):
    agnes.enqueue(
        httpx.Response(503, json={}),
        httpx.Response(503, json={}),
        agnes.ok("enfim"),
    )
    result = run(complete("chat", _messages()))
    assert result.content == "enfim"
    assert agnes.call_count == 3
    assert agnes.sleep_delays == [
        llm.RETRY_BACKOFF_BASE_SECONDS,
        llm.RETRY_BACKOFF_BASE_SECONDS * 2,
    ]


def test_retryable_statuses_exhaust_and_raise(agnes):
    agnes.handle(lambda request: httpx.Response(503, json={}))
    with pytest.raises(AgnesHTTPError) as err:
        run(complete("chat", _messages()))
    assert err.value.status_code == 503
    assert err.value.retryable is True
    assert agnes.call_count == config.AGNES_MAX_RETRIES + 1


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_errors_never_retry(agnes, status):
    agnes.handle(lambda request: httpx.Response(status, json={"error": "permanent"}))
    with pytest.raises(AgnesHTTPError) as err:
        run(complete("chat", _messages()))
    assert err.value.status_code == status
    assert err.value.retryable is False
    assert agnes.call_count == 1, f"HTTP {status} é permanente e não pode ter retry"


def test_error_messages_never_leak_credentials(agnes):
    agnes.handle(lambda request: httpx.Response(401, json={}))
    with pytest.raises(AgnesHTTPError) as err:
        run(complete("chat", _messages()))
    message = str(err.value)
    assert config.AGNES_API_KEY not in message
    assert "Bearer" not in message
    assert "ping" not in message, "Erro não deve conter conteúdo de mensagens"


def test_retryable_status_set_matches_agnes_docs():
    assert llm.RETRYABLE_STATUS_CODES == frozenset({408, 429, 500, 502, 503, 504, 520, 522, 524})
