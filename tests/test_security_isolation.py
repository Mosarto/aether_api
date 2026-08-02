"""Session ownership isolation: reading, continuing and deleting sessions
of another user must fail closed (404, never data)."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.models import ChatRequest
from app.routes import chat as chat_route
from app.routes import conversations as conv_route


def run(coro):
    return asyncio.run(coro)


SESSION_ID = "11111111-1111-1111-1111-111111111111"
OWNER = {"uid": "user-a", "subscription_tier": "free", "is_anonymous": False}
ATTACKER = {"uid": "user-b", "subscription_tier": "free", "is_anonymous": False}


def _meta_point(user_id: str):
    return SimpleNamespace(
        id="meta-point-id",
        payload={
            "session_id": SESSION_ID,
            "user_id": user_id,
            "title": "Sessão privada",
            "is_session_meta": True,
        },
    )


def _turn_point(user_id: str, content: str = "segredo"):
    return SimpleNamespace(
        id="turn-point-id",
        payload={
            "session_id": SESSION_ID,
            "user_id": user_id,
            "role": "user",
            "content": content,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "is_session_meta": False,
        },
    )


@pytest.fixture
def qdrant_with_foreign_session(monkeypatch):
    """qdrant.scroll honoring the user_id filter like the real backend."""

    def fake_scroll(collection_name, scroll_filter=None, **kwargs):
        conditions = {c.key: c.match.value for c in (scroll_filter.must or [])}
        is_meta = conditions.get("is_session_meta")
        session = conditions.get("session_id")
        user = conditions.get("user_id")

        if session != SESSION_ID:
            return [], None
        if is_meta is True:
            return [_meta_point(OWNER["uid"])], None
        # Turn queries must carry the user filter; a foreign user gets nothing.
        if user == OWNER["uid"]:
            return [_turn_point(OWNER["uid"])], None
        return [], None

    monkeypatch.setattr(conv_route.qdrant, "scroll", fake_scroll)
    return fake_scroll


def test_get_session_of_other_user_returns_404(qdrant_with_foreign_session):
    with pytest.raises(HTTPException) as err:
        run(conv_route.get_session(SESSION_ID, user=ATTACKER))
    assert err.value.status_code == 404


def test_get_session_owner_still_works(qdrant_with_foreign_session):
    result = run(conv_route.get_session(SESSION_ID, user=OWNER))
    assert result["sessionId"] == SESSION_ID
    assert result["turnCount"] == 1


def test_delete_session_of_other_user_returns_404(qdrant_with_foreign_session, monkeypatch):
    deleted = []
    monkeypatch.setattr(conv_route.qdrant, "delete", lambda **kw: deleted.append(kw))
    monkeypatch.setattr(conv_route.qdrant, "set_payload", lambda **kw: None)

    with pytest.raises(HTTPException) as err:
        run(conv_route.delete_session(SESSION_ID, user=ATTACKER))
    assert err.value.status_code == 404
    assert deleted == [], "Nada pode ser deletado sem posse da sessão"


def test_ownership_check_fails_closed_without_owner_field(monkeypatch):
    """Meta without user_id (or unreadable) must deny access, not allow it."""

    def fake_scroll(collection_name, scroll_filter=None, **kwargs):
        return [SimpleNamespace(id="m", payload={"session_id": SESSION_ID, "is_session_meta": True})], None

    monkeypatch.setattr(conv_route.qdrant, "scroll", fake_scroll)

    with pytest.raises(HTTPException) as err:
        run(conv_route.get_session(SESSION_ID, user=ATTACKER))
    assert err.value.status_code == 404


def test_chat_cannot_continue_foreign_session(monkeypatch):
    async def _noop_async(*a, **k):
        return {"remaining": 5}

    monkeypatch.setattr(chat_route, "check_rate_limit", _noop_async)
    monkeypatch.setattr(chat_route, "check_quota", _noop_async)
    monkeypatch.setattr(chat_route, "_ensure_collection", lambda: None)
    monkeypatch.setattr(chat_route, "_get_session_meta", lambda sid: _meta_point(OWNER["uid"]).payload)

    llm_calls = []

    async def _fake_complete(*a, **k):
        llm_calls.append(a)
        raise AssertionError("LLM não pode ser chamado para sessão alheia")

    monkeypatch.setattr(chat_route, "complete", _fake_complete)

    req = ChatRequest(message="continua aí", sessionId=SESSION_ID)
    with pytest.raises(HTTPException) as err:
        run(chat_route.chat(req, user=ATTACKER))

    assert err.value.status_code == 404
    assert llm_calls == [], "Sequestro de sessão bloqueado antes de gastar tokens"


def test_chat_refunds_quota_when_agnes_fails(monkeypatch):
    from app.llm import AgnesTimeoutError

    refunds = []

    async def _rate(*a, **k):
        return None

    async def _quota(user):
        return {"remaining": 4}

    async def _refund(user):
        refunds.append(user["uid"])

    async def _fail_complete(*a, **k):
        raise AgnesTimeoutError("Agnes timeout após 45s (chat)")

    async def _compress(*a, **k):
        return ""

    monkeypatch.setattr(chat_route, "check_rate_limit", _rate)
    monkeypatch.setattr(chat_route, "check_quota", _quota)
    monkeypatch.setattr(chat_route, "refund_quota", _refund)
    monkeypatch.setattr(chat_route, "_ensure_collection", lambda: None)
    monkeypatch.setattr(chat_route, "retrieve_context", lambda *a, **k: ([], []))
    monkeypatch.setattr(chat_route, "ensure_profiles_collection", lambda: None)
    monkeypatch.setattr(chat_route, "fetch_user_profile", lambda uid: {"personality_summary": "x"})
    monkeypatch.setattr(chat_route, "fetch_firestore_user", lambda uid: None)
    monkeypatch.setattr(chat_route, "compress_history", _compress)
    monkeypatch.setattr(chat_route, "complete", _fail_complete)

    req = ChatRequest(message="mensagem real")
    with pytest.raises(HTTPException) as err:
        run(chat_route.chat(req, user=dict(OWNER)))

    assert err.value.status_code == 503
    assert err.value.detail == {"error": "llm_unavailable"}
    assert refunds == [OWNER["uid"]], "Falha da Agnes deve devolver a quota reservada"


def test_chat_success_does_not_refund(monkeypatch):
    refunds = []
    saved_turns = []

    async def _rate(*a, **k):
        return None

    async def _quota(user):
        return {"remaining": 4}

    async def _refund(user):
        refunds.append(user["uid"])

    async def _ok_complete(use_case, messages, **k):
        return SimpleNamespace(content="resposta da Nyx", model="agnes-2.5-flash")

    async def _title(*a, **k):
        return "Título curto"

    monkeypatch.setattr(chat_route, "check_rate_limit", _rate)
    monkeypatch.setattr(chat_route, "check_quota", _quota)
    monkeypatch.setattr(chat_route, "refund_quota", _refund)
    monkeypatch.setattr(chat_route, "_ensure_collection", lambda: None)
    monkeypatch.setattr(chat_route, "retrieve_context", lambda *a, **k: ([], []))
    monkeypatch.setattr(chat_route, "ensure_profiles_collection", lambda: None)
    monkeypatch.setattr(chat_route, "fetch_user_profile", lambda uid: {"personality_summary": "x"})
    monkeypatch.setattr(chat_route, "fetch_firestore_user", lambda uid: None)
    monkeypatch.setattr(chat_route, "complete", _ok_complete)
    monkeypatch.setattr(chat_route, "_generate_session_title", _title)
    monkeypatch.setattr(chat_route, "_save_turn", lambda *a, **k: saved_turns.append(a))
    monkeypatch.setattr(chat_route, "_upsert_session_meta", lambda *a, **k: None)

    req = ChatRequest(message="mensagem real")
    resp = run(chat_route.chat(req, user=dict(OWNER)))

    assert resp.response == "resposta da Nyx"
    assert resp.model == "agnes-2.5-flash"
    assert resp.remaining == 4
    assert refunds == []
    assert len(saved_turns) == 2
