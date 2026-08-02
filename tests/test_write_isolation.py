"""Write-path isolation: client-supplied ids must never overwrite another
user's data (user memories) nor another user's catalog entries (reflections)."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.config import deterministic_uuid
from app.models import ReflectionCreate, UserAnswer
from app.routes import answers as answers_route
from app.routes import reflections as reflections_route

USER_A = {"uid": "user-a", "subscription_tier": "free", "is_anonymous": False}
USER_B = {"uid": "user-b", "subscription_tier": "free", "is_anonymous": False}


def run(coro):
    return asyncio.run(coro)


def test_user_answer_point_id_is_scoped_per_user(monkeypatch):
    captured = {}

    def fake_add(collection_name, documents, metadata, ids):
        captured[collection_name] = ids

    def fake_query(**kwargs):
        return []

    monkeypatch.setattr(answers_route.qdrant, "add", fake_add)
    monkeypatch.setattr(answers_route.qdrant, "query", fake_query)

    answer = UserAnswer(id="shared-client-id", reflectionId="r1", content="minha memória")

    run(answers_route.submit_user_answer(answer, user=dict(USER_A)))
    ids_a = captured["user_memories"]
    run(answers_route.submit_user_answer(answer, user=dict(USER_B)))
    ids_b = captured["user_memories"]

    assert ids_a != ids_b, "Mesmo id de cliente não pode colidir entre usuários"
    assert ids_a == [deterministic_uuid("user-a:shared-client-id")]
    assert ids_b == [deterministic_uuid("user-b:shared-client-id")]


def _reflection(rid: str = "catalog-entry-1") -> ReflectionCreate:
    return ReflectionCreate(id=rid, categoryId="faith", title="Título", description="Descrição")


def test_reflection_overwrite_by_other_user_is_rejected(monkeypatch):
    added = []
    existing_point = SimpleNamespace(id="p1", payload={"created_by": USER_A["uid"]})

    monkeypatch.setattr(reflections_route.qdrant, "retrieve", lambda **kw: [existing_point])
    monkeypatch.setattr(reflections_route.qdrant, "add", lambda **kw: added.append(kw))

    with pytest.raises(HTTPException) as err:
        run(reflections_route.create_reflection(_reflection(), user=dict(USER_B)))

    assert err.value.status_code == 409
    assert added == [], "Sobrescrita cross-user não pode chegar ao Qdrant"


def test_reflection_owner_can_reindex_own_entry(monkeypatch):
    added = []
    existing_point = SimpleNamespace(id="p1", payload={"created_by": USER_A["uid"]})

    monkeypatch.setattr(reflections_route.qdrant, "retrieve", lambda **kw: [existing_point])
    monkeypatch.setattr(reflections_route.qdrant, "add", lambda **kw: added.append(kw))

    result = run(reflections_route.create_reflection(_reflection(), user=dict(USER_A)))

    assert result["status"] == "indexed"
    assert len(added) == 1
    assert added[0]["metadata"][0]["created_by"] == USER_A["uid"]


def test_reflection_first_write_records_creator(monkeypatch):
    added = []

    monkeypatch.setattr(reflections_route.qdrant, "retrieve", lambda **kw: [])
    monkeypatch.setattr(reflections_route.qdrant, "add", lambda **kw: added.append(kw))

    run(reflections_route.create_reflection(_reflection("nova"), user=dict(USER_B)))

    assert added[0]["metadata"][0]["created_by"] == USER_B["uid"]
    assert added[0]["ids"] == [deterministic_uuid("nova")]
