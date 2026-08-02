"""Account deletion must erase every store keyed by uid, scope itself to the
caller, stay idempotent, and never report success on a partial purge."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import account
from app.account import PERSONAL_COLLECTIONS, purge_user_data
from app.routes import user_profile as profile_route

USER = {"uid": "user-a", "subscription_tier": "free", "is_anonymous": False}


def run(coro):
    return asyncio.run(coro)


class FakeQdrant:
    """Records delete filters and reports what is left afterwards."""

    def __init__(self, remaining: dict[str, int] | None = None, fail_on: str | None = None):
        self.deletes: list[tuple[str, str]] = []
        self._remaining = remaining or {}
        self._fail_on = fail_on

    def delete(self, collection_name, points_selector, **kwargs):
        if collection_name == self._fail_on:
            raise RuntimeError("qdrant down")
        conditions = {
            c.key: c.match.value for c in (points_selector.filter.must or [])
        }
        self.deletes.append((collection_name, conditions.get("user_id")))

    def count(self, collection_name, count_filter, **kwargs):
        return SimpleNamespace(count=self._remaining.get(collection_name, 0))


@pytest.fixture
def fake_firebase(monkeypatch):
    calls = {"firestore": [], "auth": []}

    def _delete_data(uid):
        calls["firestore"].append(uid)
        return True, 7

    def _delete_auth(uid):
        calls["auth"].append(uid)
        return True

    monkeypatch.setattr(account, "delete_user_data", _delete_data)
    monkeypatch.setattr(account, "delete_auth_user", _delete_auth)
    return calls


def test_purge_covers_every_personal_store(monkeypatch, fake_firebase):
    fake = FakeQdrant()
    monkeypatch.setattr(account, "qdrant", fake)

    report = purge_user_data("user-a")

    assert report["complete"] is True
    deleted_collections = [c for c, _ in fake.deletes]
    assert set(deleted_collections) == set(PERSONAL_COLLECTIONS)
    assert {"conversations", "user_memories", "user_profiles"} <= set(deleted_collections)
    assert fake_firebase["firestore"] == ["user-a"]
    assert fake_firebase["auth"] == ["user-a"]


def test_purge_is_scoped_to_the_requesting_user(monkeypatch, fake_firebase):
    fake = FakeQdrant()
    monkeypatch.setattr(account, "qdrant", fake)

    purge_user_data("user-a")

    assert all(uid == "user-a" for _, uid in fake.deletes), \
        "Filtro de exclusão deve conter apenas o uid do próprio usuário"


def test_purge_leaves_shared_catalog_untouched(monkeypatch, fake_firebase):
    fake = FakeQdrant()
    monkeypatch.setattr(account, "qdrant", fake)

    purge_user_data("user-a")

    assert "reflections" not in [c for c, _ in fake.deletes], \
        "Catálogo compartilhado é conteúdo de produto, não dado pessoal"


def test_purge_is_idempotent(monkeypatch, fake_firebase):
    fake = FakeQdrant()
    monkeypatch.setattr(account, "qdrant", fake)

    first = purge_user_data("user-a")
    second = purge_user_data("user-a")

    assert first["complete"] is True
    assert second["complete"] is True, "Repetir a exclusão não pode falhar"


def test_missing_collection_is_not_a_failure(monkeypatch, fake_firebase):
    class Missing404(FakeQdrant):
        def delete(self, collection_name, points_selector, **kwargs):
            if collection_name == "user_memories":
                raise type("NotFound", (Exception,), {"status_code": 404})()
            super().delete(collection_name, points_selector, **kwargs)

    monkeypatch.setattr(account, "qdrant", Missing404())

    assert purge_user_data("user-a")["complete"] is True


def test_qdrant_failure_marks_purge_incomplete(monkeypatch, fake_firebase):
    monkeypatch.setattr(account, "qdrant", FakeQdrant(fail_on="conversations"))

    report = purge_user_data("user-a")

    assert report["complete"] is False
    assert report["stores"]["conversations"] == "failed"


def test_leftover_points_mark_purge_incomplete(monkeypatch, fake_firebase):
    monkeypatch.setattr(account, "qdrant", FakeQdrant(remaining={"user_profiles": 2}))

    report = purge_user_data("user-a")

    assert report["complete"] is False, "Pontos remanescentes não podem passar por sucesso"


def test_firestore_failure_marks_purge_incomplete(monkeypatch):
    monkeypatch.setattr(account, "qdrant", FakeQdrant())
    monkeypatch.setattr(account, "delete_user_data", lambda uid: (False, 0))
    monkeypatch.setattr(account, "delete_auth_user", lambda uid: True)

    assert purge_user_data("user-a")["complete"] is False


def test_auth_is_deleted_after_the_data(monkeypatch):
    order = []
    monkeypatch.setattr(account, "qdrant", FakeQdrant())
    monkeypatch.setattr(account, "delete_user_data", lambda uid: (order.append("firestore"), (True, 1))[1])
    monkeypatch.setattr(account, "delete_auth_user", lambda uid: (order.append("auth"), True)[1])

    purge_user_data("user-a")

    assert order == ["firestore", "auth"], \
        "Auth por último: enquanto a conta existe o usuário pode repetir a purga"


def test_report_contains_no_personal_data(monkeypatch, fake_firebase):
    monkeypatch.setattr(account, "qdrant", FakeQdrant())

    report = purge_user_data("user-a")

    assert "user-a" not in str(report), "Relatório não pode carregar identificadores"


# --- Route behavior ----------------------------------------------------------


def test_route_uses_token_uid_only(monkeypatch):
    seen = []

    async def _rate(*a, **k):
        return None

    monkeypatch.setattr(profile_route, "check_rate_limit", _rate)
    monkeypatch.setattr(
        profile_route, "purge_user_data",
        lambda uid: (seen.append(uid), {"complete": True, "stores": {}})[1],
    )

    response = run(profile_route.delete_user_account(user=dict(USER)))

    assert response.status_code == 204
    assert seen == ["user-a"], "Rota só pode apagar o dono do token"


def test_route_reports_incomplete_purge_as_error(monkeypatch):
    async def _rate(*a, **k):
        return None

    monkeypatch.setattr(profile_route, "check_rate_limit", _rate)
    monkeypatch.setattr(
        profile_route, "purge_user_data",
        lambda uid: {"complete": False, "stores": {"firestore": "failed"}},
    )

    with pytest.raises(HTTPException) as err:
        run(profile_route.delete_user_account(user=dict(USER)))

    assert err.value.status_code == 500
    assert err.value.detail == {"error": "account_deletion_incomplete"}
