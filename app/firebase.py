import json
import os
from datetime import datetime

import firebase_admin
import firebase_admin.auth
from firebase_admin import credentials, firestore

from app.config import (
    FIREBASE_SERVICE_ACCOUNT_JSON,
    FIREBASE_SERVICE_ACCOUNT_PATH,
    logger,
)

_firebase_app: firebase_admin.App | None = None
_firestore_db = None


def initialize_firebase() -> bool:
    global _firebase_app, _firestore_db

    if _firebase_app:
        return True

    try:
        if FIREBASE_SERVICE_ACCOUNT_JSON:
            service_account_data = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
            cred = credentials.Certificate(service_account_data)
        elif os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
        else:
            return False

        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_db = firestore.client()
        return True
    except json.JSONDecodeError as e:
        logger.critical("firebase JSON inválido: %s", e)
        return False
    except Exception as e:
        logger.critical("firebase init falhou: %s", e)
        return False


def get_firestore_db():
    return _firestore_db


def fetch_firestore_user(user_id: str) -> dict | None:
    db = get_firestore_db()
    if not db:
        return None

    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:  # type: ignore[union-attr]
            data: dict = doc.to_dict()  # type: ignore[union-attr,assignment]
            data["uid"] = doc.id  # type: ignore[union-attr]
            return data
        return None
    except Exception as e:
        logger.warning("Falha ao buscar usuário Firebase %s: %s", user_id, e)
        return None


def check_firebase_connection() -> bool:
    db = get_firestore_db()
    if not db:
        return False
    try:
        db.collection("users").limit(1).get()
        return True
    except Exception:
        return False


def list_all_users() -> list[dict]:
    db = get_firestore_db()
    if not db:
        return []
    try:
        docs = db.collection("users").stream()
        users = []
        for doc in docs:
            data = doc.to_dict() or {}
            data["uid"] = doc.id
            users.append(data)
        return users
    except Exception as e:
        logger.warning("Falha ao listar usuários Firebase: %s", e)
        return []


def update_daily_verse(user_id: str, verse: str, date_str: str) -> bool:
    db = get_firestore_db()
    if not db:
        return False
    try:
        db.collection("users").document(user_id).update({
            "dailyVerse": verse,
            "dailyVerseDate": date_str,
        })
        return True
    except Exception as e:
        logger.warning("Falha ao atualizar dailyVerse do usuário %s: %s", user_id, e)
        return False


def get_user_quota(uid: str) -> dict | None:
    db = get_firestore_db()
    if not db:
        return None

    try:
        doc = db.collection("users").document(uid).collection("quota").document("daily").get()
        if doc.exists:
            data = doc.to_dict()
            return data if isinstance(data, dict) else None
        return None
    except Exception as e:
        logger.warning("Falha ao buscar quota do usuário %s: %s", uid, e)
        return None


def set_user_quota(uid: str, date: str, used: int) -> bool:
    db = get_firestore_db()
    if not db:
        return False

    try:
        db.collection("users").document(uid).collection("quota").document("daily").set({
            "date": date,
            "used": used,
        })
        return True
    except Exception as e:
        logger.warning("Falha ao definir quota do usuário %s: %s", uid, e)
        return False


def increment_user_quota(uid: str, date: str) -> bool:
    """Atomically increment daily quota for a user. Returns True on success."""
    db = get_firestore_db()
    if not db:
        return False

    try:
        from google.cloud.firestore_v1 import transactional
        quota_ref = db.collection("users").document(uid).collection("quota").document("daily")

        @transactional
        def _incr(transaction):
            snapshot = quota_ref.get(transaction=transaction)
            used = 0
            if snapshot.exists:
                data = snapshot.to_dict() or {}
                if data.get("date") == date:
                    used = int(data.get("used", 0))
            transaction.set(quota_ref, {"date": date, "used": used + 1})

        _incr(db.transaction())
        return True
    except Exception as e:
        logger.warning("Falha ao incrementar quota do usuário %s: %s", uid, e)
        return False


# Subcollections hanging off users/{uid}. Kept explicit so a purge never
# depends on listing collections at runtime.
USER_SUBCOLLECTIONS = ("settings", "tracker", "summaries", "chat_sessions", "quota")


def _delete_collection(collection_ref, batch_size: int = 200) -> int:
    """Delete every document of a collection, including nested subcollections."""
    deleted = 0
    while True:
        docs = list(collection_ref.limit(batch_size).stream())
        if not docs:
            return deleted
        for doc in docs:
            for sub in doc.reference.collections():
                deleted += _delete_collection(sub, batch_size)
            doc.reference.delete()
            deleted += 1
        if len(docs) < batch_size:
            return deleted


def delete_user_data(uid: str) -> tuple[bool, int]:
    """Delete users/{uid} and every subcollection below it.

    Returns (ok, documents_deleted). ok is False when Firestore is
    unavailable or the purge failed, so the caller can refuse to report a
    successful deletion.
    """
    db = get_firestore_db()
    if not db:
        logger.error("Exclusão de conta: Firestore indisponível")
        return False, 0

    user_ref = db.collection("users").document(uid)
    deleted = 0
    try:
        for name in USER_SUBCOLLECTIONS:
            deleted += _delete_collection(user_ref.collection(name))

        # Catch subcollections created outside the documented schema.
        for sub in user_ref.collections():
            deleted += _delete_collection(sub)

        user_ref.delete()
        deleted += 1
        return True, deleted
    except Exception as e:
        logger.error("Exclusão de conta: falha ao apagar Firestore — %s", e.__class__.__name__)
        return False, deleted


def delete_auth_user(uid: str) -> bool:
    """Delete the Firebase Auth account. Missing user counts as success."""
    try:
        firebase_admin.auth.delete_user(uid)
        return True
    except firebase_admin.auth.UserNotFoundError:
        return True
    except Exception as e:
        logger.error("Exclusão de conta: falha ao apagar Auth — %s", e.__class__.__name__)
        return False


def _coerce_summary_date(value: object):
    """Normalize a summary date into a value Firestore stores as a Timestamp.

    Callers pass ISO strings, which Firestore keeps as plain strings. The app
    reads `date` as a Timestamp, and `orderBy('date')` sorts strings in a
    separate type bucket from real timestamps — so a string here both breaks
    parsing and scrambles the archive order. Anything unparseable falls back
    to the server clock so a record is never stored without an ordering key.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return firestore.SERVER_TIMESTAMP


def save_summary_to_firestore(uid: str, summary: dict) -> str | None:
    db = get_firestore_db()
    if not db:
        return None

    try:
        summaries = db.collection("users").document(uid).collection("summaries")
        session_id = summary.get("sessionId")
        payload = {
            "title": summary.get("title", ""),
            "snippet": summary.get("snippet", ""),
            "tags": summary.get("tags", []),
            "date": _coerce_summary_date(summary.get("date")),
            "tool": summary.get("tool", ""),
        }

        if session_id:
            payload["sessionId"] = session_id

        # Enriched akashic fields (optional — backward compatible)
        if summary.get("mood"):
            payload["mood"] = summary["mood"]
        if summary.get("emotionalIntensity") is not None:
            payload["emotionalIntensity"] = summary["emotionalIntensity"]
        if summary.get("keyInsight"):
            payload["keyInsight"] = summary["keyInsight"]
        if summary.get("turnCount") is not None:
            payload["turnCount"] = summary["turnCount"]

        if session_id:
            existing_docs = list(summaries.where("sessionId", "==", session_id).limit(1).stream())
            if existing_docs:
                doc = existing_docs[0]
                summaries.document(doc.id).set(payload, merge=True)
                return doc.id

        payload["createdAt"] = firestore.SERVER_TIMESTAMP
        _, doc_ref = summaries.add(payload)
        return doc_ref.id
    except Exception as e:
        logger.warning("Falha ao salvar summary do usuário %s: %s", uid, e)
        return None
