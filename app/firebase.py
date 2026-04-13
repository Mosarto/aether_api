import os
import json

import firebase_admin
from firebase_admin import credentials, firestore

from app.config import FIREBASE_SERVICE_ACCOUNT_PATH, FIREBASE_SERVICE_ACCOUNT_JSON, logger

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


def save_summary_to_firestore(uid: str, summary: dict) -> str | None:
    db = get_firestore_db()
    if not db:
        return None

    try:
        payload = {
            "title": summary.get("title", ""),
            "snippet": summary.get("snippet", ""),
            "tags": summary.get("tags", []),
            "date": summary.get("date", ""),
            "tool": summary.get("tool", ""),
            "createdAt": firestore.SERVER_TIMESTAMP,
        }

        # Enriched akashic fields (optional — backward compatible)
        if summary.get("mood"):
            payload["mood"] = summary["mood"]
        if summary.get("emotionalIntensity") is not None:
            payload["emotionalIntensity"] = summary["emotionalIntensity"]
        if summary.get("keyInsight"):
            payload["keyInsight"] = summary["keyInsight"]
        if summary.get("turnCount") is not None:
            payload["turnCount"] = summary["turnCount"]

        _, doc_ref = db.collection("users").document(uid).collection("summaries").add(payload)
        return doc_ref.id
    except Exception as e:
        logger.warning("Falha ao salvar summary do usuário %s: %s", uid, e)
        return None
