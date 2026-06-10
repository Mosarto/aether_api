import asyncio
from datetime import datetime, timezone, timedelta

from qdrant_client.http import models as qmodels

from app.config import (
    COL_CONVERSATIONS, SESSION_TTL_HOURS,
    PROFILE_JOB_INTERVAL_MINUTES, logger,
)
from app.providers import qdrant
from app.profile import (
    ensure_profiles_collection, fetch_user_profile,
    upsert_user_profile, compress_history, extract_profile_updates,
    extract_akashic_metadata,
)
from app.firebase import save_summary_to_firestore

_TRIVIAL_GREETING_TOKENS = {
    "oi", "olá", "ola", "eai", "e aí", "e ai", "fala", "hey",
    "salve", "bom dia", "boa tarde", "boa noite", "hello", "hi",
    "opa", "yo", "fala aí", "fala ai", "coé", "coe",
}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().rstrip("!?.,").split())


def _should_create_akashic(turns: list[dict], summary: str) -> bool:
    if len(turns) < 2 or not summary.strip():
        return False

    user_turns = [str(t.get("content", "")).strip() for t in turns if t.get("role") == "user"]
    assistant_turns = [
        str(t.get("content", "")).strip()
        for t in turns
        if t.get("role") in {"assistant", "ai", "model"}
    ]

    if not user_turns or not assistant_turns:
        return False

    user_text = " ".join(t for t in user_turns if t)
    assistant_text = " ".join(t for t in assistant_turns if t)
    normalized_user = _normalize_text(user_text)

    if len(user_turns) == 1 and normalized_user in _TRIVIAL_GREETING_TOKENS:
        return False

    if len(user_text) < 10 and len(assistant_text) < 80:
        return False

    if len(summary.strip()) < 40 and len(user_text) < 20:
        return False

    return True


def _find_expired_unprocessed_sessions() -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=SESSION_TTL_HOURS)).isoformat()

    try:
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
                    qmodels.FieldCondition(key="processed", match=qmodels.MatchValue(value=False)),
                ],
            ),
            limit=50,
            with_payload=True,
            with_vectors=False,
        )

        expired = []
        for point in results:
            payload = point.payload or {}
            last_activity = payload.get("last_activity", "")
            if last_activity and last_activity < cutoff:
                expired.append({"point_id": point.id, **payload})

        return expired
    except Exception as e:
        logger.warning("Falha ao buscar sessões expiradas: %s", e)
        return []


def _load_session_turns(session_id: str) -> list[dict]:
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=False)),
                ],
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        turns = [r.payload for r in results if r.payload]
        turns.sort(key=lambda t: t.get("timestamp", ""))
        return turns
    except Exception as e:
        logger.warning("Falha ao carregar turns da sessão %s: %s", session_id, e)
        return []


def _mark_session_processed(point_id: str, summary: str):
    try:
        qdrant.set_payload(
            collection_name=COL_CONVERSATIONS,
            payload={"processed": True, "summary": summary},
            points=[point_id],
        )
    except Exception as e:
        logger.warning("Falha ao marcar sessão como processada (%s): %s", point_id, e)


def process_finalized_sessions():
    expired = _find_expired_unprocessed_sessions()
    if not expired:
        return 0

    logger.info("profile-job: %d sessões expiradas", len(expired))
    processed_count = 0

    for session_meta in expired:
        session_id = session_meta.get("session_id", "")
        user_id = session_meta.get("user_id", "")
        point_id = session_meta.get("point_id", "")

        if not session_id or not user_id:
            continue

        try:
            turns = _load_session_turns(session_id)
            if not turns:
                logger.info("profile-job: sessão %s sem turns, marcando processada", session_id[:8])
                _mark_session_processed(point_id, "")
                continue

            logger.info("profile-job: sessão %s com %d turns", session_id[:8], len(turns))

            summary = compress_history(turns)
            if not summary:
                logger.warning("profile-job: sessão %s sem summary, mantendo pendente para retry", session_id[:8])
                continue

            create_akashic = _should_create_akashic(turns, summary)
            if create_akashic:
                akashic_meta = extract_akashic_metadata(summary, len(turns))
                akashic_payload = {
                    "sessionId": session_id,
                    "title": session_meta.get("title", ""),
                    "snippet": summary,
                    "tags": [akashic_meta["mood"]] if akashic_meta.get("mood") else [],
                    "date": datetime.now(timezone.utc).isoformat(),
                    "tool": "session_summary",
                    **akashic_meta,
                }
                summary_id = save_summary_to_firestore(user_id, akashic_payload)
                if not summary_id:
                    logger.warning(
                        "profile-job: falha ao salvar Akashic da sessão %s, mantendo pendente para retry",
                        session_id[:8],
                    )
                    continue
                logger.info("profile-job: Akashic %s salvo para sessão %s", summary_id[:8], session_id[:8])
            else:
                logger.info(
                    "profile-job: sessão %s sem Akashic (trivial ou curta)",
                    session_id[:8],
                )

            current_profile = fetch_user_profile(user_id)
            updates = extract_profile_updates(current_profile, summary)

            if updates:
                merged = current_profile or {}
                merged.update(updates)
                merged["version"] = (current_profile or {}).get("version", 0) + 1
                merged["conversation_count"] = (current_profile or {}).get("conversation_count", 0) + 1
                upsert_user_profile(user_id, merged)

            _mark_session_processed(point_id, summary)
            processed_count += 1
            logger.debug("profile-job: sessão %s processada", session_id[:8])

        except Exception as e:
            logger.warning("profile-job: erro sessão %s: %s", session_id[:8], e)
            continue

    logger.info("profile-job: %d/%d processadas", processed_count, len(expired))
    return processed_count


async def start_profile_job():
    logger.info("profile-job: ativo (cada %dmin)", PROFILE_JOB_INTERVAL_MINUTES)
    while True:
        await asyncio.sleep(PROFILE_JOB_INTERVAL_MINUTES * 60)
        try:
            ensure_profiles_collection()
            count = process_finalized_sessions()
            if count > 0:
                logger.info("profile-job: %d perfis atualizados", count)
        except Exception as e:
            logger.error("profile-job: erro — %s", e)
