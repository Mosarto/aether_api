import asyncio
from datetime import datetime, timezone, timedelta

from qdrant_client.http import models as qmodels

from app.config import (
    COL_CONVERSATIONS, SESSION_TTL_HOURS,
    PROFILE_JOB_INTERVAL_MINUTES, COMPRESSION_MIN_TURNS, logger,
)
from app.providers import qdrant
from app.profile import (
    ensure_profiles_collection, fetch_user_profile,
    upsert_user_profile, compress_history, extract_profile_updates,
    extract_akashic_metadata,
)
from app.firebase import save_summary_to_firestore


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
                _mark_session_processed(point_id, "")
                continue

            summary = compress_history(turns)
            if not summary:
                _mark_session_processed(point_id, "")
                continue

            # Save akashic record to Firestore (only for meaningful sessions)
            if len(turns) >= COMPRESSION_MIN_TURNS:
                akashic_meta = extract_akashic_metadata(summary, len(turns))
                akashic_payload = {
                    "title": "",
                    "snippet": summary,
                    "tags": [],
                    "date": datetime.now(timezone.utc).isoformat(),
                    "tool": "session_summary",
                    **akashic_meta,
                }
                save_summary_to_firestore(user_id, akashic_payload)

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
