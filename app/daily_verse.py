import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from qdrant_client.http import models as qmodels

from app.config import (
    COL_CONVERSATIONS, DAILY_VERSE_PROMPT,
    DAILY_VERSE_TIMEZONE, DAILY_VERSE_DELAY_SECONDS, logger,
)
from app.providers import qdrant, llm_create
from app.profile import fetch_user_profile
from app.firebase import list_all_users, update_daily_verse
from app.toon import build_profile_toon, build_conversation_summary_toon

BRT = ZoneInfo(DAILY_VERSE_TIMEZONE)


def _today_str() -> str:
    return datetime.now(BRT).strftime("%Y-%m-%d")


def _should_update(user_data: dict) -> bool:
    return user_data.get("dailyVerseDate") != _today_str()


def _fetch_recent_summaries(user_id: str, limit: int = 3) -> list[str]:
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
                    qmodels.FieldCondition(key="processed", match=qmodels.MatchValue(value=True)),
                ],
            ),
            limit=50,
            with_payload=True,
            with_vectors=False,
        )

        sessions = []
        for point in results:
            payload = point.payload or {}
            summary = payload.get("summary", "")
            last_activity = payload.get("last_activity", "")
            if summary:
                sessions.append((last_activity, summary))

        sessions.sort(key=lambda s: s[0], reverse=True)
        return [s[1] for s in sessions[:limit]]
    except Exception as e:
        logger.warning("daily-verse: falha ao buscar resumos de %s: %s", user_id, e)
        return []


def _generate_verse(user_id: str) -> str | None:
    profile = fetch_user_profile(user_id)
    summaries = _fetch_recent_summaries(user_id)

    parts = []
    if profile:
        parts.append(build_profile_toon(profile))
    else:
        parts.append("Perfil: novo usuário, sem dados anteriores.")

    if summaries:
        parts.append("Conversas recentes:")
        for i, s in enumerate(summaries, 1):
            toon = build_conversation_summary_toon(s)
            if toon:
                parts.append(f"  {i}. {toon}")
    else:
        parts.append("Sem conversas recentes registradas.")

    user_context = "\n".join(parts)

    try:
        content, label = llm_create(
            messages=[
                {"role": "system", "content": DAILY_VERSE_PROMPT},
                {"role": "user", "content": user_context},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        verse = content.strip()
        if verse:
            logger.debug("daily-verse: %s gerado via %s", user_id, label)
            return verse
        return None
    except Exception as e:
        logger.warning("daily-verse: falha ao gerar verso para %s: %s", user_id, e)
        return None


def process_single_user(user_id: str, user_data: dict | None = None, force: bool = False) -> bool:
    try:
        if not force and user_data and not _should_update(user_data):
            return False

        verse = _generate_verse(user_id)
        if not verse:
            return False

        today = _today_str()
        if update_daily_verse(user_id, verse, today):
            logger.info("daily-verse: %s atualizado", user_id)
            return True
        return False
    except Exception as e:
        logger.warning("daily-verse: erro ao processar %s: %s", user_id, e)
        return False


def run_daily_verse_for_all() -> tuple[int, int]:
    users = list_all_users()
    if not users:
        logger.info("daily-verse: nenhum usuário encontrado")
        return 0, 0

    total = len(users)
    updated = 0
    logger.info("daily-verse: iniciando para %d usuários", total)

    for i, user_data in enumerate(users, 1):
        uid = user_data.get("uid", "")
        if not uid:
            continue

        if not _should_update(user_data):
            logger.debug("daily-verse: %s já atualizado hoje — pulando", uid)
            continue

        if process_single_user(uid, user_data):
            updated += 1

        if i < total:
            import time
            time.sleep(DAILY_VERSE_DELAY_SECONDS)

    logger.info("daily-verse: %d/%d atualizados", updated, total)
    return updated, total


def _seconds_until_midnight_brt() -> float:
    now = datetime.now(BRT)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = (midnight - now).total_seconds()
    return max(delta, 60)


async def start_daily_verse_job():
    wait = _seconds_until_midnight_brt()
    logger.info("daily-verse: ativo — próxima execução em %.0fmin", wait / 60)

    while True:
        await asyncio.sleep(wait)
        try:
            updated, total = run_daily_verse_for_all()
            if updated > 0:
                logger.info("daily-verse: ciclo completo — %d/%d", updated, total)
        except Exception as e:
            logger.error("daily-verse: erro no ciclo — %s", e)

        wait = _seconds_until_midnight_brt()
        logger.info("daily-verse: próxima execução em %.0fmin", wait / 60)
