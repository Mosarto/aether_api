import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import (
    logger, QDRANT_URL,
    EMBEDDING_MODEL, COL_REFLECTIONS, COL_USER_MEMORIES, COL_CONVERSATIONS,
    COL_USER_PROFILES, OPENROUTER_API_KEY,
)
from app.providers import qdrant, create_chat_completion, create_background_completion

API_VERSION = "0.8.0"


def _check_env_vars():
    if not OPENROUTER_API_KEY:
        logger.critical("OPENROUTER_API_KEY ausente")
        sys.exit(1)

    if os.environ.get("ALLOWED_ORIGINS", "*") == "*":
        logger.warning("ALLOWED_ORIGINS='*'")


def _check_firebase():
    from app.firebase import initialize_firebase, check_firebase_connection

    if not initialize_firebase():
        logger.warning("firebase ✗")
        return False

    if not check_firebase_connection():
        logger.warning("firestore ✗")
        return False

    return True


def _check_qdrant(max_retries: int = 5, delay: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            collections = qdrant.get_collections().collections
            return True
        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.critical("qdrant inacessível em %s — %s", QDRANT_URL, e)
                sys.exit(1)


def _check_openrouter() -> list[str]:
    ok = []
    probes = (
        ("openrouter_chat", create_chat_completion),
        ("openrouter_background", create_background_completion),
    )
    for label, create_completion in probes:
        try:
            create_completion(
                [{"role": "user", "content": "ping"}],
                temperature=0,
                max_tokens=1,
            )
            ok.append(label)
        except Exception:
            logger.warning("%s ✗", label)

    return ok


def _check_embedding_model():
    try:
        qdrant.query(collection_name=COL_REFLECTIONS, query_text="teste", limit=1)
    except Exception:
        try:
            from fastembed import TextEmbedding
            TextEmbedding(EMBEDDING_MODEL)
        except Exception as e:
            logger.critical("embedding falhou — %s", e)
            sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Aether v%s", API_VERSION)

    _check_env_vars()
    _check_qdrant()
    firebase_ok = _check_firebase()
    llm_ok = _check_openrouter()
    _check_embedding_model()

    services = ["qdrant"] + llm_ok + (["firebase"] if firebase_ok else []) + ["embedding"]
    logger.info("%s", "  ".join(f"{s} ✓ " for s in services))

    from app.test_battery import run_battery
    report = run_battery()
    if not report.all_passed:
        logger.critical("tests failed (%d errors) — aborting", report.failed)
        sys.exit(1)

    from app.profile import ensure_profiles_collection
    ensure_profiles_collection()

    from app.background import start_profile_job
    asyncio.create_task(start_profile_job())

    from app.daily_verse import start_daily_verse_job
    asyncio.create_task(start_daily_verse_job())

    logger.info("✅ ready")

    yield
