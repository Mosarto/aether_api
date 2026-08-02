import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import (
    AGNES_STARTUP_PROBE,
    COL_REFLECTIONS,
    EMBEDDING_MODEL,
    QDRANT_URL,
    logger,
)
from app.llm import close_llm_client, complete, init_llm_client, missing_agnes_config
from app.providers import qdrant

API_VERSION = "0.9.0"


def _check_env_vars():
    """Validate required configuration without ever printing values."""
    missing = missing_agnes_config()
    if missing:
        logger.critical("Configuração Agnes ausente: %s", ", ".join(missing))
        sys.exit(1)

    if os.environ.get("ALLOWED_ORIGINS", "*") == "*":
        logger.warning("ALLOWED_ORIGINS='*'")


def _check_firebase():
    from app.firebase import check_firebase_connection, initialize_firebase

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
            qdrant.get_collections()
            return True
        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.critical("qdrant inacessível em %s — %s", QDRANT_URL, e)
                sys.exit(1)


async def _probe_agnes() -> bool:
    """Optional real completion probe (AGNES_STARTUP_PROBE=1). Costs tokens.

    Never logs prompt, response or credentials — only the outcome.
    """
    try:
        await complete(
            "session_title",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        return True
    except Exception as e:
        logger.warning("agnes probe ✗ (%s)", e.__class__.__name__)
        return False


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
    _check_embedding_model()

    init_llm_client()

    services = ["qdrant", "agnes:config"] + (["firebase"] if firebase_ok else []) + ["embedding"]
    if AGNES_STARTUP_PROBE and await _probe_agnes():
        services.append("agnes:probe")
    logger.info("%s", "  ".join(f"{s} ✓ " for s in services))

    from app.profile import ensure_profiles_collection
    ensure_profiles_collection()

    from app.background import start_profile_job
    profile_task = asyncio.create_task(start_profile_job())

    from app.daily_verse import start_daily_verse_job
    verse_task = asyncio.create_task(start_daily_verse_job())

    logger.info("✅ ready")

    try:
        yield
    finally:
        for task in (profile_task, verse_task):
            task.cancel()
        await close_llm_client()
