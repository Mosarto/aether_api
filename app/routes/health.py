from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import EMBEDDING_MODEL, OPENROUTER_API_KEY
from app.providers import qdrant, create_chat_completion, create_background_completion

router = APIRouter(tags=["Sistema"])


def _probe_openrouter(create_completion) -> str:
    if not OPENROUTER_API_KEY:
        return "not_configured"

    try:
        create_completion(
            [{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=1,
        )
        return "ok"
    except Exception:
        return "unreachable"


@router.get("/health")
async def health():
    checks: dict[str, object] = {"api": "ok"}

    try:
        collections = qdrant.get_collections().collections
        checks["qdrant"] = "ok"
        checks["collections"] = [c.name for c in collections]
    except Exception:
        checks["qdrant"] = "offline"

    llm_status = {
        "openrouter_chat": _probe_openrouter(create_chat_completion),
        "openrouter_background": _probe_openrouter(create_background_completion),
    }

    checks["llm"] = llm_status
    checks["embedding"] = EMBEDDING_MODEL

    try:
        from app.firebase import check_firebase_connection, get_firestore_db
        if get_firestore_db():
            checks["firebase"] = "ok" if check_firebase_connection() else "unreachable"
        else:
            checks["firebase"] = "not_configured"
    except Exception:
        checks["firebase"] = "error"

    any_llm_ok = any(v == "ok" for v in llm_status.values())
    healthy = checks.get("qdrant") == "ok" and any_llm_ok
    checks["status"] = "ok" if healthy else "degraded"

    return JSONResponse(content=checks, status_code=200 if healthy else 503)
