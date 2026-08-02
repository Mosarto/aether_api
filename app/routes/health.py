from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import EMBEDDING_MODEL
from app.llm import missing_agnes_config
from app.providers import qdrant

router = APIRouter(tags=["Sistema"])


def _agnes_status() -> str:
    """Configuration-only readiness — never spends a completion."""
    return "configured" if not missing_agnes_config() else "not_configured"


@router.get("/health")
async def health():
    checks: dict[str, object] = {"api": "ok"}

    try:
        collections = qdrant.get_collections().collections
        checks["qdrant"] = "ok"
        checks["collections"] = [c.name for c in collections]
    except Exception:
        checks["qdrant"] = "offline"

    llm_status = {"agnes": _agnes_status()}

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

    healthy = checks.get("qdrant") == "ok" and llm_status["agnes"] == "configured"
    checks["status"] = "ok" if healthy else "degraded"

    return JSONResponse(content=checks, status_code=200 if healthy else 503)
