from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import EMBEDDING_MODEL, GOOGLE_AI_API_KEY, GOOGLE_AI_MODEL
from app.providers import qdrant, _providers, google_ai_client

router = APIRouter(tags=["Sistema"])


@router.get("/health")
async def health():
    checks: dict[str, object] = {"api": "ok"}

    try:
        collections = qdrant.get_collections().collections
        checks["qdrant"] = "ok"
        checks["collections"] = [c.name for c in collections]
    except Exception:
        checks["qdrant"] = "offline"

    llm_status = {}
    for provider in _providers:
        try:
            kwargs = {
                "model": provider.model,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0,
                provider.token_param: 1,
            }
            provider.client.chat.completions.create(**kwargs)
            llm_status[provider.name] = "ok"
        except Exception:
            llm_status[provider.name] = "unreachable"

    if google_ai_client:
        try:
            from google.genai import types as genai_types
            google_ai_client.models.generate_content(
                model=GOOGLE_AI_MODEL,
                contents="ping",
                config=genai_types.GenerateContentConfig(max_output_tokens=1),
            )
            llm_status["google_ai"] = "ok"
        except Exception:
            llm_status["google_ai"] = "unreachable"
    else:
        llm_status["google_ai"] = "not_configured"

    checks["llm_providers"] = llm_status
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
