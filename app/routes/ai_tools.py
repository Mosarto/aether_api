from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.config import (
    AURA_READING_PROMPT,
    COL_CONVERSATIONS,
    DAY_ANALYSIS_PROMPT,
    DREAM_ANALYSIS_PROMPT,
    STOIC_ADVICE_PROMPT,
    SYNCHRONICITY_PROMPT,
    logger,
)
from app.firebase import save_summary_to_firestore
from app.llm import (
    LLM_UNAVAILABLE_DETAIL,
    AgnesError,
    complete_json,
    was_billed,
    wrap_untrusted,
)
from app.llm_schemas import AIToolResult
from app.models import AIToolRequest, AIToolResponse
from app.profile import fetch_user_profile
from app.providers import qdrant
from app.quota import check_quota, refund_quota
from app.rate_limit import check_rate_limit
from app.toon import build_profile_toon

router = APIRouter(prefix="/ai", tags=["AI Tools"])


def _fetch_recent_session_summaries(uid: str, limit: int = 3) -> list[str]:
    """Fetch recent session titles from Qdrant conversations meta for extra context."""
    try:
        from qdrant_client.http import models as qmodels
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=uid)),
                qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
            ]),
            limit=limit * 2,  # fetch extra to sort by recency
            with_payload=True,
            with_vectors=False,
        )
        metas = sorted(results, key=lambda p: (p.payload or {}).get("last_activity", ""), reverse=True)
        summaries = []
        for m in metas[:limit]:
            payload = m.payload or {}
            title = payload.get("title", "")
            turn_count = payload.get("turn_count", 0)
            if title:
                summaries.append(f"- {title} ({turn_count} turnos)")
        return summaries
    except Exception as e:
        logger.debug("Falha ao buscar sessões recentes para ai_tool: %s", e)
        return []


async def _process_ai_tool(
    user: dict,
    content: str,
    prompt: str,
    tool_name: str,
    include_profile: bool = False,
) -> AIToolResponse:
    """
    Shared processing for all AI tools.
    1. Optionally fetch user profile for context (aura, sync)
    2. Call Agnes through the structured-output helper (single corrective retry)
    3. Refund the reserved quota slot and return 503 when Agnes fails
    4. Save to Firestore and return AIToolResponse
    """
    parts: list[str] = []

    if include_profile:
        try:
            profile_data = fetch_user_profile(user["uid"])
            if profile_data:
                parts.append(wrap_untrusted("perfil_usuario", build_profile_toon(profile_data)))

            recent = _fetch_recent_session_summaries(user["uid"])
            if recent:
                parts.append(wrap_untrusted("conversas_recentes", "\n".join(recent)))
        except Exception as e:
            logger.warning("Falha ao buscar perfil para %s: %s", tool_name, e)

    parts.append(wrap_untrusted("conteudo", content))

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]

    try:
        parsed = await complete_json(tool_name, messages, AIToolResult)
    except AgnesError as e:
        logger.warning("ai-tool %s indisponível: %s", tool_name, e)
        # Only refund when nothing was billed — an invalid completion was paid
        # for, and refunding it would hand out unlimited free calls.
        if not was_billed(e):
            await refund_quota(user)
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE_DETAIL)

    response = AIToolResponse(
        id=str(uuid4()),
        title=parsed.title[:500],
        snippet=parsed.snippet[:8000],
        tags=parsed.tags,
        date=datetime.now(timezone.utc),
        tool=tool_name,
        mood=parsed.mood,
        emotionalIntensity=parsed.emotionalIntensity,
        keyInsight=parsed.keyInsight or None,
    )

    try:
        save_summary_to_firestore(user["uid"], {
            "title": response.title,
            "snippet": response.snippet,
            "tags": response.tags,
            "date": response.date.isoformat(),
            "tool": response.tool,
            "mood": response.mood,
            "emotionalIntensity": response.emotionalIntensity,
            "keyInsight": response.keyInsight,
        })
    except Exception as e:
        logger.warning("Falha ao salvar summary %s: %s", tool_name, e)

    return response


def _require_full_account(user: dict) -> None:
    if user.get("is_anonymous") or user.get("subscription_tier") == "guest":
        raise HTTPException(status_code=403, detail={"error": "ai_tools_require_account", "detail": "Crie uma conta para acessar as ferramentas de IA"})


@router.post("/day", response_model=AIToolResponse)
async def day_analysis(request: AIToolRequest, user: dict = Depends(get_current_user)):
    _require_full_account(user)
    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, DAY_ANALYSIS_PROMPT, "day", include_profile=True)


@router.post("/dream", response_model=AIToolResponse)
async def dream_analysis(request: AIToolRequest, user: dict = Depends(get_current_user)):
    _require_full_account(user)
    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, DREAM_ANALYSIS_PROMPT, "dream")


@router.post("/aura", response_model=AIToolResponse)
async def aura_reading(request: AIToolRequest, user: dict = Depends(get_current_user)):
    _require_full_account(user)
    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, AURA_READING_PROMPT, "aura", include_profile=True)


@router.post("/stoic", response_model=AIToolResponse)
async def stoic_advice(request: AIToolRequest, user: dict = Depends(get_current_user)):
    _require_full_account(user)
    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, STOIC_ADVICE_PROMPT, "stoic")


@router.post("/sync", response_model=AIToolResponse)
async def sync_reading(request: AIToolRequest, user: dict = Depends(get_current_user)):
    _require_full_account(user)
    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, SYNCHRONICITY_PROMPT, "sync", include_profile=True)
