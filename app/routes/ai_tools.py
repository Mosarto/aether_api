import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.config import (
    AURA_READING_PROMPT,
    AI_TOOL_LLM_MAX_TOKENS,
    AI_TOOL_LLM_TEMPERATURE,
    COL_CONVERSATIONS,
    DREAM_ANALYSIS_PROMPT,
    STOIC_ADVICE_PROMPT,
    SYNCHRONICITY_PROMPT,
    logger,
)
from app.firebase import save_summary_to_firestore
from app.models import AIToolRequest, AIToolResponse
from app.profile import fetch_user_profile
from app.providers import llm_create, qdrant
from app.quota import check_quota
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


def _parse_json_response(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        first_nl = clean.index("\n") if "\n" in clean else 3
        clean = clean[first_nl + 1:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
    return json.loads(clean)


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
    2. Call LLM (Cerebras/Groq via llm_create)
    3. Parse JSON response
    4. Retry once with lower temperature on parse failure
    5. Save to Firestore
    6. Increment quota
    7. Return AIToolResponse
    """
    system_content = prompt
    user_content = content

    if include_profile:
        try:
            profile_data = fetch_user_profile(user["uid"])
            if profile_data:
                profile_toon = build_profile_toon(profile_data)
                user_content = f"[Contexto do usuário]\n{profile_toon}\n\n"

            recent = _fetch_recent_session_summaries(user["uid"])
            if recent:
                user_content += "[Conversas recentes]\n" + "\n".join(recent) + "\n\n"

            user_content += f"[Conteúdo]\n{content}"
        except Exception as e:
            logger.warning("Falha ao buscar perfil para %s: %s", tool_name, e)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    parsed = None
    try:
        raw_content, label = llm_create(
            messages=messages,
            temperature=AI_TOOL_LLM_TEMPERATURE,
            max_tokens=AI_TOOL_LLM_MAX_TOKENS,
        )
        parsed = _parse_json_response(raw_content)
    except (json.JSONDecodeError, Exception) as first_err:
        logger.debug("ai-tool %s: retry (1st: %s)", tool_name, str(first_err)[:80])
        try:
            raw_content, label = llm_create(
                messages=messages,
                temperature=0.4,
                max_tokens=AI_TOOL_LLM_MAX_TOKENS,
            )
            parsed = _parse_json_response(raw_content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=503, detail={"error": "llm_unavailable"})
        except RuntimeError:
            raise HTTPException(status_code=503, detail={"error": "llm_unavailable"})
        except Exception:
            raise HTTPException(status_code=503, detail={"error": "llm_unavailable"})

    if parsed is None:
        raise HTTPException(status_code=503, detail={"error": "llm_unavailable"})

    response = AIToolResponse(
        id=str(uuid4()),
        title=parsed.get("title", "Sem título")[:500],
        snippet=parsed.get("snippet", "")[:8000],
        tags=parsed.get("tags", [])[:8],
        date=datetime.now(timezone.utc),
        tool=tool_name,
    )

    try:
        save_summary_to_firestore(user["uid"], {
            "title": response.title,
            "snippet": response.snippet,
            "tags": response.tags,
            "date": response.date.isoformat(),
            "tool": response.tool,
        })
    except Exception as e:
        logger.warning("Falha ao salvar summary %s: %s", tool_name, e)

    return response


@router.post("/dream", response_model=AIToolResponse)
async def dream_analysis(request: AIToolRequest, user: dict = Depends(get_current_user)):
    if user.get("is_anonymous") or user.get("subscription_tier") == "guest":
        raise HTTPException(status_code=403, detail={"error": "ai_tools_require_account", "detail": "Crie uma conta para acessar as ferramentas de IA"})

    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, DREAM_ANALYSIS_PROMPT, "dream")


@router.post("/aura", response_model=AIToolResponse)
async def aura_reading(request: AIToolRequest, user: dict = Depends(get_current_user)):
    if user.get("is_anonymous") or user.get("subscription_tier") == "guest":
        raise HTTPException(status_code=403, detail={"error": "ai_tools_require_account", "detail": "Crie uma conta para acessar as ferramentas de IA"})

    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, AURA_READING_PROMPT, "aura", include_profile=True)


@router.post("/stoic", response_model=AIToolResponse)
async def stoic_advice(request: AIToolRequest, user: dict = Depends(get_current_user)):
    if user.get("is_anonymous") or user.get("subscription_tier") == "guest":
        raise HTTPException(status_code=403, detail={"error": "ai_tools_require_account", "detail": "Crie uma conta para acessar as ferramentas de IA"})

    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, STOIC_ADVICE_PROMPT, "stoic")


@router.post("/sync", response_model=AIToolResponse)
async def sync_reading(request: AIToolRequest, user: dict = Depends(get_current_user)):
    if user.get("is_anonymous") or user.get("subscription_tier") == "guest":
        raise HTTPException(status_code=403, detail={"error": "ai_tools_require_account", "detail": "Crie uma conta para acessar as ferramentas de IA"})

    await check_rate_limit(user["uid"])
    await check_quota(user)
    return await _process_ai_tool(user, request.content, SYNCHRONICITY_PROMPT, "sync", include_profile=True)
