from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends

from app.auth import get_current_user
from app.rate_limit import check_rate_limit
from app.quota import check_quota
from qdrant_client.http.exceptions import UnexpectedResponse
from google.genai import types as genai_types

from app.config import (
    SYSTEM_PROMPT, COL_CONVERSATIONS,
    CHAT_MAX_TURNS, COMPRESSION_MIN_TURNS, deterministic_uuid, logger,
)
from app.models import ChatRequest, ChatResponse
from app.providers import qdrant, llm_create, google_ai_create
from app.rag import retrieve_context, build_llm_prompt
from app.routes.conversations import _is_session_active, _get_session_turns
from app.profile import fetch_user_profile, compress_history, ensure_profiles_collection, create_initial_profile, sync_firebase_fields
from app.toon import build_profile_toon, build_conversation_summary_toon
from app.firebase import fetch_firestore_user

router = APIRouter(tags=["Chat"])

ZERO_VECTOR = [0.0] * 384


_turn_counter = 0

_GREETING_TOKENS = {
    "oi", "olá", "ola", "eai", "e aí", "e ai", "fala", "hey",
    "salve", "bom dia", "boa tarde", "boa noite", "hello", "hi",
    "opa", "yo", "fala aí", "fala ai", "coé", "coe",
}


def _is_trivial_greeting(message: str) -> bool:
    normalized = message.strip().lower().rstrip("!?.,")
    return normalized in _GREETING_TOKENS


def _save_turn(session_id: str, user_id: str, role: str, content: str, extra_payload: dict | None = None):
    global _turn_counter
    _turn_counter += 1
    point_id = deterministic_uuid(f"{session_id}:{role}:{_turn_counter}:{datetime.now(timezone.utc).isoformat()}")
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "is_session_meta": False,
    }
    if extra_payload:
        payload.update(extra_payload)

    from qdrant_client.http import models as qmodels
    qdrant.upsert(
        collection_name=COL_CONVERSATIONS,
        points=[qmodels.PointStruct(id=point_id, vector=ZERO_VECTOR, payload=payload)],
    )


def _upsert_session_meta(session_id: str, user_id: str, reflection_id: str | None, turn_count: int, created_at: str, title: str = ""):
    meta_id = deterministic_uuid(f"meta:{session_id}")
    from qdrant_client.http import models as qmodels
    qdrant.upsert(
        collection_name=COL_CONVERSATIONS,
        points=[qmodels.PointStruct(
            id=meta_id,
            vector=ZERO_VECTOR,
            payload={
                "session_id": session_id,
                "user_id": user_id,
                "reflection_id": reflection_id or "",
                "title": title,
                "turn_count": turn_count,
                "created_at": created_at,
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "is_session_meta": True,
                "processed": False,
            },
        )],
    )


def _generate_session_title(user_message: str, ai_response: str) -> str:
    try:
        content, _ = llm_create(
            messages=[{
                "role": "user",
                "content": (
                    "Crie um título curto (3 a 5 palavras, sem aspas, sem emoji) para esta conversa.\n"
                    f"Usuário: {user_message[:200]}\n"
                    f"Resposta: {ai_response[:200]}\n"
                    "Título:"
                ),
            }],
            temperature=0.3,
            max_tokens=20,
        )
        raw = content.strip().strip('"').strip("'").strip()
        title = raw[:60] if raw else "Nova conversa"
        logger.debug("título gerado: '%s'", title)
        return title
    except Exception as e:
        logger.warning("gerar título falhou: %s", e)
        return "Nova conversa"


_collection_verified = False


def _ensure_collection():
    global _collection_verified
    if _collection_verified:
        return
    try:
        qdrant.get_collection(COL_CONVERSATIONS)
        _collection_verified = True
    except (UnexpectedResponse, Exception):
        from qdrant_client.http import models as qmodels
        qdrant.create_collection(
            collection_name=COL_CONVERSATIONS,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        )
        _collection_verified = True


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    try:
        await check_rate_limit(user["uid"])
        await check_quota(user)
        session_id = req.sessionId or str(uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        _ensure_collection()

        history_turns: list[dict] = []
        used_memory_ids: list[str] = []
        used_scripture_refs: list[str] = []
        created_at = now_iso

        existing_title = ""
        existing_reflection_id = ""

        if req.sessionId:
            all_turns = _get_session_turns(session_id)

            if all_turns:
                first_ts = all_turns[0].get("timestamp", now_iso)
                last_ts = all_turns[-1].get("timestamp", now_iso)

                if not _is_session_active(last_ts):
                    session_id = str(uuid4())
                    logger.debug("sessão expirada, nova: %s", session_id)
                else:
                    created_at = first_ts
                    history_turns = all_turns[-CHAT_MAX_TURNS:]

                    recent_turns = all_turns[-CHAT_MAX_TURNS:]
                    for t in recent_turns:
                        used_memory_ids.extend(t.get("used_memory_ids", []))
                        used_scripture_refs.extend(t.get("used_scriptures", []))

                    try:
                        from qdrant_client.http import models as qm
                        meta, _ = qdrant.scroll(
                            collection_name=COL_CONVERSATIONS,
                            scroll_filter=qm.Filter(must=[
                                qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id)),
                                qm.FieldCondition(key="is_session_meta", match=qm.MatchValue(value=True)),
                            ]),
                            limit=1, with_payload=True, with_vectors=False,
                        )
                        if meta:
                            mp = meta[0].payload or {}
                            existing_title = mp.get("title", "")
                            existing_reflection_id = mp.get("reflection_id", "")
                    except Exception as e:
                        logger.debug("Falha ao buscar meta da sessão %s: %s", session_id, e)

        has_history = len(history_turns) > 0
        is_greeting = _is_trivial_greeting(req.message)

        if is_greeting:
            memories, recommendations = [], []
        else:
            memories, recommendations = retrieve_context(
                user["uid"], req.message,
                used_memory_ids=used_memory_ids,
                used_scripture_refs=used_scripture_refs,
            )
        user_prompt = build_llm_prompt(req.message, memories, recommendations, has_history=has_history)

        ensure_profiles_collection()
        profile_data = fetch_user_profile(user["uid"])

        if profile_data is None:
            firebase_user = fetch_firestore_user(user["uid"])
            if firebase_user:
                profile_data = create_initial_profile(user["uid"], firebase_user)
            else:
                profile_data = create_initial_profile(user["uid"], {"displayName": "", "totalXP": 0, "currentLevel": 1, "currentStreak": 0})
        elif not has_history:
            firebase_user = fetch_firestore_user(user["uid"])
            if firebase_user:
                profile_data = sync_firebase_fields(user["uid"], firebase_user, profile_data)

        profile_toon = build_profile_toon(profile_data) if profile_data else ""

        compressed_summary = ""
        if len(history_turns) >= COMPRESSION_MIN_TURNS:
            compressed_summary = compress_history(history_turns)

        contents: list[genai_types.Content] = []

        if profile_toon:
            contents.append(genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=f"[Contexto interno — perfil do usuário. NÃO narrar, NÃO mencionar diretamente.]\n{profile_toon}")],
            ))
            contents.append(genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Entendido, conheço o usuário. Vou seguir as regras do sistema.")],
            ))

        if compressed_summary:
            summary_toon = build_conversation_summary_toon(compressed_summary)
            contents.append(genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=f"[Contexto interno — resumo do histórico]\n{summary_toon}")],
            ))
            contents.append(genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Entendido, tenho o contexto.")],
            ))
        else:
            for turn in history_turns:
                role = "user" if turn["role"] == "user" else "model"
                contents.append(genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=turn["content"])],
                ))

        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_prompt)],
        ))

        ai_response, model_label = google_ai_create(
            contents=contents,
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )

        current_memory_ids = [str(m.id) for m in memories]
        current_scriptures = [r.metadata.get("scripture_refs", "") for r in recommendations if r.metadata.get("scripture_refs")]

        _save_turn(session_id, user["uid"], "user", req.message, {
            "used_memory_ids": current_memory_ids,
            "used_scriptures": current_scriptures,
        })
        _save_turn(session_id, user["uid"], "assistant", ai_response)

        total_turns = len(history_turns) + 2
        is_first_exchange = not has_history

        session_title: str | None = None
        if is_first_exchange:
            session_title = _generate_session_title(req.message, ai_response)

        final_title = session_title or existing_title
        final_reflection_id = req.reflectionId or existing_reflection_id or None

        _upsert_session_meta(
            session_id, user["uid"], final_reflection_id,
            total_turns, created_at,
            title=final_title,
        )

        follow_ups: list[str] = []
        for r in recommendations:
            raw = r.metadata.get("follow_up", "")
            if raw:
                follow_ups.extend([s.strip() for s in raw.split("|") if s.strip()])

        return ChatResponse(
            response=ai_response,
            model=model_label,
            contextSources=len(memories) + len(recommendations),
            followUp=follow_ups[:3],
            sessionId=session_id,
            sessionTitle=session_title,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro no chat: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno no chat")
