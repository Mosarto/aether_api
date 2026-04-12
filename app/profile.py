import json
from datetime import datetime, timezone

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models as qmodels

from app.config import (
    COL_USER_PROFILES, COMPRESSION_PROMPT, PROFILE_EXTRACTION_PROMPT,
    GENDER_INFERENCE_PROMPT, deterministic_uuid, logger,
)
from app.providers import qdrant, llm_create

ZERO_VECTOR = [0.0] * 384

_profiles_collection_verified = False


def ensure_profiles_collection():
    global _profiles_collection_verified
    if _profiles_collection_verified:
        return
    try:
        qdrant.get_collection(COL_USER_PROFILES)
        _profiles_collection_verified = True
    except (UnexpectedResponse, Exception):
        qdrant.create_collection(
            collection_name=COL_USER_PROFILES,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        )
        _profiles_collection_verified = True


def _profile_point_id(user_id: str) -> str:
    return deterministic_uuid(f"{user_id}_profile")


def fetch_user_profile(user_id: str) -> dict | None:
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_USER_PROFILES,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if results:
            return results[0].payload or {}
        return None
    except Exception as e:
        logger.warning("Falha ao buscar perfil do usuário %s: %s", user_id, e)
        return None


def upsert_user_profile(user_id: str, profile_data: dict):
    point_id = _profile_point_id(user_id)
    profile_data["user_id"] = user_id
    profile_data["last_updated"] = datetime.now(timezone.utc).isoformat()

    qdrant.upsert(
        collection_name=COL_USER_PROFILES,
        points=[qmodels.PointStruct(
            id=point_id,
            vector=ZERO_VECTOR,
            payload=profile_data,
        )],
    )
    logger.debug("perfil %s atualizado (v%s)", user_id, profile_data.get("version", 1))


def _infer_gender(display_name: str) -> str:
    if not display_name:
        return "indefinido"
    try:
        content, _ = llm_create(
            messages=[
                {"role": "system", "content": GENDER_INFERENCE_PROMPT},
                {"role": "user", "content": display_name},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = content.strip().lower()
        if "masculino" in raw:
            return "masculino"
        if "feminino" in raw:
            return "feminino"
        return "indefinido"
    except Exception as e:
        logger.warning("Falha ao inferir gênero de '%s': %s", display_name, e)
        return "indefinido"


def create_initial_profile(user_id: str, firebase_data: dict) -> dict:
    display_name = firebase_data.get("displayName", "")
    gender = _infer_gender(display_name)

    profile_data = {
        "user_id": user_id,
        "display_name": display_name,
        "gender": gender,
        "total_xp": firebase_data.get("totalXP", 0),
        "current_level": firebase_data.get("currentLevel", 1),
        "current_streak": firebase_data.get("currentStreak", 0),
        "personality_summary": "",
        "emotional_state": "",
        "recurring_themes": [],
        "spiritual_progress": "início do despertar",
        "version": 1,
        "conversation_count": 0,
    }

    upsert_user_profile(user_id, profile_data)
    logger.debug("perfil inicial criado para %s (%s)", user_id, gender)
    return profile_data


def sync_firebase_fields(user_id: str, firebase_data: dict, existing_profile: dict) -> dict:
    fields = {
        "display_name": firebase_data.get("displayName", ""),
        "total_xp": firebase_data.get("totalXP", 0),
        "current_level": firebase_data.get("currentLevel", 1),
        "current_streak": firebase_data.get("currentStreak", 0),
    }

    changed = any(existing_profile.get(k) != v for k, v in fields.items())
    if not changed:
        return existing_profile

    existing_profile.update(fields)
    upsert_user_profile(user_id, existing_profile)
    return existing_profile


def compress_history(turns: list[dict]) -> str:
    if not turns:
        return ""

    lines = []
    for t in turns:
        role = "Usuário" if t.get("role") == "user" else "IA"
        lines.append(f"{role}: {t.get('content', '')}")
    history_text = "\n".join(lines)

    try:
        content, label = llm_create(
            messages=[
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": f"Histórico:\n{history_text}"},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        logger.debug("histórico comprimido: %d turns → %d chars", len(turns), len(content))
        return content.strip()
    except Exception as e:
        logger.warning("compress_history falhou: %s", e)
        return ""


def extract_profile_updates(current_profile: dict | None, conversation_summary: str) -> dict:
    if not conversation_summary:
        return {}

    profile_text = "Perfil atual: VAZIO (primeira conversa)" if not current_profile else (
        f"Perfil atual:\n"
        f"  Personalidade: {current_profile.get('personality_summary', '')}\n"
        f"  Estado emocional: {current_profile.get('emotional_state', '')}\n"
        f"  Temas recorrentes: {', '.join(current_profile.get('recurring_themes', []))}\n"
        f"  Progresso espiritual: {current_profile.get('spiritual_progress', '')}"
    )

    try:
        content, label = llm_create(
            messages=[
                {"role": "system", "content": PROFILE_EXTRACTION_PROMPT},
                {"role": "user", "content": f"{profile_text}\n\nResumo da conversa recente:\n{conversation_summary}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        logger.debug("perfil extraído: %d chars", len(content))

        clean = content.strip()
        if clean.startswith("```"):
            first_nl = clean.index("\n") if "\n" in clean else 3
            clean = clean[first_nl + 1:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

        result = json.loads(clean)

        if "recurring_themes" in result and isinstance(result["recurring_themes"], list):
            result["recurring_themes"] = result["recurring_themes"][:8]

        return result
    except Exception as e:
        logger.warning("extract_profile falhou: %s", e)
        return {}
