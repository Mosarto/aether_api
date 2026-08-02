from datetime import datetime, timezone

from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import (
    AKASHIC_METADATA_PROMPT,
    COL_USER_PROFILES,
    COMPRESSION_PROMPT,
    GENDER_INFERENCE_PROMPT,
    PROFILE_EXTRACTION_PROMPT,
    deterministic_uuid,
    logger,
)
from app.llm import AgnesError, complete, complete_json, wrap_untrusted
from app.llm_schemas import AkashicMetadata, ProfileUpdate
from app.providers import qdrant

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


async def infer_gender(display_name: str) -> str:
    if not display_name:
        return "indefinido"
    try:
        result = await complete(
            "gender_inference",
            messages=[
                {"role": "system", "content": GENDER_INFERENCE_PROMPT},
                {"role": "user", "content": wrap_untrusted("dados_usuario", display_name)},
            ],
        )
        raw = result.content.strip().lower()
        if "masculino" in raw:
            return "masculino"
        if "feminino" in raw:
            return "feminino"
        return "indefinido"
    except AgnesError as e:
        logger.warning("Falha ao inferir gênero: %s", e)
        return "indefinido"


async def create_initial_profile(user_id: str, firebase_data: dict) -> dict:
    display_name = firebase_data.get("displayName", "")
    gender = await infer_gender(display_name)

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
    logger.debug("perfil inicial criado para %s", user_id)
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


async def compress_history(turns: list[dict]) -> str:
    if not turns:
        return ""

    lines = []
    for t in turns:
        role = "Usuário" if t.get("role") == "user" else "IA"
        lines.append(f"{role}: {t.get('content', '')}")
    history_text = "\n".join(lines)

    try:
        result = await complete(
            "history_compression",
            messages=[
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": wrap_untrusted("historico", history_text)},
            ],
        )
        logger.debug("histórico comprimido: %d turns → %d chars", len(turns), len(result.content))
        return result.content.strip()
    except AgnesError as e:
        logger.warning("compress_history falhou: %s", e)
        return ""


async def extract_profile_updates(current_profile: dict | None, conversation_summary: str) -> dict:
    if not conversation_summary:
        return {}

    profile_text = "VAZIO (primeira conversa)" if not current_profile else (
        f"Personalidade: {current_profile.get('personality_summary', '')}\n"
        f"Estado emocional: {current_profile.get('emotional_state', '')}\n"
        f"Temas recorrentes: {', '.join(current_profile.get('recurring_themes', []))}\n"
        f"Progresso espiritual: {current_profile.get('spiritual_progress', '')}"
    )

    user_content = (
        f"{wrap_untrusted('perfil_atual', profile_text)}\n\n"
        f"{wrap_untrusted('resumo_conversa', conversation_summary)}"
    )

    try:
        update = await complete_json(
            "profile_extraction",
            messages=[
                {"role": "system", "content": PROFILE_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            schema=ProfileUpdate,
        )
        return update.model_dump()
    except AgnesError as e:
        logger.warning("extract_profile falhou: %s", e)
        return {}


async def extract_akashic_metadata(summary: str, turn_count: int) -> dict:
    """Extract mood, emotional intensity, and key insight from a session summary.

    Returns at minimum {"turnCount": turn_count}. On success also includes
    mood, emotionalIntensity, and keyInsight.
    """
    base = {"turnCount": turn_count}

    if not summary:
        return base

    try:
        meta = await complete_json(
            "akashic_metadata",
            messages=[
                {"role": "system", "content": AKASHIC_METADATA_PROMPT},
                {"role": "user", "content": wrap_untrusted("resumo_conversa", summary)},
            ],
            schema=AkashicMetadata,
        )
    except AgnesError as e:
        logger.warning("extract_akashic_metadata falhou: %s", e)
        return base

    base["mood"] = meta.mood
    base["emotionalIntensity"] = meta.emotionalIntensity
    base["keyInsight"] = meta.keyInsight
    return base
