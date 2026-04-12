from fastapi import APIRouter, HTTPException, Depends

from app.auth import get_current_user
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import COL_REFLECTIONS, deterministic_uuid, logger
from app.models import ReflectionCreate, SemanticProfile, AIConfig
from app.providers import qdrant
from app.toon import build_reflection_toon

router = APIRouter(tags=["Reflexões"])


@router.get("/reflections/{reflection_id}/exists")
async def check_reflection_exists(reflection_id: str, user: dict = Depends(get_current_user)):
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_REFLECTIONS,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="original_id", match=models.MatchValue(value=reflection_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if results:
            point = results[0]
            return {
                "exists": True,
                "id": reflection_id,
                "title": point.payload.get("title", ""),
                "category": point.payload.get("category", ""),
            }

        return {"exists": False, "id": reflection_id}

    except UnexpectedResponse as e:
        if e.status_code == 404:
            return {"exists": False, "id": reflection_id}
        logger.error("Erro ao consultar reflexão %s: %s", reflection_id, e)
        raise HTTPException(status_code=500, detail="Erro ao consultar reflexão")
    except Exception as e:
        logger.error("Erro ao consultar reflexão %s: %s", reflection_id, e)
        raise HTTPException(status_code=500, detail="Erro ao consultar reflexão")


@router.post("/reflections", status_code=201)
async def create_reflection(reflection: ReflectionCreate, user: dict = Depends(get_current_user)):
    try:
        toon = build_reflection_toon(reflection)
        sp = reflection.semanticProfile or SemanticProfile()
        ai = reflection.aiConfig or AIConfig()

        qdrant.add(
            collection_name=COL_REFLECTIONS,
            documents=[toon],
            metadata=[{
                "original_id": reflection.id,
                "is_system": reflection.isSystem,
                "title": reflection.title,
                "category": reflection.categoryId,
                "description": reflection.description,
                "target_emotion": sp.emotionalTarget,
                "outcome_emotion": sp.emotionalOutcome,
                "depth_level": sp.depthLevel,
                "keywords": ", ".join(sp.keywords),
                "scripture_refs": ", ".join(reflection.scriptureReferences),
                "analysis_instruction": ai.analysisInstruction,
                "follow_up": " | ".join(ai.followUpSuggestions),
                "toon_content": toon,
            }],
            ids=[deterministic_uuid(reflection.id)],
        )

        return {
            "status": "indexed",
            "id": reflection.id,
            "title": reflection.title,
        }
    except Exception as e:
        logger.error("Erro ao indexar reflexão %s: %s", reflection.id, e)
        raise HTTPException(status_code=500, detail="Erro ao indexar reflexão")
