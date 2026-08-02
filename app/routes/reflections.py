from fastapi import APIRouter, Depends, HTTPException
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.auth import get_current_user
from app.config import COL_REFLECTIONS, deterministic_uuid, logger
from app.models import AIConfig, ReflectionCreate, SemanticProfile
from app.providers import qdrant
from app.rate_limit import check_rate_limit
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
    # Writes land in a catalog shared by every user and feed their RAG context,
    # so this path is rate limited like the LLM ones.
    await check_rate_limit(user["uid"])
    try:
        # The reflections catalog is shared across users (seeded idempotently by
        # clients), so ids are global — but overwriting an entry another user
        # created is rejected to block cross-user content tampering.
        point_id = deterministic_uuid(reflection.id)
        try:
            existing = qdrant.retrieve(
                collection_name=COL_REFLECTIONS, ids=[point_id], with_payload=True,
            )
        except Exception:
            existing = []
        if existing:
            owner = (existing[0].payload or {}).get("created_by", "")
            if owner and owner != user["uid"]:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "reflection_id_conflict", "id": reflection.id},
                )

        toon = build_reflection_toon(reflection)
        sp = reflection.semanticProfile or SemanticProfile()
        ai = reflection.aiConfig or AIConfig()

        qdrant.add(
            collection_name=COL_REFLECTIONS,
            documents=[toon],
            metadata=[{
                "original_id": reflection.id,
                "created_by": user["uid"],
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
            ids=[point_id],
        )

        return {
            "status": "indexed",
            "id": reflection.id,
            "title": reflection.title,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao indexar reflexão %s: %s", reflection.id, e)
        raise HTTPException(status_code=500, detail="Erro ao indexar reflexão")
