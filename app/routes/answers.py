from fastapi import APIRouter, HTTPException, Depends

from app.auth import get_current_user
from qdrant_client.http import models

from app.config import COL_REFLECTIONS, COL_USER_MEMORIES, deterministic_uuid, logger
from app.models import UserAnswer
from app.providers import qdrant
from app.toon import build_answer_toon

router = APIRouter(tags=["Respostas"])


@router.post("/user-answers", status_code=201)
async def submit_user_answer(answer: UserAnswer, user: dict = Depends(get_current_user)):
    try:
        reflection_title = ""
        try:
            results = qdrant.query(
                collection_name=COL_REFLECTIONS,
                query_text=answer.reflectionId,
                limit=1,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="original_id", match=models.MatchValue(value=answer.reflectionId))]
                ),
            )
            if results:
                reflection_title = results[0].metadata.get("title", "")
        except Exception as e:
            logger.debug("Falha ao buscar título da reflexão %s: %s", answer.reflectionId, e)

        toon = build_answer_toon(answer, reflection_title)

        qdrant.add(
            collection_name=COL_USER_MEMORIES,
            documents=[toon],
            metadata=[{
                "user_id": user["uid"],
                "reflection_id": answer.reflectionId,
                "reflection_title": reflection_title,
                "content": answer.content,
                "timestamp": answer.createdAt.timestamp(),
                "toon_context": toon,
            }],
            ids=[deterministic_uuid(answer.id)],
        )

        return {
            "status": "memory_saved",
            "id": answer.id,
            "reflectionTitle": reflection_title or answer.reflectionId,
        }
    except Exception as e:
        logger.error("Erro ao salvar resposta: %s", e)
        raise HTTPException(status_code=500, detail="Erro ao salvar resposta")
