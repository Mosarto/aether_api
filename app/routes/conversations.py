from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException, Depends
from starlette.responses import Response

from app.auth import get_current_user
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import COL_CONVERSATIONS, SESSION_TTL_HOURS, logger
from app.models import SessionInfo
from app.providers import qdrant

router = APIRouter(tags=["Conversas"])


def _is_session_active(last_activity_str: str) -> bool:
    try:
        last = datetime.fromisoformat(last_activity_str)
        return (datetime.now(timezone.utc) - last) < timedelta(hours=SESSION_TTL_HOURS)
    except Exception:
        return False


def _get_session_turns(session_id: str) -> list[dict]:
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=False)),
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        turns = sorted(results, key=lambda p: (p.payload or {}).get("timestamp", ""))
        return [
            {
                "role": (p.payload or {}).get("role", ""),
                "content": (p.payload or {}).get("content", ""),
                "timestamp": (p.payload or {}).get("timestamp", ""),
                "used_memory_ids": (p.payload or {}).get("used_memory_ids", []),
                "used_scriptures": (p.payload or {}).get("used_scriptures", []),
            }
            for p in turns if p.payload
        ]
    except UnexpectedResponse as e:
        if e.status_code == 404:
            return []
        raise
    except Exception:
        return []


@router.get("/conversations")
async def list_user_sessions(user: dict = Depends(get_current_user)):
    try:
        results, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user["uid"])),
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
                ]
            ),
            limit=50,
            with_payload=True,
            with_vectors=False,
        )

        sessions = []
        for p in results:
            payload = p.payload or {}
            active = _is_session_active(payload.get("last_activity", ""))
            sessions.append(SessionInfo(
                sessionId=payload.get("session_id", ""),
                userId=payload.get("user_id", ""),
                title=payload.get("title", ""),
                reflectionId=payload.get("reflection_id") or None,
                turnCount=payload.get("turn_count", 0),
                createdAt=payload.get("created_at", ""),
                lastActivity=payload.get("last_activity", ""),
                active=active,
            ))

        sessions.sort(key=lambda s: s.lastActivity, reverse=True)
        return {"sessions": [s.model_dump(mode="json") for s in sessions]}

    except UnexpectedResponse as e:
        if e.status_code == 404:
            return {"sessions": []}
        logger.error("Erro ao listar sessões (Qdrant): %s", e)
        raise HTTPException(status_code=500, detail="Erro ao listar sessões")
    except Exception as e:
        logger.error("Erro ao listar sessões: %s", e)
        raise HTTPException(status_code=500, detail="Erro ao listar sessões")


@router.get("/conversations/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        turns = _get_session_turns(session_id)
        if not turns:
            raise HTTPException(status_code=404, detail="Sessão não encontrada")

        clean_turns = [{"role": t["role"], "content": t["content"], "timestamp": t["timestamp"]} for t in turns]

        title = ""
        session_user_id = ""
        try:
            meta, _ = qdrant.scroll(
                collection_name=COL_CONVERSATIONS,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                        qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if meta:
                mp = meta[0].payload or {}
                title = mp.get("title", "")
                session_user_id = mp.get("user_id", "")
        except Exception as e:
            logger.warning("Falha ao buscar meta da sessão %s: %s", session_id, e)

        if session_user_id and session_user_id != user["uid"]:
            raise HTTPException(status_code=403, detail="Acesso negado a esta sessão")

        return {"sessionId": session_id, "title": title, "turns": clean_turns, "turnCount": len(clean_turns)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao buscar sessão %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Erro ao buscar sessão")


@router.delete("/conversations/{session_id}", status_code=204)
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    try:
        # Fetch meta point
        meta, _ = qdrant.scroll(
            collection_name=COL_CONVERSATIONS,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                    qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not meta:
            raise HTTPException(status_code=404, detail="Sessão não encontrada")

        meta_point = meta[0]
        meta_payload = meta_point.payload or {}
        session_user_id = meta_payload.get("user_id", "")

        if session_user_id and session_user_id != user["uid"]:
            raise HTTPException(status_code=403, detail="Acesso negado a esta sessão")

        # Race protection: mark as processed before deletion
        qdrant.set_payload(
            collection_name=COL_CONVERSATIONS,
            payload={"processed": True},
            points=[meta_point.id],
        )

        # Paginated deletion of turn points
        offset = None
        while True:
            results, next_offset = qdrant.scroll(
                collection_name=COL_CONVERSATIONS,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                        qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=False)),
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not results:
                break
            point_ids = [p.id for p in results]
            qdrant.delete(
                collection_name=COL_CONVERSATIONS,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            offset = next_offset
            if offset is None:
                break

        # Delete meta point last
        qdrant.delete(
            collection_name=COL_CONVERSATIONS,
            points_selector=qmodels.PointIdsList(points=[meta_point.id]),
        )

        return Response(status_code=204)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erro ao deletar sessão %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Erro ao deletar sessão")
