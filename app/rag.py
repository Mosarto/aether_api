from qdrant_client.http import models

from app.config import COL_USER_MEMORIES, COL_REFLECTIONS
from app.providers import qdrant


def retrieve_context(
    user_id: str,
    query: str,
    limit_memories: int = 3,
    limit_recs: int = 2,
    used_memory_ids: list[str] | None = None,
    used_scripture_refs: list[str] | None = None,
):
    used_memory_ids = used_memory_ids or []
    used_scripture_refs = used_scripture_refs or []

    try:
        memories = qdrant.query(
            collection_name=COL_USER_MEMORIES,
            query_text=query,
            limit=limit_memories + len(used_memory_ids),
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
        )
        memories = [m for m in memories if str(m.id) not in used_memory_ids][:limit_memories]
    except Exception:
        memories = []

    try:
        recommendations = qdrant.query(
            collection_name=COL_REFLECTIONS,
            query_text=query,
            limit=limit_recs + len(used_scripture_refs),
        )
        if used_scripture_refs:
            recommendations = [
                r for r in recommendations
                if not any(ref in (r.metadata.get("scripture_refs", "")) for ref in used_scripture_refs)
            ][:limit_recs]
    except Exception:
        recommendations = []

    return memories, recommendations


def build_llm_prompt(user_query: str, memories: list, recommendations: list, has_history: bool = False) -> str:
    mem_block = ""
    if memories:
        n = len(memories)
        items = []
        for m in memories:
            title = m.metadata.get("reflection_title", "")
            content = m.metadata.get("content", m.metadata.get("toon_context", ""))[:200]
            items.append(f"  - {title}: {content}" if title else f"  - {content}")
        mem_block = f"\nPassado[{n}]:\n" + "\n".join(items)

    rec_block = ""
    if recommendations:
        n = len(recommendations)
        rows = []
        for r in recommendations:
            title = r.metadata.get("title", "?")
            target = r.metadata.get("target_emotion", "?")
            refs = r.metadata.get("scripture_refs", "")
            rows.append(f"  {title}|{target}|{refs}")
        rec_block = f"\nSugestões[{n}|]{{título|alvo|refs}}:\n" + "\n".join(rows)

    has_context = bool(mem_block or rec_block)

    history_note = (
        "\nHá histórico acima. NÃO repita informações. Responda como quem já está na conversa."
    ) if has_history else ""

    if not has_context:
        return f"""Mensagem: "{user_query}"

Siga as regras do sistema.{history_note}""".strip()

    return f"""Contexto interno (NÃO narrar ao usuário):{mem_block}{rec_block}

Mensagem: "{user_query}"

Siga as regras do sistema. Use o contexto para entender, não para narrar.{history_note}""".strip()
