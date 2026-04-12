from app.models import ReflectionCreate, UserAnswer, SemanticProfile


def build_reflection_toon(r: ReflectionCreate) -> str:
    if r.embeddingPayload:
        return r.embeddingPayload

    sp = r.semanticProfile or SemanticProfile()
    tag_list = sp.keywords or ["geral"]
    question_list = r.guidingQuestions or ["-"]
    ref_list = r.scriptureReferences or ["-"]
    origin = "Sistema" if r.isSystem else "Usuário"

    questions_body = "\n  - ".join(question_list)

    return f"""Reflexão: {r.title}
Origem: {origin}
Categoria: {r.categoryId}
Descrição: {r.description}
Perfil:
  Alvo: {sp.emotionalTarget}
  Resultado: {sp.emotionalOutcome}
  Nível: {sp.depthLevel}
Tags[{len(tag_list)}]: {", ".join(tag_list)}
Referências[{len(ref_list)}]: {", ".join(ref_list)}
Perguntas[{len(question_list)}]:
  - {questions_body}""".strip()


def build_answer_toon(answer: UserAnswer, reflection_title: str = "") -> str:
    return f"""Memória:
  Reflexão: {reflection_title or answer.reflectionId}
  Data: {answer.createdAt.strftime('%Y-%m-%d')}
  Conteúdo: {answer.content}""".strip()


def build_profile_toon(profile: dict) -> str:
    name = profile.get("display_name", "") or "desconhecido"
    gender = profile.get("gender", "") or "indefinido"
    level = profile.get("current_level", 1)
    xp = profile.get("total_xp", 0)
    streak = profile.get("current_streak", 0)

    themes = profile.get("recurring_themes", [])
    themes_str = ", ".join(themes) if themes else "nenhum identificado"
    personality = profile.get("personality_summary", "") or "ainda não definido"
    emotional = profile.get("emotional_state", "") or "não avaliado"
    spiritual = profile.get("spiritual_progress", "") or "início da jornada"

    return f"""Perfil:
  Nome: {name}
  Gênero: {gender}
  Nível: {level} | XP: {xp} | Sequência: {streak}d
  Personalidade: {personality}
  Emocional: {emotional}
  Temas[{len(themes)}]: {themes_str}
  Espiritual: {spiritual}""".strip()


def build_conversation_summary_toon(summary: str) -> str:
    if not summary:
        return ""
    return f"Resumo da conversa anterior: {summary}".strip()
