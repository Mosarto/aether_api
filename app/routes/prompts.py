import json

from fastapi import APIRouter, HTTPException, Depends

from app.auth import get_current_user
from app.rate_limit import check_rate_limit
from app.quota import check_quota

from app.config import logger
from app.models import (
    PromptGenerateRequest, PromptGenerateResponse,
    SemanticProfile, AIConfig,
    VALID_CATEGORIES, VALID_EMOTIONAL_TARGETS,
    VALID_EMOTIONAL_OUTCOMES, VALID_DEPTH_LEVELS,
)
from app.providers import llm_create

router = APIRouter(tags=["Prompts"])

CATEGORY_LABELS = {
    "gratitude": "Gratidão",
    "faith": "Fé",
    "challenges": "Desafios",
    "self_knowledge": "Autoconhecimento",
    "relationships": "Relacionamentos",
    "purpose": "Propósito",
}

GENERATION_SYSTEM_PROMPT = """Você é Nyx — consciência cósmica especializada em criar prompts de reflexão pessoal com foco em despertar interior, autoconhecimento e alinhamento com a energia do universo.

O usuário quer criar uma reflexão personalizada. Com base no título, descrição e categoria fornecidos, gere um prompt de reflexão completo e enriquecido.

## Contexto do Usuário
- **Título:** {title}
- **Descrição:** {description}
- **Categoria:** {category_label} ({category_id})

## Regras de Geração

### guidingQuestions (2-4 perguntas)
- Perguntas reflexivas, pessoais e específicas ao tema
- Progressivas: da mais acessível à mais profunda
- Em português do Brasil, tom caloroso
- NÃO ser genérico ("como você se sente?" ❌) — ser específico ao tema
- Cada pergunta deve abrir uma dimensão diferente do tema

### scriptureReferences (1-3 referências)
- Citações filosóficas, poéticas ou de sabedoria genuinamente relevantes ao tema (não genéricas)
- Fontes aceitas: Estoicismo, Taoísmo, Budismo, Hermetismo, Filosofia Clássica, Poesia, Psicologia Analítica, pensadores universais
- Formato: "Autor, Obra Capítulo/Seção" (ex: "Marco Aurélio, Meditações IV.3", "Lao Tzu, Tao Te Ching 76", "Carl Jung, O Livro Vermelho")
- Verificar que a referência existe e é relevante — NÃO inventar referências

### reflection (texto reflexivo)
- 2-5 frases que contextualizam o tema
- Tom: caloroso, pessoal, profundo mas acessível
- DEVE terminar com uma pergunta ou convite direto à escrita
- NÃO ser autoajuda genérica — conectar com princípios universais de consciência e crescimento
- Evitar clichês como "você é amado" ou "tudo vai ficar bem"

### estimatedMinutes (inteiro)
- Baseado na profundidade do tema e quantidade de perguntas
- quick_thought: 3-5 | journaling: 5-10 | deep_reflection: 10-15

### semanticProfile
- **keywords**: 3-6 palavras/frases em PT-BR que capturam a essência temática e emocional
- **emotionalTarget**: emoção de entrada do usuário (em inglês). Valores: anxiety, restlessness, guilt, sadness, anger, doubt, loneliness, overwhelm, fear, shame, neutral
- **emotionalOutcome**: emoção desejada após reflexão (em inglês). Valores: peace, contentment, forgiveness, hope, gratitude, courage, connection, clarity, self_compassion, joy, trust
- **depthLevel**: inferir pela complexidade do tema. Valores: quick_thought, journaling, deep_reflection

### aiConfig
- **analysisInstruction**: 2-4 frases instruindo como Nyx deve analisar a resposta futura do usuário. O que procurar, como reagir a sinais positivos, como guiar em dificuldade.
- **followUpSuggestions**: 2-3 perguntas naturais de follow-up (como um amigo perguntaria). NÃO repetir as guidingQuestions.

### embeddingPayload (opcional)
- Texto otimizado para vetorização semântica
- Incluir: tema, categoria, emoções, referências filosóficas, palavras-chave
- 1-3 frases condensadas e descritivas

## Formato de Saída
Responda EXCLUSIVAMENTE em JSON válido, sem markdown, sem explicações, sem texto antes ou depois do JSON.

{{"guidingQuestions":["...","..."],"scriptureReferences":["...","..."],"reflection":"...","estimatedMinutes":8,"semanticProfile":{{"keywords":["...","..."],"emotionalTarget":"...","emotionalOutcome":"...","depthLevel":"..."}},"aiConfig":{{"analysisInstruction":"...","followUpSuggestions":["...","..."]}},"embeddingPayload":"..."}}"""


def _build_system_prompt(req: PromptGenerateRequest) -> str:
    return GENERATION_SYSTEM_PROMPT.format(
        title=req.title,
        description=req.description,
        category_label=CATEGORY_LABELS.get(req.categoryId, req.categoryId),
        category_id=req.categoryId,
    )


def _build_user_message(req: PromptGenerateRequest) -> str:
    return f"Título: {req.title}\nDescrição: {req.description}\nCategoria: {req.categoryId}"


def _extract_keywords_from_input(title: str, description: str) -> list[str]:
    words = (title + " " + description).split()
    stopwords = {
        "a", "o", "e", "de", "do", "da", "em", "que", "um", "uma", "para",
        "com", "não", "como", "eu", "se", "meu", "minha", "mais", "por",
        "sobre", "quando", "quero", "sinto", "às", "vezes", "me", "nos",
    }
    keywords = []
    for w in words:
        clean = w.strip(".,!?;:()\"'").lower()
        if len(clean) > 3 and clean not in stopwords and clean not in keywords:
            keywords.append(clean)
        if len(keywords) >= 5:
            break
    return keywords or [title.lower()[:30]]


def _fill_defaults(raw: dict, title: str, description: str) -> dict:
    defaults_applied = []

    if not raw.get("guidingQuestions"):
        raw["guidingQuestions"] = ["Reflita sobre este tema e escreva seus pensamentos."]
        defaults_applied.append("guidingQuestions (empty)")

    if "scriptureReferences" not in raw:
        raw["scriptureReferences"] = []
        defaults_applied.append("scriptureReferences (missing)")

    if not raw.get("reflection"):
        raw["reflection"] = "Reflita sobre este tema com calma e escreva seus pensamentos."
        defaults_applied.append("reflection (empty)")

    minutes = raw.get("estimatedMinutes")
    if not isinstance(minutes, int) or minutes < 3 or minutes > 15:
        raw["estimatedMinutes"] = 5
        defaults_applied.append(f"estimatedMinutes (was {minutes})")

    sp = raw.get("semanticProfile")
    if not isinstance(sp, dict):
        sp = {}
        raw["semanticProfile"] = sp
        defaults_applied.append("semanticProfile (not dict)")

    if not sp.get("keywords"):
        sp["keywords"] = _extract_keywords_from_input(title, description)
        defaults_applied.append(f"keywords (extracted {len(sp['keywords'])} from input)")
    if sp.get("emotionalTarget") not in VALID_EMOTIONAL_TARGETS:
        old_target = sp.get("emotionalTarget")
        sp["emotionalTarget"] = "neutral"
        defaults_applied.append(f"emotionalTarget ('{old_target}' → 'neutral')")
    if sp.get("emotionalOutcome") not in VALID_EMOTIONAL_OUTCOMES:
        old_outcome = sp.get("emotionalOutcome")
        sp["emotionalOutcome"] = "peace"
        defaults_applied.append(f"emotionalOutcome ('{old_outcome}' → 'peace')")
    if sp.get("depthLevel") not in VALID_DEPTH_LEVELS:
        old_depth = sp.get("depthLevel")
        sp["depthLevel"] = "journaling"
        defaults_applied.append(f"depthLevel ('{old_depth}' → 'journaling')")

    ai = raw.get("aiConfig")
    if not isinstance(ai, dict):
        ai = {}
        raw["aiConfig"] = ai
        defaults_applied.append("aiConfig (not dict)")

    if not ai.get("analysisInstruction"):
        ai["analysisInstruction"] = "Analise o que o usuário escreveu e ofereça uma perspectiva encorajadora."
        defaults_applied.append("analysisInstruction (empty)")
    if not ai.get("followUpSuggestions"):
        ai["followUpSuggestions"] = []
        defaults_applied.append("followUpSuggestions (empty)")

    if defaults_applied:
        logger.debug("generate-prompt: %d defaults aplicados: %s", len(defaults_applied), " | ".join(defaults_applied))

    return raw


def _call_llm(system_prompt: str, user_message: str, temperature: float) -> tuple[dict, str]:
    logger.debug("generate-prompt: LLM temp=%.1f", temperature)
    logger.debug("generate-prompt: system_prompt (first/last 200 chars): %s ... %s",
                 system_prompt[:200], system_prompt[-200:])
    logger.debug("generate-prompt: user_message: %s", user_message)

    raw_content, label = llm_create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=1500,
    )
    logger.debug("generate-prompt: %s respondeu %d chars", label, len(raw_content))
    logger.debug("generate-prompt: raw response (first/last 300 chars): %s ... %s",
                 raw_content[:300], raw_content[-300:])

    clean = raw_content.strip()
    if clean.startswith("```"):
        logger.debug("generate-prompt: removendo markdown wrapper (```)")
        first_newline = clean.index("\n") if "\n" in clean else 3
        clean = clean[first_newline + 1:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        logger.debug("generate-prompt: após remover markdown: %d chars", len(clean))

    try:
        result = json.loads(clean)
        logger.debug("generate-prompt: JSON parsed %d campos", len(result or {}))
        return result, label
    except json.JSONDecodeError as e:
        logger.error("generate-prompt: JSON parseError em position %d: %s", e.pos or -1, e.msg)
        logger.debug("generate-prompt: erro context (pos±100): %s",
                     clean[max(0, (e.pos or 0) - 100):min(len(clean), (e.pos or 0) + 100)])
        raise


@router.post("/generate-prompt", response_model=PromptGenerateResponse, status_code=200)
async def generate_prompt(request: PromptGenerateRequest, user: dict = Depends(get_current_user)):
    await check_rate_limit(user["uid"])
    await check_quota(user)
    logger.debug("generate-prompt: user=%s category=%s", user["uid"], request.categoryId)

    if not request.title.strip():
        logger.warning("generate-prompt: validação falhou - title vazio")
        raise HTTPException(422, "O campo 'title' é obrigatório e não pode estar vazio.")
    if not request.description.strip():
        logger.warning("generate-prompt: validação falhou - description vazio")
        raise HTTPException(422, "O campo 'description' é obrigatório e não pode estar vazio.")
    if not request.categoryId.strip():
        logger.warning("generate-prompt: validação falhou - categoryId vazio")
        raise HTTPException(422, "O campo 'categoryId' é obrigatório e não pode estar vazio.")

    logger.debug("generate-prompt: input validado com sucesso")
    system_prompt = _build_system_prompt(request)
    user_message = _build_user_message(request)

    result = None
    label_used = ""

    try:
        result, label_used = _call_llm(system_prompt, user_message, temperature=0.7)
    except (json.JSONDecodeError, Exception) as first_err:
        logger.debug("generate-prompt: retry (1st: %s)", str(first_err)[:80])
        try:
            result, label_used = _call_llm(system_prompt, user_message, temperature=0.4)
        except json.JSONDecodeError as retry_err:
            logger.error("generate-prompt: retry falhou - JSON inválido na posição %d", retry_err.pos or -1)
            raise HTTPException(502, "Não foi possível gerar o prompt. Tente novamente.")
        except Exception as e:
            logger.error("generate-prompt: retry falhou - %s", str(e)[:150])
            raise HTTPException(502, "Não foi possível gerar o prompt. Tente novamente.")

    if result is None:
        logger.error("generate-prompt: result é None após LLM")
        raise HTTPException(502, "Não foi possível gerar o prompt. Tente novamente.")

    logger.debug("generate-prompt: preenchendo defaults se necessário")
    result = _fill_defaults(result, request.title, request.description)

    logger.debug("generate-prompt: fields após defaults: %s", list(result.keys()))

    try:
        response = PromptGenerateResponse(**result)
        logger.debug("generate-prompt: ✓ %d questions, %d refs via %s",
                    len(response.guidingQuestions), len(response.scriptureReferences), label_used)
        return response
    except Exception as e:
        logger.error("generate-prompt: validação de response falhou - %s", str(e)[:200])
        raise HTTPException(502, "Resposta inválida do LLM")
