from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from app.auth import get_current_user
from app.config import PROMPT_GENERATION_SYSTEM_PROMPT, logger
from app.llm import (
    LLM_UNAVAILABLE_DETAIL,
    AgnesError,
    complete_json_dict,
    was_billed,
    wrap_untrusted,
)
from app.models import (
    VALID_DEPTH_LEVELS,
    VALID_EMOTIONAL_OUTCOMES,
    VALID_EMOTIONAL_TARGETS,
    PromptGenerateRequest,
    PromptGenerateResponse,
)
from app.quota import check_quota, refund_quota
from app.rate_limit import check_rate_limit

router = APIRouter(tags=["Prompts"])

CATEGORY_LABELS = {
    "gratitude": "Gratidão",
    "faith": "Fé",
    "challenges": "Desafios",
    "self_knowledge": "Autoconhecimento",
    "relationships": "Relacionamentos",
    "purpose": "Propósito",
}


def _build_user_message(req: PromptGenerateRequest) -> str:
    category_label = CATEGORY_LABELS.get(req.categoryId, req.categoryId)
    return wrap_untrusted(
        "dados_usuario",
        f"Título: {req.title}\n"
        f"Descrição: {req.description}\n"
        f"Categoria: {category_label} ({req.categoryId})",
    )


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
        defaults_applied.append("estimatedMinutes")

    sp = raw.get("semanticProfile")
    if not isinstance(sp, dict):
        sp = {}
        raw["semanticProfile"] = sp
        defaults_applied.append("semanticProfile (not dict)")

    if not sp.get("keywords"):
        sp["keywords"] = _extract_keywords_from_input(title, description)
        defaults_applied.append("keywords (extracted from input)")
    if sp.get("emotionalTarget") not in VALID_EMOTIONAL_TARGETS:
        sp["emotionalTarget"] = "neutral"
        defaults_applied.append("emotionalTarget → neutral")
    if sp.get("emotionalOutcome") not in VALID_EMOTIONAL_OUTCOMES:
        sp["emotionalOutcome"] = "peace"
        defaults_applied.append("emotionalOutcome → peace")
    if sp.get("depthLevel") not in VALID_DEPTH_LEVELS:
        sp["depthLevel"] = "journaling"
        defaults_applied.append("depthLevel → journaling")

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


@router.post("/generate-prompt", response_model=PromptGenerateResponse, status_code=200)
async def generate_prompt(request: PromptGenerateRequest, user: dict = Depends(get_current_user)):
    await check_rate_limit(user["uid"])
    await check_quota(user)

    if not request.title.strip():
        raise HTTPException(422, "O campo 'title' é obrigatório e não pode estar vazio.")
    if not request.description.strip():
        raise HTTPException(422, "O campo 'description' é obrigatório e não pode estar vazio.")
    if not request.categoryId.strip():
        raise HTTPException(422, "O campo 'categoryId' é obrigatório e não pode estar vazio.")

    messages = [
        {"role": "system", "content": PROMPT_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(request)},
    ]

    try:
        result = await complete_json_dict("prompt_generation", messages)
    except AgnesError as e:
        logger.warning("generate-prompt indisponível: %s", e)
        if not was_billed(e):
            await refund_quota(user)
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE_DETAIL)

    result = _fill_defaults(result, request.title, request.description)

    try:
        response = PromptGenerateResponse(**result)
        logger.debug(
            "generate-prompt: ✓ %d questions, %d refs",
            len(response.guidingQuestions), len(response.scriptureReferences),
        )
        return response
    except ValidationError as e:
        # The completion was already billed, so the quota slot is NOT refunded.
        # Log only field locations — never the model output itself.
        fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in e.errors()[:5])
        logger.error("generate-prompt: resposta fora do contrato (campos: %s)", fields or "?")
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE_DETAIL)
