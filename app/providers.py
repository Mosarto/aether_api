import itertools
from dataclasses import dataclass

from qdrant_client import QdrantClient
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

from app.config import (
    QDRANT_URL, QDRANT_API_KEY,
    CEREBRAS_BASE_URL, CEREBRAS_API_KEY, CEREBRAS_MODEL,
    GROQ_BASE_URL, GROQ_API_KEY, GROQ_MODEL,
    GOOGLE_AI_API_KEY, GOOGLE_AI_MODEL,
    logger,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


@dataclass
class LLMProvider:
    name: str
    client: OpenAI
    model: str
    token_param: str


_providers: list[LLMProvider] = []

if CEREBRAS_API_KEY:
    _providers.append(LLMProvider(
        name="cerebras",
        client=OpenAI(base_url=CEREBRAS_BASE_URL, api_key=CEREBRAS_API_KEY),
        model=CEREBRAS_MODEL,
        token_param="max_tokens",
    ))

if GROQ_API_KEY:
    _providers.append(LLMProvider(
        name="groq",
        client=OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY),
        model=GROQ_MODEL,
        token_param="max_completion_tokens",
    ))

if not _providers:
    raise RuntimeError("Nenhuma API key configurada (CEREBRAS_API_KEY ou GROQ_API_KEY)")

_cycle = itertools.cycle(_providers)

google_ai_client: genai.Client | None = None
if GOOGLE_AI_API_KEY:
    google_ai_client = genai.Client(api_key=GOOGLE_AI_API_KEY)


def get_next_llm() -> LLMProvider:
    return next(_cycle)


def llm_create(messages: list, temperature: float = 0.7, max_tokens: int = 600) -> tuple[str, str]:
    first = get_next_llm()
    providers_to_try = [first] + [p for p in _providers if p.name != first.name]

    for provider in providers_to_try:
        try:
            kwargs = {
                "model": provider.model,
                "messages": messages,
                "temperature": temperature,
                provider.token_param: max_tokens,
            }
            completion = provider.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content or ""
            label = f"{provider.name}-{provider.model}"
            return content, label
        except Exception as e:
            logger.warning("Falha no provider %s: %s", provider.name, e)
            continue

    raise RuntimeError("Todos os providers LLM falharam")


def google_ai_create(
    contents: list[genai_types.Content],
    system_instruction: str,
    temperature: float = 0.7,
) -> tuple[str, str]:
    if not google_ai_client:
        raise RuntimeError("GOOGLE_AI_API_KEY não configurada")

    response = google_ai_client.models.generate_content(
        model=GOOGLE_AI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    label = f"google-{GOOGLE_AI_MODEL}"
    text = ""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if not getattr(part, "thought", False):
                text += part.text or ""
    if not text and response.candidates:
        c = response.candidates[0]
        logger.warning(
            "google-ai resposta vazia — finish_reason=%s",
            getattr(c, "finish_reason", "?"),
        )
    return text, label
