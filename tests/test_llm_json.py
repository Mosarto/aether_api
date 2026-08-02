"""Structured-output tests: strict parse, Pydantic validation, single corrective retry."""

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from app.llm import (
    AgnesInvalidOutputError,
    complete_json,
    complete_json_dict,
    strip_markdown_fences,
)
from app.llm_schemas import AIToolResult, AkashicMetadata, ProfileUpdate


def run(coro):
    return asyncio.run(coro)


def _messages():
    return [{"role": "system", "content": "responda JSON"}, {"role": "user", "content": "dados"}]


class TinySchema(BaseModel):
    title: str
    value: int


def test_strip_markdown_fences_variants():
    assert strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_markdown_fences('{"a": 1}') == '{"a": 1}'
    assert strip_markdown_fences("texto normal") == "texto normal"


def test_complete_json_valid_first_try(agnes):
    agnes.enqueue(agnes.ok('{"title": "T", "value": 3}'))
    result = run(complete_json("profile_extraction", _messages(), TinySchema))
    assert result.title == "T"
    assert result.value == 3
    assert agnes.call_count == 1


def test_complete_json_accepts_fenced_json(agnes):
    agnes.enqueue(agnes.ok('```json\n{"title": "T", "value": 1}\n```'))
    result = run(complete_json("profile_extraction", _messages(), TinySchema))
    assert result.title == "T"


def test_complete_json_single_corrective_retry_then_success(agnes):
    agnes.enqueue(agnes.ok("isto não é json"), agnes.ok('{"title": "OK", "value": 2}'))
    result = run(complete_json("profile_extraction", _messages(), TinySchema))
    assert result.title == "OK"
    assert agnes.call_count == 2

    retry_body = agnes.bodies[1]
    roles = [m["role"] for m in retry_body["messages"]]
    assert roles[-1] == "user", "Retry deve terminar com instrução corretiva"
    assert "JSON" in retry_body["messages"][-1]["content"]
    assert retry_body["temperature"] <= 0.3, "Retry corretivo deve reduzir temperatura"


def test_complete_json_fails_after_single_retry(agnes):
    agnes.enqueue(agnes.ok("lixo 1"), agnes.ok("lixo 2"), agnes.ok('{"title": "n", "value": 1}'))
    with pytest.raises(AgnesInvalidOutputError):
        run(complete_json("profile_extraction", _messages(), TinySchema))
    assert agnes.call_count == 2, "Apenas UMA nova tentativa é permitida para JSON inválido"


def test_complete_json_schema_violation_triggers_retry(agnes):
    # Valid JSON, wrong shape → still one corrective retry.
    agnes.enqueue(agnes.ok('{"errado": true}'), agnes.ok('{"title": "certo", "value": 9}'))
    result = run(complete_json("profile_extraction", _messages(), TinySchema))
    assert result.value == 9
    assert agnes.call_count == 2


def test_complete_json_dict_requires_object(agnes):
    agnes.enqueue(agnes.ok('[1, 2, 3]'), agnes.ok('{"ok": 1}'))
    result = run(complete_json_dict("prompt_generation", _messages()))
    assert result == {"ok": 1}
    assert agnes.call_count == 2


# --- Output schemas ----------------------------------------------------------


def test_akashic_schema_clamps_and_coerces():
    meta = AkashicMetadata.model_validate(
        {"mood": "inventado", "emotionalIntensity": 7, "keyInsight": 123}
    )
    assert meta.mood == "sereno"
    assert meta.emotionalIntensity == 1.0
    assert meta.keyInsight == ""


def test_ai_tool_schema_requires_title_and_snippet():
    with pytest.raises(ValidationError):
        AIToolResult.model_validate({"tags": []})

    ok = AIToolResult.model_validate({
        "title": " Título ",
        "snippet": "Um parágrafo.",
        "tags": ["a"] * 20,
        "mood": "qualquer",
        "emotionalIntensity": "0.83",
        "keyInsight": "x" * 900,
    })
    assert ok.title == "Título"
    assert len(ok.tags) == 8
    assert ok.mood == "sereno"
    assert ok.emotionalIntensity == 0.83
    assert len(ok.keyInsight) == 500


def test_profile_update_caps_themes():
    update = ProfileUpdate.model_validate({
        "personality_summary": "s",
        "emotional_state": "e",
        "recurring_themes": [f"tema {i}" for i in range(20)],
        "spiritual_progress": "p",
    })
    assert len(update.recurring_themes) == 8


def test_profile_update_missing_field_is_invalid():
    with pytest.raises(ValidationError):
        ProfileUpdate.model_validate({"personality_summary": "só um campo"})
