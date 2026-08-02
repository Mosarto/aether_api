"""Pydantic schemas for structured LLM outputs.

Structure is strict (required fields, correct types) so a malformed answer
triggers the single corrective retry in app.llm. Cosmetic fields (mood enum,
intensity range) coerce/clamp instead of failing, to avoid paid retries over
details the API can safely normalize.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import MOOD_VALUES

Mood = Literal["sereno", "ansioso", "esperançoso", "catártico", "melancólico", "empoderado"]


def _coerce_mood(value: object) -> str:
    return value if isinstance(value, str) and value in MOOD_VALUES else "sereno"


def _clamp_intensity(value: object, default: float = 0.5) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 2)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class ProfileUpdate(BaseModel):
    """Merged user profile produced by the profile-extraction use case."""

    personality_summary: str = Field(max_length=2000)
    emotional_state: str = Field(max_length=400)
    recurring_themes: list[str]
    spiritual_progress: str = Field(max_length=2000)

    @field_validator("recurring_themes", mode="after")
    @classmethod
    def _cap_themes(cls, value: list[str]) -> list[str]:
        return [str(t).strip()[:80] for t in value if str(t).strip()][:8]


class AkashicMetadata(BaseModel):
    """Emotional metadata extracted from a session summary."""

    mood: Mood
    emotionalIntensity: float
    keyInsight: str = Field(max_length=500)

    @field_validator("mood", mode="before")
    @classmethod
    def _mood(cls, value: object) -> str:
        return _coerce_mood(value)

    @field_validator("emotionalIntensity", mode="before")
    @classmethod
    def _intensity(cls, value: object) -> float:
        return _clamp_intensity(value)

    @field_validator("keyInsight", mode="before")
    @classmethod
    def _insight(cls, value: object) -> str:
        return value.strip()[:500] if isinstance(value, str) else ""


class AIToolResult(BaseModel):
    """Shared output of the dream/aura/stoic/sync tools."""

    title: str = Field(min_length=1, max_length=500)
    snippet: str = Field(min_length=1, max_length=8000)
    tags: list[str] = Field(default_factory=list)
    mood: Mood = "sereno"
    emotionalIntensity: float = 0.5
    keyInsight: str = Field(default="", max_length=500)

    @field_validator("title", "snippet", mode="before")
    @classmethod
    def _trim_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("tags", mode="after")
    @classmethod
    def _cap_tags(cls, value: list[str]) -> list[str]:
        return [str(t).strip()[:60] for t in value if str(t).strip()][:8]

    @field_validator("mood", mode="before")
    @classmethod
    def _mood(cls, value: object) -> str:
        return _coerce_mood(value)

    @field_validator("emotionalIntensity", mode="before")
    @classmethod
    def _intensity(cls, value: object) -> float:
        return _clamp_intensity(value)

    @field_validator("keyInsight", mode="before")
    @classmethod
    def _insight(cls, value: object) -> str:
        return value.strip()[:500] if isinstance(value, str) else ""
