from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class SemanticProfile(BaseModel):
    keywords: list[str] = []
    emotionalTarget: str = Field(default="", max_length=50)
    emotionalOutcome: str = Field(default="", max_length=50)
    depthLevel: str = Field(default="journaling", max_length=30)


class AIConfig(BaseModel):
    analysisInstruction: str = Field(default="", max_length=2000)
    followUpSuggestions: list[str] = []


class ReflectionCreate(BaseModel):
    id: str = Field(max_length=100)
    isSystem: bool = True
    categoryId: str = Field(max_length=50)
    title: str = Field(max_length=200)
    description: str = Field(max_length=2000)
    guidingQuestions: list[str] = []
    scriptureReferences: list[str] = []
    reflection: str = Field(default="", max_length=8000)
    order: int = 0
    estimatedMinutes: int = 5
    semanticProfile: SemanticProfile | None = None
    aiConfig: AIConfig | None = None
    embeddingPayload: str | None = Field(default=None, max_length=10000)


class UserAnswer(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), max_length=100)
    reflectionId: str = Field(max_length=100)
    content: str = Field(max_length=8000, min_length=1)
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    message: str = Field(max_length=4000, min_length=1)
    sessionId: str | None = Field(default=None, max_length=100)
    reflectionId: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    response: str = Field(max_length=8000)
    sessionId: str = Field(max_length=100)
    sessionTitle: str | None = Field(default=None, max_length=500)
    model: str = Field(max_length=100)
    contextSources: int
    followUp: list[str] = []


class SessionInfo(BaseModel):
    sessionId: str = Field(max_length=100)
    userId: str = Field(max_length=128)
    title: str = Field(default="", max_length=500)
    reflectionId: str | None = Field(default=None, max_length=100)
    turnCount: int = 0
    createdAt: datetime
    lastActivity: datetime
    active: bool = True


VALID_CATEGORIES = {
    "gratitude", "faith", "challenges", "self_knowledge", "relationships", "purpose",
}

VALID_EMOTIONAL_TARGETS = {
    "anxiety", "restlessness", "guilt", "sadness", "anger",
    "doubt", "loneliness", "overwhelm", "fear", "shame", "neutral",
}

VALID_EMOTIONAL_OUTCOMES = {
    "peace", "contentment", "forgiveness", "hope", "gratitude",
    "courage", "connection", "clarity", "self_compassion", "joy", "trust",
}

VALID_DEPTH_LEVELS = {"quick_thought", "journaling", "deep_reflection"}


class UserProfile(BaseModel):
    user_id: str = Field(max_length=128)
    display_name: str = Field(default="", max_length=200)
    gender: str = Field(default="", max_length=20)
    total_xp: int = 0
    current_level: int = 1
    current_streak: int = 0
    personality_summary: str = Field(default="", max_length=2000)
    emotional_state: str = Field(default="", max_length=200)
    recurring_themes: list[str] = []
    spiritual_progress: str = Field(default="", max_length=2000)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    conversation_count: int = 0


class PromptGenerateRequest(BaseModel):
    title: str = Field(max_length=200)
    description: str = Field(max_length=2000)
    categoryId: str = Field(max_length=50)


class PromptGenerateResponse(BaseModel):
    guidingQuestions: list[str] = Field(default_factory=list)
    scriptureReferences: list[str] = Field(default_factory=list)
    reflection: str = Field(default="Reflita sobre este tema com calma e escreva seus pensamentos.", max_length=8000)
    estimatedMinutes: int = 5
    semanticProfile: SemanticProfile = Field(default_factory=SemanticProfile)
    aiConfig: AIConfig = Field(default_factory=AIConfig)
    embeddingPayload: str | None = Field(default=None, max_length=10000)


class AIToolRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class AIToolResponse(BaseModel):
    id: str = Field(max_length=100)
    title: str = Field(max_length=500)
    snippet: str = Field(max_length=8000)
    tags: list[str] = []
    date: datetime
    tool: str = Field(max_length=50)
    mood: str | None = Field(default=None, max_length=30)
    emotionalIntensity: float | None = Field(default=None, ge=0.0, le=1.0)
    keyInsight: str | None = Field(default=None, max_length=500)
