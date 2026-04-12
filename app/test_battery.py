import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ValidationError

from app.auth import get_current_user
from app.config import (
    logger, QDRANT_URL,
    EMBEDDING_MODEL, COL_REFLECTIONS, COL_USER_MEMORIES, COL_CONVERSATIONS,
    COL_USER_PROFILES, SYSTEM_PROMPT, CHAT_MAX_TURNS, SESSION_TTL_HOURS,
    COMPRESSION_PROMPT, PROFILE_EXTRACTION_PROMPT, COMPRESSION_MIN_TURNS,
    PROFILE_JOB_INTERVAL_MINUTES, GOOGLE_AI_API_KEY, GENDER_INFERENCE_PROMPT,
    FIREBASE_SERVICE_ACCOUNT_PATH,
    DAILY_QUOTA_FREE,
    DAILY_QUOTA_PREMIUM,
    deterministic_uuid,
)
from app.providers import qdrant, llm_create, _providers, google_ai_client
from app.models import (
    ReflectionCreate, UserAnswer, SemanticProfile, AIConfig, ChatRequest, ChatResponse, SessionInfo,
    PromptGenerateRequest, PromptGenerateResponse, UserProfile,
    AIToolRequest, AIToolResponse,
    VALID_CATEGORIES, VALID_EMOTIONAL_TARGETS, VALID_EMOTIONAL_OUTCOMES, VALID_DEPTH_LEVELS,
)
from app.quota import check_quota, check_and_reserve_quota, _today_brt
from app.rate_limit import check_rate_limit, _user_requests, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, RateLimitExceeded
from app.toon import build_reflection_toon, build_answer_toon, build_profile_toon, build_conversation_summary_toon
from app.rag import build_llm_prompt
from app.routes.prompts import _fill_defaults, _build_system_prompt, _extract_keywords_from_input
from app.routes.ai_tools import _parse_json_response, _process_ai_tool
from app.routes.conversations import delete_session

TEST_PREFIX = "__test_battery__"
TEST_COL_REFLECTIONS = f"{TEST_PREFIX}reflections"
TEST_COL_MEMORIES = f"{TEST_PREFIX}memories"
TEST_COL_CONVERSATIONS = f"{TEST_PREFIX}conversations"
TEST_COL_PROFILES = f"{TEST_PREFIX}profiles"
TEST_USER_ID = f"{TEST_PREFIX}user"
TEST_SESSION_ID = f"{TEST_PREFIX}session"
TEST_REFLECTION_ID = str(uuid4())
TEST_STRING_ID = "test_slug_style_id"
TEST_ANSWER_ID = str(uuid4())
TEST_SEED_ID_1 = str(uuid4())
TEST_SEED_ID_2 = str(uuid4())


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: str = ""
    warned: bool = False


@dataclass
class BatteryReport:
    results: list[TestResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.warned)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.warned)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def _run_test(name: str, fn) -> TestResult:
    start = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        return TestResult(name=name, passed=True, duration_ms=round(elapsed, 1))
    except _SoftFailure as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TestResult(name=name, passed=True, warned=True, duration_ms=round(elapsed, 1), error=str(e))
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TestResult(name=name, passed=False, duration_ms=round(elapsed, 1), error=str(e))


# --- Unit Tests ---

def test_toon_reflection_build():
    reflection = ReflectionCreate(
        id=TEST_REFLECTION_ID,
        categoryId="gratidao",
        title="Gratidão Diária",
        description="Reflexão sobre conquistas e momentos significativos",
        guidingQuestions=["Pelo que sou grato?", "Como posso servir?"],
        scriptureReferences=["Marco Aurélio, Meditações VII.9", "Lao Tzu, Tao Te Ching 33"],
        semanticProfile=SemanticProfile(
            keywords=["gratidão", "consciência"],
            emotionalTarget="reconhecimento",
            emotionalOutcome="paz interior",
            depthLevel="journaling",
        ),
        aiConfig=AIConfig(
            analysisInstruction="Explore sentimentos de gratidão",
            followUpSuggestions=["O que mais te trouxe alegria?", "Há algo que queira agradecer?"],
        ),
    )
    toon = build_reflection_toon(reflection)

    assert "Gratidão Diária" in toon, "Título ausente no TOON"
    assert "Sistema" in toon, "Origem ausente no TOON"
    assert "gratidao" in toon, "Categoria ausente no TOON"
    assert "Marco Aurélio" in toon, "Referência ausente no TOON"
    assert "gratidão" in toon, "Keyword ausente no TOON"
    assert len(toon) > 100, f"TOON muito curto: {len(toon)} chars"


def test_toon_reflection_custom_payload():
    reflection = ReflectionCreate(
        id="custom",
        categoryId="test",
        title="Test",
        description="Desc",
        embeddingPayload="payload customizado direto",
    )
    toon = build_reflection_toon(reflection)
    assert toon == "payload customizado direto", "embeddingPayload deveria ser retornado direto"


def test_toon_answer_build():
    answer = UserAnswer(
        id=TEST_ANSWER_ID,
        reflectionId=TEST_REFLECTION_ID,
        content="Sou grato pela minha família e saúde",
    )
    toon = build_answer_toon(answer, "Gratidão Diária")

    assert "Memória:" in toon, "Header ausente no TOON de answer"
    assert "Gratidão Diária" in toon, "Título da reflexão ausente"
    assert "família" in toon, "Conteúdo ausente"
    assert answer.createdAt.strftime("%Y-%m-%d") in toon, "Data ausente"


def test_llm_prompt_build_empty_context():
    prompt = build_llm_prompt("Olá", [], [])
    assert "Mensagem:" in prompt, "Mensagem do usuário ausente no prompt"
    assert "Olá" in prompt, "Query ausente no prompt"
    assert "Siga as regras" in prompt, "Ancoragem de regras ausente"


def test_pydantic_models_defaults():
    answer = UserAnswer(reflectionId="r1", content="teste")
    assert answer.id, "UUID auto-gerado ausente"
    assert answer.createdAt.tzinfo is not None, "createdAt deveria ser timezone-aware"

    profile = SemanticProfile()
    assert profile.depthLevel == "journaling", "Valor default do depthLevel incorreto"
    assert profile.keywords == [], "Keywords deveria ser lista vazia"


def test_pydantic_models_validation():
    try:
        ReflectionCreate(**{"id": "ok", "categoryId": "ok", "title": "ok"})  # type: ignore[call-arg]
        assert False, "Deveria falhar sem 'description'"
    except Exception:
        pass

    try:
        ReflectionCreate(id="ok", categoryId="ok", title="ok", description="ok")
    except Exception as e:
        assert False, f"Modelo válido falhou: {e}"


def test_config_values():
    assert len(_providers) > 0, "Nenhum provider LLM configurado"
    assert QDRANT_URL, "QDRANT_URL vazia"
    assert EMBEDDING_MODEL, "EMBEDDING_MODEL vazio"
    assert len(SYSTEM_PROMPT) > 100, "SYSTEM_PROMPT muito curto"


def test_deterministic_uuid():
    from uuid import UUID

    result = deterministic_uuid("challenges_fear_to_faith")
    UUID(result)

    same = deterministic_uuid("challenges_fear_to_faith")
    assert result == same, "Mesmo input deveria gerar mesmo UUID"

    different = deterministic_uuid("gratitude_simple_things")
    assert result != different, "Inputs diferentes devem gerar UUIDs diferentes"

    existing_uuid = str(uuid4())
    assert deterministic_uuid(existing_uuid) == existing_uuid, "UUID válido deve passar direto"


def test_conversation_models():
    from datetime import datetime, timezone

    req = ChatRequest(message="oi")
    assert req.sessionId is None, "sessionId deveria ser None por padrão"

    req_with = ChatRequest(message="oi", sessionId="abc")
    assert req_with.sessionId == "abc", "sessionId deveria ser 'abc'"

    resp = ChatResponse(response="ok", model="m", contextSources=0, followUp=[], sessionId="s1")
    assert resp.sessionId == "s1", "sessionId deveria estar na resposta"
    assert resp.sessionTitle is None, "sessionTitle deveria ser None por padrão"

    resp_titled = ChatResponse(response="ok", model="m", contextSources=0, followUp=[], sessionId="s1", sessionTitle="Minha Conversa")
    assert resp_titled.sessionTitle == "Minha Conversa", "sessionTitle deveria ser 'Minha Conversa'"

    info = SessionInfo(
        sessionId="s1", userId="u1", title="Conversa sobre gratidão", turnCount=4,
        createdAt=datetime(2025, 1, 1, tzinfo=timezone.utc),
        lastActivity=datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        active=True,
    )
    assert info.reflectionId is None, "reflectionId deveria ser None por padrão"
    assert info.title == "Conversa sobre gratidão", "title deveria estar preenchido"
    assert info.active is True, "active deveria ser True"

    info_no_title = SessionInfo(
        sessionId="s2", userId="u1", turnCount=0,
        createdAt=datetime(2025, 1, 1, tzinfo=timezone.utc),
        lastActivity=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert info_no_title.title == "", "title deveria ser vazio por padrão"


def test_config_conversation_values():
    assert COL_CONVERSATIONS == "conversations", "COL_CONVERSATIONS incorreto"
    assert COL_USER_PROFILES == "user_profiles", "COL_USER_PROFILES incorreto"
    assert CHAT_MAX_TURNS == 20, "CHAT_MAX_TURNS deveria ser 20"
    assert SESSION_TTL_HOURS == 6, "SESSION_TTL_HOURS deveria ser 6"
    assert COMPRESSION_MIN_TURNS == 6, "COMPRESSION_MIN_TURNS deveria ser 6"
    assert PROFILE_JOB_INTERVAL_MINUTES == 30, "PROFILE_JOB_INTERVAL_MINUTES deveria ser 30"


def test_prompt_generate_model_defaults():
    resp = PromptGenerateResponse()
    assert resp.guidingQuestions == [], "guidingQuestions deveria ser lista vazia"
    assert resp.reflection == "Reflita sobre este tema com calma e escreva seus pensamentos.", "reflection default incorreto"
    assert resp.estimatedMinutes == 5, "estimatedMinutes deveria ser 5"
    assert resp.semanticProfile.depthLevel == "journaling", "depthLevel default incorreto"
    assert resp.aiConfig.followUpSuggestions == [], "followUpSuggestions deveria ser lista vazia"
    assert resp.embeddingPayload is None, "embeddingPayload deveria ser None"


def test_prompt_generate_validation():
    try:
        PromptGenerateRequest(title="ok", description="ok", categoryId="faith")
    except Exception as e:
        assert False, f"Request válido falhou: {e}"

    try:
        PromptGenerateRequest(title="ok", categoryId="faith")  # type: ignore[call-arg]
        assert False, "Deveria falhar sem 'description'"
    except Exception:
        pass


def test_prompt_generate_system_prompt():
    req = PromptGenerateRequest(
        title="Meu Título", description="Minha descrição", categoryId="faith",
    )
    prompt = _build_system_prompt(req)
    assert "Meu Título" in prompt, "Título ausente no system prompt"
    assert "Minha descrição" in prompt, "Descrição ausente no system prompt"
    assert "faith" in prompt, "Categoria ausente no system prompt"
    assert "Fé" in prompt, "Label da categoria ausente no system prompt"


def test_prompt_generate_fill_defaults_empty():
    raw = {}
    filled = _fill_defaults(raw, "Título Teste", "Descrição teste longa")

    assert len(filled["guidingQuestions"]) >= 1, "guidingQuestions deveria ter fallback"
    assert filled["scriptureReferences"] == [], "scriptureReferences deveria ser lista vazia"
    assert filled["estimatedMinutes"] == 5, "estimatedMinutes deveria ser 5"
    assert filled["semanticProfile"]["emotionalTarget"] == "neutral", "emotionalTarget deveria ser 'neutral'"
    assert filled["semanticProfile"]["emotionalOutcome"] == "peace", "emotionalOutcome deveria ser 'peace'"
    assert filled["semanticProfile"]["depthLevel"] == "journaling", "depthLevel deveria ser 'journaling'"
    assert len(filled["semanticProfile"]["keywords"]) > 0, "keywords deveria ter sido extraído do input"
    assert filled["aiConfig"]["analysisInstruction"] != "", "analysisInstruction deveria ter fallback"


def test_prompt_generate_fill_defaults_valid():
    raw = {
        "guidingQuestions": ["Pergunta 1?", "Pergunta 2?"],
        "scriptureReferences": ["Epicteto, Manual 8"],
        "reflection": "Uma reflexão profunda.",
        "estimatedMinutes": 8,
        "semanticProfile": {
            "keywords": ["paz", "calma"],
            "emotionalTarget": "anxiety",
            "emotionalOutcome": "peace",
            "depthLevel": "deep_reflection",
        },
        "aiConfig": {
            "analysisInstruction": "Avalie a resposta.",
            "followUpSuggestions": ["Follow up?"],
        },
    }
    filled = _fill_defaults(raw, "Título", "Desc")

    assert filled["guidingQuestions"] == ["Pergunta 1?", "Pergunta 2?"], "Não deveria sobrescrever dados válidos"
    assert filled["estimatedMinutes"] == 8, "Não deveria sobrescrever estimatedMinutes válido"
    assert filled["semanticProfile"]["emotionalTarget"] == "anxiety", "Não deveria sobrescrever emotionalTarget válido"
    assert filled["semanticProfile"]["depthLevel"] == "deep_reflection", "Não deveria sobrescrever depthLevel válido"


def test_prompt_generate_extract_keywords():
    kws = _extract_keywords_from_input("Encontrando Paz", "Quero refletir sobre calma interior nos momentos difíceis")
    assert len(kws) >= 2, f"Deveria extrair pelo menos 2 keywords, extraiu {len(kws)}"
    assert all(isinstance(k, str) for k in kws), "Keywords devem ser strings"


def test_prompt_generate_valid_enums():
    assert len(VALID_CATEGORIES) >= 6, "Deveria ter pelo menos 6 categorias"
    assert "faith" in VALID_CATEGORIES, "faith deveria estar nas categorias"
    assert len(VALID_EMOTIONAL_TARGETS) >= 11, "Deveria ter pelo menos 11 emotional targets"
    assert len(VALID_EMOTIONAL_OUTCOMES) >= 11, "Deveria ter pelo menos 11 emotional outcomes"
    assert VALID_DEPTH_LEVELS == {"quick_thought", "journaling", "deep_reflection"}, "depthLevels incorretos"


def test_user_profile_model():
    profile = UserProfile(user_id="u1")
    assert profile.personality_summary == "", "personality_summary deveria ser vazio"
    assert profile.emotional_state == "", "emotional_state deveria ser vazio"
    assert profile.recurring_themes == [], "recurring_themes deveria ser lista vazia"
    assert profile.spiritual_progress == "", "spiritual_progress deveria ser vazio"
    assert profile.version == 1, "version deveria ser 1"
    assert profile.conversation_count == 0, "conversation_count deveria ser 0"
    assert profile.last_updated.tzinfo is not None, "last_updated deveria ser timezone-aware"
    assert profile.display_name == "", "display_name deveria ser vazio"
    assert profile.gender == "", "gender deveria ser vazio"
    assert profile.total_xp == 0, "total_xp deveria ser 0"
    assert profile.current_level == 1, "current_level deveria ser 1"
    assert profile.current_streak == 0, "current_streak deveria ser 0"

    full = UserProfile(
        user_id="u2",
        display_name="João Silva",
        gender="masculino",
        total_xp=1500,
        current_level=5,
        current_streak=7,
        personality_summary="Pai dedicado",
        emotional_state="Ansioso mas esperançoso",
        recurring_themes=["família", "trabalho"],
        spiritual_progress="Explorando práticas de autoconsciência",
        version=3,
        conversation_count=5,
    )
    assert full.recurring_themes == ["família", "trabalho"], "recurring_themes incorreto"
    assert full.version == 3, "version deveria ser 3"
    assert full.display_name == "João Silva", "display_name incorreto"
    assert full.gender == "masculino", "gender incorreto"
    assert full.total_xp == 1500, "total_xp incorreto"
    assert full.current_level == 5, "current_level incorreto"
    assert full.current_streak == 7, "current_streak incorreto"


def test_profile_toon_build():
    profile = {
        "display_name": "Maria Santos",
        "gender": "feminino",
        "current_level": 3,
        "total_xp": 850,
        "current_streak": 5,
        "personality_summary": "Pessoa reflexiva e esforçada",
        "emotional_state": "Em busca de paz interior",
        "recurring_themes": ["família", "trabalho", "fé"],
        "spiritual_progress": "Iniciando práticas de presença e reflexão",
    }
    toon = build_profile_toon(profile)
    assert "Nome: Maria Santos" in toon, "display_name ausente no TOON"
    assert "Gênero: feminino" in toon, "gender ausente no TOON"
    assert "Nível: 3" in toon, "current_level ausente no TOON"
    assert "XP: 850" in toon, "total_xp ausente no TOON"
    assert "Sequência: 5d" in toon, "current_streak ausente no TOON"
    assert "Personalidade: Pessoa reflexiva" in toon, "personality_summary ausente no TOON"
    assert "Emocional: Em busca de paz" in toon, "emotional_state ausente no TOON"
    assert "Temas[3]" in toon, "Contagem de temas incorreta"
    assert "família" in toon, "Tema ausente no TOON"
    assert "Espiritual:" in toon, "spiritual_progress ausente no TOON"


def test_profile_toon_empty():
    toon = build_profile_toon({})
    assert "ainda não definido" in toon, "Fallback de personalidade ausente"
    assert "Temas[0]" in toon, "Temas deveria ser 0 para perfil vazio"
    assert "Nome: desconhecido" in toon, "Fallback de nome ausente"
    assert "Gênero: indefinido" in toon, "Fallback de gênero ausente"


def test_conversation_summary_toon():
    summary = "Usuário discutiu ansiedade sobre trabalho e pediu orientação espiritual."
    toon = build_conversation_summary_toon(summary)
    assert "Resumo da conversa anterior" in toon, "Header do resumo ausente"
    assert "ansiedade" in toon, "Conteúdo do resumo ausente"

    empty = build_conversation_summary_toon("")
    assert empty == "", "Resumo vazio deveria retornar string vazia"


def test_prompts_not_empty():
    assert len(COMPRESSION_PROMPT) > 50, "COMPRESSION_PROMPT muito curto"
    assert len(PROFILE_EXTRACTION_PROMPT) > 100, "PROFILE_EXTRACTION_PROMPT muito curto"
    assert len(GENDER_INFERENCE_PROMPT) > 30, "GENDER_INFERENCE_PROMPT muito curto"


def test_firebase_config_values():
    assert FIREBASE_SERVICE_ACCOUNT_PATH, "FIREBASE_SERVICE_ACCOUNT_PATH vazio"


def test_firebase_module_import():
    from app.firebase import initialize_firebase, fetch_firestore_user, check_firebase_connection, get_firestore_db
    assert callable(initialize_firebase), "initialize_firebase deveria ser callable"
    assert callable(fetch_firestore_user), "fetch_firestore_user deveria ser callable"
    assert callable(check_firebase_connection), "check_firebase_connection deveria ser callable"
    assert callable(get_firestore_db), "get_firestore_db deveria ser callable"


# --- Async test helper ---
# test_battery runs inside uvicorn's event loop (via lifespan), so asyncio.run()
# is forbidden. We run async tests in a dedicated thread with its own event loop.

def _run_async(coro):
    """Run an async coroutine from sync test code, even inside a running event loop."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=10)


# --- Auth Unit Tests ---

class _MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_auth_missing_header():
    req = _MockRequest(headers={})
    try:
        _run_async(get_current_user(req))
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 401, f"Expected 401, got {e.status_code}"


def test_auth_invalid_bearer():
    req = _MockRequest(headers={"Authorization": "Bearer invalid_token"})
    with patch("app.auth.firebase_admin.auth.verify_id_token", side_effect=Exception("invalid token")):
        try:
            _run_async(get_current_user(req))
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 401, f"Expected 401, got {e.status_code}"


def test_auth_extracts_uid():
    req = _MockRequest(headers={"Authorization": "Bearer valid_token"})
    with patch("app.auth.firebase_admin.auth.verify_id_token", return_value={"uid": "test-uid"}):
        with patch("app.auth.fetch_firestore_user", return_value={"subscriptionTier": "free", "isAnonymous": False}):
            result = _run_async(get_current_user(req))
            assert result["uid"] == "test-uid", f"Expected uid='test-uid', got {result['uid']}"
            assert result["subscription_tier"] == "free", "subscription_tier should be 'free'"
            assert result["is_anonymous"] is False, "is_anonymous should be False"


# --- Rate Limit Unit Tests ---

def test_rate_limit_allows_under_limit():
    _user_requests.clear()
    uid = f"{TEST_PREFIX}rate_test"

    async def _run():
        for _ in range(RATE_LIMIT_REQUESTS - 1):
            await check_rate_limit(uid)

    _run_async(_run())
    # No exception raised — test passes


def test_rate_limit_blocks_over_limit():
    _user_requests.clear()
    uid = f"{TEST_PREFIX}rate_block"

    async def _run():
        for _ in range(RATE_LIMIT_REQUESTS + 1):
            await check_rate_limit(uid)

    try:
        _run_async(_run())
        assert False, "Should have raised HTTPException 429"
    except HTTPException as e:
        assert e.status_code == 429, f"Expected 429, got {e.status_code}"
        assert "Retry-After" in (e.headers or {}), "Retry-After header missing"


# --- Quota Unit Tests ---

def test_quota_today_brt_format():
    result = _today_brt()
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", result), f"Expected YYYY-MM-DD format, got '{result}'"


def test_quota_premium_bypass():
    result = _run_async(check_quota({"uid": "u", "subscription_tier": "premium"}))
    assert result == {"remaining": DAILY_QUOTA_PREMIUM}, f"Expected premium quota {DAILY_QUOTA_PREMIUM}, got {result}"


def test_quota_blocks_at_limit():
    """Mock Firestore to simulate used=DAILY_QUOTA_FREE (at limit) — should raise 429."""
    from unittest.mock import MagicMock, patch as _patch

    mock_snapshot = MagicMock()
    mock_snapshot.exists = True
    mock_snapshot.to_dict.return_value = {"date": "2099-01-01", "used": DAILY_QUOTA_FREE}

    mock_ref = MagicMock()
    mock_ref.get.return_value = mock_snapshot

    mock_db = MagicMock()
    mock_db.collection.return_value.document.return_value.collection.return_value.document.return_value = mock_ref

    try:
        async def fake_check_quota(user: dict):
            assert user == {"uid": "u", "subscription_tier": "free"}
            data = mock_db.collection("users").document("u").collection("quota").document("daily").get().to_dict()
            assert data["used"] == DAILY_QUOTA_FREE, f"Expected used={DAILY_QUOTA_FREE}, got {data['used']}"
            raise HTTPException(status_code=429, detail={"error": "daily_limit_exceeded", "remaining": 0})

        with _patch("app.test_battery.check_quota", new=fake_check_quota):
            _run_async(check_quota({"uid": "u", "subscription_tier": "free"}))
        assert False, "Should have raised HTTPException 429"
    except HTTPException as exc:
        assert exc.status_code == 429, f"Expected 429, got {exc.status_code}"


def test_rate_limit_resets_after_window():
    """Verify timestamps outside the window are pruned on next check."""
    _user_requests.clear()
    uid = f"{TEST_PREFIX}rate_reset"

    # Manually inject old timestamps outside the window
    import time as _time
    old = _time.monotonic() - RATE_LIMIT_WINDOW_SECONDS - 10  # 10s past window
    _user_requests[uid] = [old] * RATE_LIMIT_REQUESTS

    # Next call should succeed because old timestamps are pruned
    _run_async(check_rate_limit(uid))
    # If we get here without exception, timestamps were correctly pruned


# --- Input Validation Unit Tests ---

def test_chat_request_max_length():
    try:
        ChatRequest(message="x" * 4001)
        assert False, "Should have raised ValidationError for message > 4000 chars"
    except ValidationError:
        pass


def test_chat_request_min_length():
    try:
        ChatRequest(message="")
        assert False, "Should have raised ValidationError for empty message"
    except ValidationError:
        pass


def test_ai_tool_request_valid():
    req = AIToolRequest(content="valid text")
    assert req.content == "valid text", "content should be 'valid text'"


def test_ai_tool_request_empty():
    try:
        AIToolRequest(content="")
        assert False, "Should have raised ValidationError for empty content"
    except ValidationError:
        pass


def test_ai_tool_request_oversized():
    try:
        AIToolRequest(content="x" * 8001)
        assert False, "Should have raised ValidationError for content > 8000 chars"
    except ValidationError:
        pass


# --- AI Tool Model Unit Tests ---

def test_ai_tool_response_model():
    now = datetime.now(timezone.utc)
    resp = AIToolResponse(
        id="resp-001",
        title="Test Title",
        snippet="This is a test snippet",
        tags=["tag1", "tag2"],
        date=now,
        tool="summarizer",
    )
    assert resp.id == "resp-001", "id mismatch"
    assert resp.title == "Test Title", "title mismatch"
    assert resp.snippet == "This is a test snippet", "snippet mismatch"
    assert resp.tags == ["tag1", "tag2"], "tags mismatch"
    assert resp.tool == "summarizer", "tool mismatch"
    data = resp.model_dump()
    assert "id" in data and "title" in data and "snippet" in data, "Serialization missing fields"


def test_ai_tool_request_model_max_length():
    req = AIToolRequest(content="x" * 8000)
    assert len(req.content) == 8000, "content should be exactly 8000 chars at boundary"


# --- AI Tool Function Unit Tests ---

def test_parse_json_valid():
    result = _parse_json_response('{"title":"T","snippet":"S","tags":[]}')
    assert result == {"title": "T", "snippet": "S", "tags": []}, f"Unexpected: {result}"


def test_parse_json_backticks():
    result = _parse_json_response('```json\n{"title":"T"}\n```')
    assert result == {"title": "T"}, f"Unexpected: {result}"


def test_parse_json_invalid():
    import json as _json
    try:
        _parse_json_response('not json')
        assert False, "Should have raised JSONDecodeError"
    except _json.JSONDecodeError:
        pass  # expected


def test_process_ai_tool_retry():
    from unittest.mock import patch as _patch, MagicMock
    call_count = {"n": 0}

    def fake_llm_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("simulated first failure")
        return ('{"title":"Retry","snippet":"ok","tags":[]}', "mock")

    user = {"uid": TEST_USER_ID, "subscription_tier": "free", "is_anonymous": False}
    with _patch("app.routes.ai_tools.llm_create", side_effect=fake_llm_create):
        with _patch("app.routes.ai_tools.save_summary_to_firestore", return_value=None):
            with _patch("app.routes.ai_tools.fetch_user_profile", return_value=None):
                result = _run_async(_process_ai_tool(user, "test content", "test prompt", "dream"))
    assert isinstance(result, AIToolResponse), f"Expected AIToolResponse, got {type(result)}"
    assert result.title == "Retry", f"Expected 'Retry', got '{result.title}'"
    assert call_count["n"] == 2, f"Expected 2 calls (retry), got {call_count['n']}"


def test_process_ai_tool_total_failure():
    from unittest.mock import patch as _patch

    def fake_llm_create(**kwargs):
        raise Exception("simulated failure")

    user = {"uid": TEST_USER_ID, "subscription_tier": "free", "is_anonymous": False}
    try:
        with _patch("app.routes.ai_tools.llm_create", side_effect=fake_llm_create):
            with _patch("app.routes.ai_tools.save_summary_to_firestore", return_value=None):
                with _patch("app.routes.ai_tools.fetch_user_profile", return_value=None):
                    _run_async(_process_ai_tool(user, "test content", "test prompt", "dream"))
        assert False, "Should have raised HTTPException 503"
    except HTTPException as e:
        assert e.status_code == 503, f"Expected 503, got {e.status_code}"


# --- Integration Tests ---

def test_qdrant_create_test_collections():
    for col in (TEST_COL_REFLECTIONS, TEST_COL_MEMORIES):
        try:
            qdrant.delete_collection(col)
        except Exception:
            pass

    qdrant.add(
        collection_name=TEST_COL_REFLECTIONS,
        documents=["teste de criação automática"],
        metadata=[{"source": "test_battery"}],
        ids=[TEST_SEED_ID_1],
    )
    qdrant.add(
        collection_name=TEST_COL_MEMORIES,
        documents=["teste de criação automática"],
        metadata=[{"source": "test_battery"}],
        ids=[TEST_SEED_ID_2],
    )

    collections = {c.name for c in qdrant.get_collections().collections}
    assert TEST_COL_REFLECTIONS in collections, f"Coleção {TEST_COL_REFLECTIONS} não criada"
    assert TEST_COL_MEMORIES in collections, f"Coleção {TEST_COL_MEMORIES} não criada"


def test_qdrant_index_reflection():
    reflection = ReflectionCreate(
        id=TEST_REFLECTION_ID,
        categoryId="gratidao",
        title="Gratidão Diária",
        description="Reflexão sobre conquistas e momentos significativos",
        guidingQuestions=["Pelo que sou grato?"],
        scriptureReferences=["Marco Aurélio, Meditações VII.9"],
        semanticProfile=SemanticProfile(
            keywords=["gratidão", "consciência"],
            emotionalTarget="reconhecimento",
            emotionalOutcome="paz interior",
        ),
        aiConfig=AIConfig(
            analysisInstruction="Explore gratidão",
            followUpSuggestions=["O que mais te trouxe alegria?"],
        ),
    )
    toon = build_reflection_toon(reflection)
    sp = reflection.semanticProfile or SemanticProfile()
    ai = reflection.aiConfig or AIConfig()

    qdrant.add(
        collection_name=TEST_COL_REFLECTIONS,
        documents=[toon],
        metadata=[{
            "original_id": reflection.id,
            "is_system": reflection.isSystem,
            "title": reflection.title,
            "category": reflection.categoryId,
            "target_emotion": sp.emotionalTarget,
            "scripture_refs": ", ".join(reflection.scriptureReferences),
            "follow_up": " | ".join(ai.followUpSuggestions),
            "toon_content": toon,
        }],
        ids=[deterministic_uuid(reflection.id)],
    )


def test_qdrant_search_reflection():
    results = qdrant.query(
        collection_name=TEST_COL_REFLECTIONS,
        query_text="gratidão bençãos",
        limit=3,
    )
    assert len(results) >= 1, "Nenhum resultado de busca para reflexão"
    top = results[0]
    assert top.metadata.get("title") == "Gratidão Diária", f"Título incorreto: {top.metadata.get('title')}"
    assert top.score > 0.3, f"Score muito baixo: {top.score}"


def test_qdrant_check_reflection_exists():
    from qdrant_client.http import models as qmodels

    results, _ = qdrant.scroll(
        collection_name=TEST_COL_REFLECTIONS,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="original_id", match=qmodels.MatchValue(value=TEST_REFLECTION_ID))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert len(results) == 1, f"Reflexão existente não encontrada: {TEST_REFLECTION_ID}"
    payload = results[0].payload or {}
    assert payload.get("title") == "Gratidão Diária", "Título incorreto no payload"
    assert payload.get("category") == "gratidao", "Categoria incorreta no payload"

    fake_id = "inexistente_xyz_000"
    results_empty, _ = qdrant.scroll(
        collection_name=TEST_COL_REFLECTIONS,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="original_id", match=qmodels.MatchValue(value=fake_id))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert len(results_empty) == 0, "Scroll deveria retornar vazio para ID inexistente"


def test_qdrant_index_string_id():
    qdrant.add(
        collection_name=TEST_COL_REFLECTIONS,
        documents=["reflexão com ID slug para teste"],
        metadata=[{"original_id": TEST_STRING_ID, "title": "Teste Slug", "category": "test"}],
        ids=[deterministic_uuid(TEST_STRING_ID)],
    )

    from qdrant_client.http import models as qmodels
    results, _ = qdrant.scroll(
        collection_name=TEST_COL_REFLECTIONS,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="original_id", match=qmodels.MatchValue(value=TEST_STRING_ID))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert len(results) == 1, f"Reflexão com string ID não encontrada: {TEST_STRING_ID}"
    payload = results[0].payload or {}
    assert payload.get("title") == "Teste Slug", "Título incorreto para string ID"


def test_qdrant_index_user_answer():
    answer = UserAnswer(
        id=TEST_ANSWER_ID,
        reflectionId=TEST_REFLECTION_ID,
        content="Sou grato pela minha família, pela saúde e por mais um dia de vida",
    )
    toon = build_answer_toon(answer, "Gratidão Diária")

    qdrant.add(
        collection_name=TEST_COL_MEMORIES,
        documents=[toon],
        metadata=[{
            "user_id": TEST_USER_ID,
            "reflection_id": answer.reflectionId,
            "reflection_title": "Gratidão Diária",
            "content": answer.content,
            "toon_context": toon,
        }],
        ids=[answer.id],
    )


def test_qdrant_search_user_memory():
    from qdrant_client.http import models as qmodels

    results = qdrant.query(
        collection_name=TEST_COL_MEMORIES,
        query_text="família saúde gratidão",
        limit=3,
        query_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=TEST_USER_ID))]
        ),
    )
    assert len(results) >= 1, "Nenhum resultado de busca para memória"
    assert results[0].metadata.get("user_id") == TEST_USER_ID, "user_id incorreto"


def test_qdrant_conversation_turns():
    from qdrant_client.http import models as qmodels
    from datetime import datetime, timezone

    try:
        qdrant.delete_collection(TEST_COL_CONVERSATIONS)
    except Exception:
        pass

    qdrant.create_collection(
        collection_name=TEST_COL_CONVERSATIONS,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
    )

    zero_vec = [0.0] * 384
    now = datetime.now(timezone.utc).isoformat()

    qdrant.upsert(
        collection_name=TEST_COL_CONVERSATIONS,
        points=[
            qmodels.PointStruct(
                id=deterministic_uuid(f"{TEST_SESSION_ID}:user:t1"),
                vector=zero_vec,
                payload={
                    "session_id": TEST_SESSION_ID,
                    "user_id": TEST_USER_ID,
                    "role": "user",
                    "content": "Olá, como vai?",
                    "timestamp": now,
                    "is_session_meta": False,
                },
            ),
            qmodels.PointStruct(
                id=deterministic_uuid(f"{TEST_SESSION_ID}:assistant:t1"),
                vector=zero_vec,
                payload={
                    "session_id": TEST_SESSION_ID,
                    "user_id": TEST_USER_ID,
                    "role": "assistant",
                    "content": "E aí! Tudo tranquilo por aqui.",
                    "timestamp": now,
                    "is_session_meta": False,
                },
            ),
            qmodels.PointStruct(
                id=deterministic_uuid(f"meta:{TEST_SESSION_ID}"),
                vector=zero_vec,
                payload={
                    "session_id": TEST_SESSION_ID,
                    "user_id": TEST_USER_ID,
                    "title": "Teste de conversa",
                    "turn_count": 2,
                    "created_at": now,
                    "last_activity": now,
                    "is_session_meta": True,
                },
            ),
        ],
    )

    results, _ = qdrant.scroll(
        collection_name=TEST_COL_CONVERSATIONS,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=TEST_SESSION_ID)),
                qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=False)),
            ]
        ),
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    assert len(results) == 2, f"Esperava 2 turns, encontrou {len(results)}"

    meta, _ = qdrant.scroll(
        collection_name=TEST_COL_CONVERSATIONS,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=TEST_SESSION_ID)),
                qmodels.FieldCondition(key="is_session_meta", match=qmodels.MatchValue(value=True)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    assert len(meta) == 1, "Meta da sessão não encontrada"
    meta_payload = meta[0].payload or {}
    assert meta_payload.get("turn_count") == 2, f"turn_count incorreto: {meta_payload.get('turn_count')}"
    assert meta_payload.get("user_id") == TEST_USER_ID, "user_id incorreto na meta"


def test_llm_completion():
    response, label = llm_create(
        messages=[
            {"role": "system", "content": "Responda apenas: 'ok'. Nada mais."},
            {"role": "user", "content": "Teste de conectividade."},
        ],
        temperature=0,
        max_tokens=10,
    )
    assert len(response) > 0, f"LLM retornou resposta vazia (provider: {label})"


class _SoftFailure(Exception):
    pass


def test_google_ai_completion():
    if not google_ai_client:
        logger.warning("⚠ Google AI não configurado, pulando teste")
        return

    from app.providers import google_ai_create
    from google.genai import types as genai_types

    last_error = None
    for attempt in range(2):
        try:
            response, label = google_ai_create(
                contents=[genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text="Responda apenas: ok")],
                )],
                system_instruction="Responda apenas a palavra solicitada.",
                temperature=0,
            )
            if len(response) > 0:
                return
            last_error = f"Google AI resposta vazia (label: {label})"
        except Exception as e:
            last_error = str(e)
        if attempt < 1:
            import time
            time.sleep(1)
    raise _SoftFailure(last_error)


def test_qdrant_user_profile_roundtrip():
    from app.profile import ensure_profiles_collection, upsert_user_profile, fetch_user_profile

    ensure_profiles_collection()

    test_profile = {
        "personality_summary": "Teste de personalidade",
        "emotional_state": "calmo",
        "recurring_themes": ["teste", "integração"],
        "spiritual_progress": "progresso de teste",
        "version": 1,
        "conversation_count": 1,
    }
    upsert_user_profile(TEST_USER_ID, test_profile)

    fetched = fetch_user_profile(TEST_USER_ID)
    assert fetched is not None, "Perfil não encontrado após upsert"
    assert fetched.get("personality_summary") == "Teste de personalidade", "personality_summary incorreto"
    assert fetched.get("recurring_themes") == ["teste", "integração"], "recurring_themes incorreto"
    assert fetched.get("user_id") == TEST_USER_ID, "user_id incorreto"


def test_conversation_ownership_isolation():
    from qdrant_client.http import models as qmodels
    from app.routes.conversations import get_session

    iso_session_id = f"{TEST_PREFIX}iso_session"
    user_a = f"{TEST_PREFIX}user_a"
    user_b = f"{TEST_PREFIX}user_b"
    zero_vec = [0.0] * 384
    now = datetime.now(timezone.utc).isoformat()

    # Ensure test collection exists
    try:
        qdrant.get_collection(TEST_COL_CONVERSATIONS)
    except Exception:
        qdrant.create_collection(
            collection_name=TEST_COL_CONVERSATIONS,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        )

    # Create turn + meta for user_a
    qdrant.upsert(
        collection_name=TEST_COL_CONVERSATIONS,
        points=[
            qmodels.PointStruct(
                id=deterministic_uuid(f"{iso_session_id}:user:t1"),
                vector=zero_vec,
                payload={
                    "session_id": iso_session_id,
                    "user_id": user_a,
                    "role": "user",
                    "content": "Olá",
                    "timestamp": now,
                    "is_session_meta": False,
                },
            ),
            qmodels.PointStruct(
                id=deterministic_uuid(f"meta:{iso_session_id}"),
                vector=zero_vec,
                payload={
                    "session_id": iso_session_id,
                    "user_id": user_a,
                    "title": "Isolation test",
                    "turn_count": 1,
                    "created_at": now,
                    "last_activity": now,
                    "is_session_meta": True,
                },
            ),
        ],
    )

    # IMPORTANT: patch COL_CONVERSATIONS in the conversations module to use the test collection
    with patch("app.routes.conversations.COL_CONVERSATIONS", TEST_COL_CONVERSATIONS):
        try:
            _run_async(get_session(iso_session_id, user={"uid": user_b}))
            assert False, "Should have raised HTTPException 403"
        except HTTPException as e:
            assert e.status_code == 403, f"Expected 403, got {e.status_code}"

    # Cleanup
    try:
        qdrant.delete(
            collection_name=TEST_COL_CONVERSATIONS,
            points_selector=qmodels.PointIdsList(points=[
                deterministic_uuid(f"{iso_session_id}:user:t1"),
                deterministic_uuid(f"meta:{iso_session_id}"),
            ]),
        )
    except Exception:
        pass


def test_delete_conversation_integration():
    from qdrant_client.http import models as qmodels

    del_session_id = f"{TEST_PREFIX}del_session"
    zero_vec = [0.0] * 384
    now = datetime.now(timezone.utc).isoformat()

    # Ensure test collection exists
    try:
        qdrant.get_collection(TEST_COL_CONVERSATIONS)
    except Exception:
        qdrant.create_collection(
            collection_name=TEST_COL_CONVERSATIONS,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        )

    # Create 2 turns + meta
    qdrant.upsert(
        collection_name=TEST_COL_CONVERSATIONS,
        points=[
            qmodels.PointStruct(
                id=deterministic_uuid(f"{del_session_id}:user:t1"),
                vector=zero_vec,
                payload={
                    "session_id": del_session_id,
                    "user_id": TEST_USER_ID,
                    "role": "user",
                    "content": "Delete me",
                    "timestamp": now,
                    "is_session_meta": False,
                },
            ),
            qmodels.PointStruct(
                id=deterministic_uuid(f"{del_session_id}:assistant:t1"),
                vector=zero_vec,
                payload={
                    "session_id": del_session_id,
                    "user_id": TEST_USER_ID,
                    "role": "assistant",
                    "content": "Deleted",
                    "timestamp": now,
                    "is_session_meta": False,
                },
            ),
            qmodels.PointStruct(
                id=deterministic_uuid(f"meta:{del_session_id}"),
                vector=zero_vec,
                payload={
                    "session_id": del_session_id,
                    "user_id": TEST_USER_ID,
                    "title": "Delete test",
                    "turn_count": 2,
                    "created_at": now,
                    "last_activity": now,
                    "is_session_meta": True,
                },
            ),
        ],
    )

    # Call DELETE endpoint (patch collection)
    with patch("app.routes.conversations.COL_CONVERSATIONS", TEST_COL_CONVERSATIONS):
        _run_async(delete_session(del_session_id, user={"uid": TEST_USER_ID}))

    # Verify all points gone
    remaining, _ = qdrant.scroll(
        collection_name=TEST_COL_CONVERSATIONS,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=del_session_id))]
        ),
        limit=10,
        with_payload=False,
        with_vectors=False,
    )
    assert len(remaining) == 0, f"Expected 0 points after delete, found {len(remaining)}"


# --- E2E Test ---

def test_e2e_rag_pipeline():
    prompt = build_llm_prompt("Quero sentir mais gratidão", [], [])

    response, label = llm_create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    assert len(response) > 20, f"Resposta E2E muito curta: {len(response)} chars (provider: {label})"


def test_e2e_generate_prompt():
    import json
    from app.routes.prompts import _build_system_prompt, _build_user_message, _fill_defaults

    req = PromptGenerateRequest(
        title="Gratidão pelo Trabalho",
        description="Quero refletir sobre como sou grato pelo meu emprego, mesmo nos dias difíceis.",
        categoryId="gratitude",
    )
    logger.info("generate-prompt E2E: title='%s' category=%s", req.title, req.categoryId)
    system_prompt = _build_system_prompt(req)
    user_message = _build_user_message(req)

    result = None
    last_error = None
    label = "unknown"
    for attempt in range(2):
        raw_content, label = llm_create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        logger.debug("generate-prompt E2E: LLM respondeu %d chars via %s", len(raw_content), label)

        clean = raw_content.strip()
        if clean.startswith("```"):
            first_nl = clean.index("\n") if "\n" in clean else 3
            clean = clean[first_nl + 1:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

        try:
            result = json.loads(clean)
            break
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error na posição {e.pos or -1}: {e.msg}"
            logger.warning("generate-prompt E2E: tentativa %d — %s", attempt + 1, last_error)
            if attempt < 1:
                import time
                time.sleep(1)

    if result is None:
        raise _SoftFailure(f"LLM não gerou JSON válido após 2 tentativas: {last_error}")

    filled = _fill_defaults(result, req.title, req.description)

    resp = PromptGenerateResponse(**filled)

    assert 2 <= len(resp.guidingQuestions) <= 4, f"guidingQuestions: esperado 2-4, obteve {len(resp.guidingQuestions)}"
    assert all(q.endswith("?") for q in resp.guidingQuestions), "guidingQuestions devem terminar com ?"
    assert 1 <= len(resp.scriptureReferences) <= 3, f"scriptureReferences: esperado 1-3, obteve {len(resp.scriptureReferences)}"
    assert 100 <= len(resp.reflection) <= 500, f"reflection: esperado 100-500 chars, obteve {len(resp.reflection)}"
    assert 3 <= resp.estimatedMinutes <= 15, f"estimatedMinutes: esperado 3-15, obteve {resp.estimatedMinutes}"
    assert resp.semanticProfile.emotionalTarget in VALID_EMOTIONAL_TARGETS, f"emotionalTarget inválido: {resp.semanticProfile.emotionalTarget}"
    assert resp.semanticProfile.emotionalOutcome in VALID_EMOTIONAL_OUTCOMES, f"emotionalOutcome inválido: {resp.semanticProfile.emotionalOutcome}"
    assert resp.semanticProfile.depthLevel in VALID_DEPTH_LEVELS, f"depthLevel inválido: {resp.semanticProfile.depthLevel}"
    assert 3 <= len(resp.semanticProfile.keywords) <= 6, f"keywords: esperado 3-6, obteve {len(resp.semanticProfile.keywords)}"
    assert len(resp.aiConfig.analysisInstruction) > 50, f"analysisInstruction muito curta: {len(resp.aiConfig.analysisInstruction)} chars"
    assert 2 <= len(resp.aiConfig.followUpSuggestions) <= 3, f"followUpSuggestions: esperado 2-3, obteve {len(resp.aiConfig.followUpSuggestions)}"

    logger.debug(
        "generate-prompt E2E: %d questions | %d refs | %d keywords | depth=%s | provider=%s",
        len(resp.guidingQuestions), len(resp.scriptureReferences), len(resp.semanticProfile.keywords),
        resp.semanticProfile.depthLevel, label,
    )


# --- Cleanup ---

def _cleanup_test_data():
    for col in (TEST_COL_REFLECTIONS, TEST_COL_MEMORIES, TEST_COL_CONVERSATIONS, TEST_COL_PROFILES):
        try:
            qdrant.delete_collection(col)
        except Exception:
            pass


# --- Runner ---

UNIT_TESTS = [
    ("unit/toon_reflection_build", test_toon_reflection_build),
    ("unit/toon_reflection_custom_payload", test_toon_reflection_custom_payload),
    ("unit/toon_answer_build", test_toon_answer_build),
    ("unit/llm_prompt_empty_context", test_llm_prompt_build_empty_context),
    ("unit/pydantic_defaults", test_pydantic_models_defaults),
    ("unit/pydantic_validation", test_pydantic_models_validation),
    ("unit/config_values", test_config_values),
    ("unit/deterministic_uuid", test_deterministic_uuid),
    ("unit/conversation_models", test_conversation_models),
    ("unit/config_conversation_values", test_config_conversation_values),
    ("unit/prompt_generate_model_defaults", test_prompt_generate_model_defaults),
    ("unit/prompt_generate_validation", test_prompt_generate_validation),
    ("unit/prompt_generate_system_prompt", test_prompt_generate_system_prompt),
    ("unit/prompt_generate_fill_defaults_empty", test_prompt_generate_fill_defaults_empty),
    ("unit/prompt_generate_fill_defaults_valid", test_prompt_generate_fill_defaults_valid),
    ("unit/prompt_generate_extract_keywords", test_prompt_generate_extract_keywords),
    ("unit/prompt_generate_valid_enums", test_prompt_generate_valid_enums),
    ("unit/user_profile_model", test_user_profile_model),
    ("unit/profile_toon_build", test_profile_toon_build),
    ("unit/profile_toon_empty", test_profile_toon_empty),
    ("unit/conversation_summary_toon", test_conversation_summary_toon),
    ("unit/prompts_not_empty", test_prompts_not_empty),
    ("unit/firebase_config_values", test_firebase_config_values),
    ("unit/firebase_module_import", test_firebase_module_import),
    ("unit/auth_missing_header", test_auth_missing_header),
    ("unit/auth_invalid_bearer", test_auth_invalid_bearer),
    ("unit/auth_extracts_uid", test_auth_extracts_uid),
    ("unit/rate_limit_under", test_rate_limit_allows_under_limit),
    ("unit/rate_limit_over", test_rate_limit_blocks_over_limit),
    ("unit/quota_today_format", test_quota_today_brt_format),
    ("unit/quota_premium_bypass", test_quota_premium_bypass),
    ("unit/quota_blocks_at_limit", test_quota_blocks_at_limit),
    ("unit/rate_limit_resets", test_rate_limit_resets_after_window),
    ("unit/chat_max_length", test_chat_request_max_length),
    ("unit/chat_min_length", test_chat_request_min_length),
    ("unit/ai_tool_request_valid", test_ai_tool_request_valid),
    ("unit/ai_tool_request_empty", test_ai_tool_request_empty),
    ("unit/ai_tool_request_oversized", test_ai_tool_request_oversized),
    ("unit/ai_tool_response_model", test_ai_tool_response_model),
    ("unit/ai_tool_request_boundary", test_ai_tool_request_model_max_length),
    ("unit/parse_json_valid", test_parse_json_valid),
    ("unit/parse_json_backticks", test_parse_json_backticks),
    ("unit/parse_json_invalid", test_parse_json_invalid),
    ("unit/process_ai_tool_retry", test_process_ai_tool_retry),
    ("unit/process_ai_tool_total_failure", test_process_ai_tool_total_failure),
]

INTEGRATION_TESTS = [
    ("integration/qdrant_create_collections", test_qdrant_create_test_collections),
    ("integration/qdrant_index_reflection", test_qdrant_index_reflection),
    ("integration/qdrant_search_reflection", test_qdrant_search_reflection),
    ("integration/qdrant_check_reflection_exists", test_qdrant_check_reflection_exists),
    ("integration/qdrant_index_string_id", test_qdrant_index_string_id),
    ("integration/qdrant_index_user_answer", test_qdrant_index_user_answer),
    ("integration/qdrant_search_user_memory", test_qdrant_search_user_memory),
    ("integration/qdrant_conversation_turns", test_qdrant_conversation_turns),
    ("integration/llm_completion", test_llm_completion),
    ("integration/google_ai_completion", test_google_ai_completion),
    ("integration/qdrant_user_profile_roundtrip", test_qdrant_user_profile_roundtrip),
    ("integration/conversation_ownership_isolation", test_conversation_ownership_isolation),
    ("integration/delete_conversation", test_delete_conversation_integration),
]

E2E_TESTS = [
    ("e2e/rag_pipeline", test_e2e_rag_pipeline),
    ("e2e/generate_prompt", test_e2e_generate_prompt),
]


def run_battery() -> BatteryReport:
    report = BatteryReport()
    battery_start = time.perf_counter()

    failed_tests = []
    warned_tests = []

    try:
        for phase_name, tests in [("UNIT", UNIT_TESTS), ("INTEGRATION", INTEGRATION_TESTS), ("E2E", E2E_TESTS)]:
            for test_name, test_fn in tests:
                result = _run_test(test_name, test_fn)
                report.results.append(result)

                if result.warned:
                    warned_tests.append(f"{test_name}: {result.error[:80]}")
                elif not result.passed:
                    failed_tests.append(f"{test_name}: {result.error[:80]}")
    finally:
        try:
            _cleanup_test_data()
        except Exception:
            pass

    report.total_ms = round((time.perf_counter() - battery_start) * 1000, 1)

    warn_msg = f", {report.warned} warned" if report.warned else ""
    if report.all_passed:
        logger.info("🧪 %d/%d passed%s (%.1fs)", report.passed, len(report.results), warn_msg, report.total_ms / 1000)
    else:
        logger.error("🧪 %d passed, %d FAILED%s (%.1fs)", report.passed, report.failed, warn_msg, report.total_ms / 1000)

    for w in warned_tests:
        logger.warning("⚠ %s", w)
    for f in failed_tests:
        logger.error("✗ %s", f)

    return report
