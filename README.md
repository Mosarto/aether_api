# ✨ Jornada Celestial — Backend

> *"E se você pudesse conversar com a sua melhor versão? Aquela que já superou tudo, que te entende porque É você — só que celestial?"*

Backend do app **Jornada Celestial**: uma plataforma de reflexão pessoal com IA que age como o **Eu Celestial** do usuário — sua versão mais elevada, luminosa e cheia de graça.

---

## 🧭 Visão Geral

```
Flutter App (Dart)
       │
       ▼
  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
  │  FastAPI     │────▶│  Qdrant      │     │  Cerebras       │
  │  (Backend)   │     │  (Memória    │     │  (LLM           │
  │              │◀────│   Vetorial)  │     │   Llama 3.3)    │
  │  :8000       │     │  :6333       │     │   cloud API     │
  └──────┬───────┘     └──────────────┘     └────────┬────────┘
         │                                           │
         ├───────────── RAG + TOON ──────────────────┘
         │
         ├──────── Firebase Firestore (cadastro do usuário)
         │
         └──────── Google AI (Gemini 3 Flash — chat)
```

**O fluxo:**
1. O app Flutter envia reflexões e respostas do usuário
2. O backend vetoriza tudo localmente (FastEmbed, zero custo de API)
3. Na conversa, busca memórias pessoais + reflexões relevantes via RAG
4. Carrega o histórico da sessão (até 20 turns) e deduplica contexto já usado
5. Monta um prompt compacto em formato TOON (economia de tokens)
6. Envia para o Cerebras (Llama 3.3 70B) que responde como o Eu Celestial do usuário
7. Salva os turns (user + assistant) e metadados da sessão no Qdrant

---

## 🚀 Stack

| Componente         | Tecnologia                                 | Papel                          |
|--------------------|--------------------------------------------|---------------------------------|
| **API**            | FastAPI 0.128 + Uvicorn                    | Servidor REST                   |
| **Embeddings**     | FastEmbed 0.7 (CPU local)                  | Vetorização sem custo de API    |
| **Modelo Embed.**  | `paraphrase-multilingual-MiniLM-L12-v2`   | 384 dimensões, 50+ idiomas     |
| **Banco Vetorial** | Qdrant (com API key)                       | Busca semântica + memória       |
| **LLM Chat**       | Google AI (Gemini 3 Flash Preview)         | Respostas do Eu Celestial       |
| **LLM Background** | Cerebras Llama 3.3 70B + Groq (failover)   | Compressão, perfil, títulos     |
| **Firebase**       | Firebase Admin SDK (Firestore)             | Cadastro do usuário             |
| **Deploy**         | Docker Compose + Dokploy                   | Infra unificada                 |
| **Frontend**       | Flutter (Dart)                             | App mobile                      |

---

## 📁 Estrutura do Projeto

```
backend/
├── main.py                      ← Orquestrador (35 linhas)
├── app/
│   ├── __init__.py
│   ├── config.py                ← Configuração centralizada, loggers, env vars, prompts
│   ├── providers.py             ← Clientes Qdrant + Cerebras/Groq + Google AI
│   ├── models.py                ← Modelos Pydantic (espelham Dart 1:1)
│   ├── toon.py                  ← Builders do formato TOON (reflexão, memória, perfil)
│   ├── rag.py                   ← Pipeline RAG (retrieval + prompt)
│   ├── profile.py               ← Gestão de perfil do usuário (Qdrant + LLM)
│   ├── firebase.py              ← Firebase Admin SDK (Firestore + cadastro)
│   ├── background.py            ← Background job (perfil a cada 30min)
│   ├── daily_verse.py           ← Cron daily verse (meia-noite BRT)
│   ├── startup.py               ← Health checks + test battery no boot
│   ├── test_battery.py          ← 37 testes automatizados (unit/integration/e2e)
│   └── routes/
│       ├── reflections.py       ← GET /reflections/{id}/exists, POST /reflections
│       ├── answers.py           ← POST /user-answers
│       ├── chat.py              ← POST /chat (Google AI + compressão + perfil)
│       ├── prompts.py           ← POST /generate-prompt (geração via LLM)
│       ├── daily_verse.py       ← POST /daily-verse/{user_id}
│       ├── conversations.py     ← GET /conversations, GET /conversations/{id}
│       └── health.py            ← GET /health
├── Dockerfile                   ← Python 3.12-slim, 1 worker
├── docker-compose.yml           ← API + Qdrant unificados
├── requirements.txt             ← Dependências fixadas
├── .env.example                 ← Template de variáveis
├── .github/
│   └── copilot-instructions.md  ← Regras do projeto para IA
├── .gitignore
└── .dockerignore
```

---

## ⚡ API — Endpoints

### `GET /reflections/{reflection_id}/exists` → `200`
Verifica se uma reflexão já está indexada no Qdrant. Útil para o front admin evitar retrabalho em caso de erro parcial.

**Response (encontrada):**
```json
{
  "exists": true,
  "id": "gratitude_simple_things",
  "title": "Gratidão pelas Coisas Simples",
  "category": "gratitude"
}
```

**Response (não encontrada):**
```json
{
  "exists": false,
  "id": "gratitude_simple_things"
}
```

**Busca:** `scroll` no Qdrant filtrando por `original_id` no metadata, sem vetores (`with_vectors=False`).

---

### `POST /reflections` → `201`
Indexa uma reflexão no Qdrant com formato TOON. Aceita `embeddingPayload` customizado ou gera automaticamente a partir dos campos.

**Request:**
```json
{
  "id": "gratitude_simple_things",
  "isSystem": true,
  "categoryId": "gratitude",
  "title": "Gratidão pelas Coisas Simples",
  "description": "Aprecie as bênçãos cotidianas",
  "guidingQuestions": ["Quais bênçãos simples você esquece de agradecer?"],
  "scriptureReferences": ["D&C 59:21", "Lucas 17:11-19"],
  "reflection": "A gratidão transforma o que temos em suficiente...",
  "order": 1,
  "estimatedMinutes": 5,
  "semanticProfile": {
    "keywords": ["simplicidade", "cotidiano", "contentamento"],
    "emotionalTarget": "restlessness",
    "emotionalOutcome": "contentment",
    "depthLevel": "quick_thought"
  },
  "aiConfig": {
    "analysisInstruction": "Confirme se o usuário mencionou bênçãos simples.",
    "followUpSuggestions": ["Qual pequena alegria passou despercebida hoje?"]
  }
}
```

**Response:**
```json
{
  "status": "indexed",
  "id": "gratitude_simple_things",
  "title": "Gratidão pelas Coisas Simples"
}
```

**Metadata indexada no Qdrant:** `original_id`, `is_system`, `title`, `category`, `description`, `target_emotion`, `outcome_emotion`, `depth_level`, `keywords`, `scripture_refs`, `analysis_instruction`, `follow_up`, `toon_content`

---

### `POST /user-answers` → `201`
Vetoriza a resposta do usuário a uma reflexão como memória pessoal. Busca automaticamente o título da reflexão no Qdrant.

**Request:**
```json
{
  "userId": "user_123",
  "reflectionId": "gratitude_simple_things",
  "content": "Hoje percebi que o café da manhã com minha família é uma bênção enorme."
}
```

**Response:**
```json
{
  "status": "memory_saved",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "reflectionTitle": "Gratidão pelas Coisas Simples"
}
```

**Campos auto-gerados:** `id` (UUID v4), `createdAt` (UTC timezone-aware)

**Metadata indexada:** `user_id`, `reflection_id`, `reflection_title`, `content`, `timestamp`, `toon_context`

---

### `POST /chat` → `200`
Conversa inteligente com RAG e sessões persistentes. Cria ou continua uma sessão de conversa com contexto acumulado e deduplicação de memórias/escrituras.

**Request (nova conversa — sem sessionId):**
```json
{
  "userId": "user_123",
  "message": "Estou me sentindo perdido ultimamente..."
}
```

**Request (continuando conversa — com sessionId):**
```json
{
  "userId": "user_123",
  "message": "É que ando meio sozinho...",
  "sessionId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "reflectionId": "challenges_fear_to_faith"
}
```

**Response:**
```json
{
  "response": "Ei, eu te entendo — e sabe por quê? Porque eu sou você...",
  "sessionId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "sessionTitle": "Sentindo-se perdido",
  "model": "google-gemini-3-flash-preview",
  "contextSources": 4,
  "followUp": ["O que te trouxe paz na última semana?"]
}
```

**Comportamento de sessão:**
- Sem `sessionId` → cria nova sessão automaticamente (UUID4)
- Com `sessionId` válido e ativo (< 6h) → continua a conversa
- Com `sessionId` expirado (> 6h de inatividade) → cria nova sessão automaticamente
- O `sessionId` retornado deve ser armazenado pelo front para mensagens seguintes
- Na **primeira troca**, a LLM gera automaticamente um título curto (3-5 palavras) para a sessão, retornado em `sessionTitle`. Em mensagens seguintes, `sessionTitle` é `null`.

**Pipeline Híbrido (dois cérebros):**
1. Carrega histórico da sessão (até 20 turns — sliding window)
2. Coleta `used_memory_ids` e `used_scriptures` de turns anteriores
3. Busca até 3 memórias pessoais + 2 reflexões relevantes (com deduplicação)
3. Busca o **perfil do usuário** no Qdrant (personalidade, estado emocional, temas, progresso espiritual)
4. Se o perfil **não existe**, busca dados de cadastro no **Firebase Firestore** (nome, XP, nível, sequência) e cria o perfil inicial — o gênero é inferido pelo nome via LLM
5. Se há >=6 turns, **Cerebras/Groq comprime** o histórico em 3-5 frases (LLM grátis); caso contrário, envia turns direto
6. Monta prompt com: System Prompt + perfil TOON + (resumo comprimido OU histórico) + contexto RAG + mensagem
7. **Google AI (Gemini 3 Flash)** gera a resposta final como o Eu Celestial
8. Salva 2 turns (user + assistant) no Qdrant
9. Atualiza metadados da sessão (turn_count, last_activity, title, processed=false)
10. Se for a primeira troca, gera título via Cerebras/Groq

---

### `GET /conversations?userId=xxx` → `200`
Lista todas as sessões de conversa de um usuário, ordenadas por última atividade (mais recente primeiro). O campo `active` é calculado em tempo real (TTL de 6h).

**Response:**
```json
{
  "sessions": [
    {
      "sessionId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "userId": "user_123",
      "title": "Sentindo-se perdido",
      "reflectionId": "challenges_fear_to_faith",
      "turnCount": 8,
      "createdAt": "2026-02-08T10:00:00+00:00",
      "lastActivity": "2026-02-08T10:30:00+00:00",
      "active": true
    }
  ]
}
```

**Se a coleção `conversations` ainda não existir:** retorna `{"sessions": []}`.

---

### `GET /conversations/{session_id}` → `200` | `404`
Retorna o histórico completo de turns de uma sessão específica.

**Response:**
```json
{
  "sessionId": "a1b2c3d4-...",
  "title": "Sentindo-se perdido",
  "turns": [
    {"role": "user", "content": "Oi, como vai?", "timestamp": "2026-02-08T10:00:00+00:00"},
    {"role": "assistant", "content": "E aí! Tudo tranquilo...", "timestamp": "2026-02-08T10:00:01+00:00"}
  ],
  "turnCount": 2
}
```

**Se a sessão não existir:** retorna `404` com `{"detail": "Sessão não encontrada"}`

---

### `POST /generate-prompt` → `200`
Gera um prompt de reflexão completo e enriquecido via LLM. O usuário fornece título, descrição e categoria; a IA retorna perguntas guia, escrituras, texto reflexivo, perfil semântico e configuração para o Eu Celestial.

**Request:**
```json
{
  "userId": "firebase_auth_uid_abc123",
  "title": "Encontrando Paz na Tempestade",
  "description": "Quero refletir sobre como encontrar calma interior mesmo quando tudo ao redor parece caótico.",
  "categoryId": "faith"
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `userId` | `string` | ✅ | Firebase Auth UID do usuário |
| `title` | `string` | ✅ | Título do prompt (não pode estar vazio) |
| `description` | `string` | ✅ | Descrição livre do tema (não pode estar vazia) |
| `categoryId` | `string` | ✅ | Slug da categoria: `gratitude`, `faith`, `challenges`, `self_knowledge`, `relationships`, `purpose` |

**Response:**
```json
{
  "guidingQuestions": [
    "Em quais momentos recentes você sentiu que perdeu o controle da situação?",
    "Existe alguma escritura que te trouxe conforto em tempos difíceis?",
    "O que 'paz interior' significa para você, na prática?"
  ],
  "scriptureReferences": ["Filipenses 4:6-7", "D&C 121:7-8", "João 14:27"],
  "reflection": "A paz que o Salvador oferece não é a ausência de tempestades — é a presença d'Ele no meio delas...",
  "estimatedMinutes": 8,
  "semanticProfile": {
    "keywords": ["paz interior", "tempestade", "calma", "controle", "confiança"],
    "emotionalTarget": "anxiety",
    "emotionalOutcome": "peace",
    "depthLevel": "journaling"
  },
  "aiConfig": {
    "analysisInstruction": "Verifique se o usuário identificou um momento específico de calma no caos...",
    "followUpSuggestions": [
      "O que você faz quando sente que está perdendo o controle?",
      "Tem alguma oração ou hábito que te ajuda a se ancorar?"
    ]
  },
  "embeddingPayload": "Reflexão sobre encontrar paz interior durante tempestades da vida..."
}
```

**Pipeline:** Monta system prompt com regras detalhadas → chama LLM (temperature=0.7, max_tokens=1500) → parseia JSON → preenche defaults → valida → retorna.

**Retry:** Se a primeira chamada falhar ou retornar JSON inválido, faz 1 retry com temperature=0.4. Se falhar novamente, retorna `502`.

**Defaults (campos faltantes):** `guidingQuestions` → fallback genérico | `estimatedMinutes` → 5 | `emotionalTarget` → `"neutral"` | `emotionalOutcome` → `"peace"` | `depthLevel` → `"journaling"` | `keywords` → extraídos do título+descrição.

**Erros:** `422` (input vazio) | `502` (LLM falhou) | `504` (timeout)

---

### `POST /daily-verse/{user_id}` → `200`
Força a geração/atualização do versículo diário de um usuário específico, ignorando a verificação de `dailyVerseDate`. Útil para testes e regeneração manual.

**Request:** Sem body — o `user_id` é passado na URL.

**Response:**
```json
{
  "status": "updated",
  "userId": "user_123"
}
```

**Pipeline:**
1. Verifica se o usuário existe no Firebase Firestore
2. Carrega o perfil do Qdrant (`user_profiles`) — personalidade, estado emocional, temas, progresso espiritual
3. Busca os últimos 3 resumos de conversas processadas (`conversations` com `processed=true`)
4. Monta prompt com perfil TOON + resumos e envia ao Cerebras/Groq
5. LLM escolhe um versículo personalizado (Bíblia, Livro de Mórmon, D&C ou Pérola de Grande Valor)
6. Salva `dailyVerse` e `dailyVerseDate` no documento do usuário no Firestore

**Formato do `dailyVerse`:**
```
1 Néfi 3:7 - Irei e cumprirei as coisas que o Senhor ordenou, pois sei que o Senhor não dá mandamentos aos filhos dos homens sem lhes preparar um caminho.
```

**Erros:** `404` (usuário não existe) | `502` (LLM falhou)

---

### Cron — Daily Verse (meia-noite BRT)
Background job que roda automaticamente à meia-noite no horário de Brasília (UTC-3). Para cada usuário:

1. Verifica se `dailyVerseDate` já é a data de hoje — se sim, pula
2. Carrega perfil do Qdrant + últimos resumos de conversas
3. Gera versículo personalizado via Cerebras/Groq (~5 chamadas/min com delay de 12s)
4. Salva `dailyVerse` + `dailyVerseDate` no Firebase Firestore

**Campos escritos no Firebase (coleção `users`):**
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `dailyVerse` | `string` | Referência + texto do versículo (ex: `"Alma 37:37 - Aconselha-te com o Senhor..."`) |
| `dailyVerseDate` | `string` | Data ISO da última geração (`"2026-02-16"`) — controle de idempotência |

**Idempotência:** Se o servidor reiniciar, o job recalcula o tempo até a próxima meia-noite BRT e não reprocessa usuários que já receberam o verso do dia.

---

### `GET /health` → `200` | `503`
Health check em tempo real. Verifica Qdrant, Cerebras/Groq e Google AI individualmente.

**Response (ok):**
```json
{
  "api": "ok",
  "qdrant": "ok",
  "collections": ["reflections", "user_memories", "conversations", "user_profiles"],
  "llm_providers": {
    "cerebras": "ok",
    "groq": "ok",
    "google_ai": "ok"
  },
  "embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "firebase": "ok",
  "status": "ok"
}
```

**Response (degraded) → `503`:**
```json
{
  "api": "ok",
  "qdrant": "offline",
  "llm_providers": {
    "cerebras": "unreachable",
    "google_ai": "unreachable"
  },
  "firebase": "not_configured",
  "status": "degraded"
}
```

---

## 🧪 Test Battery (37 testes)

A API executa uma bateria de testes automaticamente em toda inicialização, **antes de aceitar requests**. Se qualquer teste falhar, a API aborta com `sys.exit(1)`.

### Testes Unitários (24)
| Teste | O que valida |
|-------|--------------|
| `unit/toon_reflection_build` | Geração TOON de reflexão com todos os campos |
| `unit/toon_reflection_custom_payload` | Passthrough de `embeddingPayload` customizado |
| `unit/toon_answer_build` | Geração TOON de resposta do usuário |
| `unit/llm_prompt_empty_context` | Prompt LLM sem contexto prévio |
| `unit/pydantic_defaults` | UUID auto-gerado, datetime UTC-aware, defaults |
| `unit/pydantic_validation` | Validação de campos obrigatórios |
| `unit/config_values` | Variáveis de ambiente carregadas |
| `unit/deterministic_uuid` | UUID determinístico para IDs string (slug → UUID5, UUID → passthrough) |
| `unit/conversation_models` | ChatRequest/ChatResponse com sessionId/sessionTitle, SessionInfo |
| `unit/config_conversation_values` | Constantes de sessão + perfil + compressão |
| `unit/prompt_generate_model_defaults` | Defaults do `PromptGenerateResponse` |
| `unit/prompt_generate_validation` | Rejeitar `description` ausente no request |
| `unit/prompt_generate_system_prompt` | System prompt contém título/descrição/categoria |
| `unit/prompt_generate_fill_defaults_empty` | Preenchimento de defaults quando LLM retorna vazio |
| `unit/prompt_generate_fill_defaults_valid` | Dados válidos não são sobrescritos |
| `unit/prompt_generate_extract_keywords` | Extração de keywords do título + descrição |
| `unit/prompt_generate_valid_enums` | Sets de categorias e emoções contêm valores esperados |
| `unit/user_profile_model` | Modelo UserProfile: defaults, campos Firebase, timezone |
| `unit/profile_toon_build` | TOON de perfil com todos os campos (nome, gênero, nível, XP, sequência) |
| `unit/profile_toon_empty` | TOON de perfil vazio com fallbacks (desconhecido, indefinido) |
| `unit/conversation_summary_toon` | TOON de resumo de conversa |
| `unit/prompts_not_empty` | COMPRESSION_PROMPT, PROFILE_EXTRACTION_PROMPT e GENDER_INFERENCE_PROMPT não vazios |
| `unit/firebase_config_values` | FIREBASE_SERVICE_ACCOUNT_PATH configurado |
| `unit/firebase_module_import` | Módulo firebase.py importável com funções corretas |

### Testes de Integração (11)
| Teste | O que valida |
|-------|--------------|
| `integration/qdrant_create_collections` | Criação de coleções de teste no Qdrant |
| `integration/qdrant_index_reflection` | Indexação de reflexão com metadados |
| `integration/qdrant_search_reflection` | Busca semântica retorna resultado correto (score > 0.3) |
| `integration/qdrant_check_reflection_exists` | Scroll por `original_id` (existente + inexistente) |
| `integration/qdrant_index_string_id` | Indexação com ID slug via `deterministic_uuid` |
| `integration/qdrant_index_user_answer` | Indexação de resposta com TOON |
| `integration/qdrant_search_user_memory` | Busca filtrada por `user_id` |
| `integration/qdrant_conversation_turns` | Upsert/scroll de turns + meta de sessão |
| `integration/llm_completion` | Chamada real à API Cerebras/Groq |
| `integration/google_ai_completion` | Chamada real à Google AI (Gemini 3 Flash) |
| `integration/qdrant_user_profile_roundtrip` | Upsert + fetch de perfil de usuário no Qdrant |

### Teste E2E (2)
| Teste | O que valida |
|-------|-------------|
| `e2e/rag_pipeline` | Pipeline completo: prompt TOON → Cerebras → resposta > 20 chars |
| `e2e/generate_prompt` | Request → LLM → Parse JSON → Validate campos → Response completa com quality checks |

**Isolamento:** usa coleções prefixadas `__test_battery__` com IDs UUID. Cleanup garantido via `try/finally`.

**Output no boot:**
```
🧪 Test Battery — iniciando...
  ┌─ UNIT (24 testes)
  │ ✓ unit/toon_reflection_build (0.2ms)
  │ ✓ ...
  └─ UNIT concluído
  ┌─ INTEGRATION (11 testes)
  │ ✓ integration/qdrant_create_collections (112.3ms)
  │ ✓ integration/google_ai_completion (250.1ms)
  │ ✓ ...
  └─ INTEGRATION concluído
  ┌─ E2E (2 testes)
  │ ✓ e2e/rag_pipeline (450.1ms)
  │ ✓ e2e/generate_prompt (1200.5ms)
  └─ E2E concluído
  🧹 Cleanup — concluído
🧪 Test Battery — 37/37 passed (2500.3ms)
```

---

## 🔒 Startup — Health Checks

Antes da test battery, a API executa 5 verificações obrigatórias:

1. **Variáveis de ambiente** — `CEREBRAS_API_KEY` obrigatória, `GOOGLE_AI_API_KEY` recomendada, warning se `ALLOWED_ORIGINS=*`
2. **Qdrant** — 5 tentativas com 3s de delay entre cada (espera o container subir)
3. **Firebase** — Inicializa Firebase Admin SDK com service account, verifica conexão Firestore
4. **LLM Providers** — Valida autenticação Cerebras/Groq + Google AI
5. **Embedding model** — Garante que o modelo está acessível (download se necessário)
6. **Coleção de perfis** — Garante que `user_profiles` existe no Qdrant
7. **Background job** — Inicia o job de atualização de perfis (a cada 30min)
8. **Daily verse job** — Inicia o cron de versículo diário (meia-noite BRT)

Qualquer falha = `sys.exit(1)` (o container reinicia via `restart: unless-stopped`).

---

## 🧠 Conceitos-Chave

### TOON (Text Object Oriented Notation)
Formato compacto para payloads vetorizados. Economiza tokens na LLM mantendo estrutura legível:

```
Reflexão: Gratidão pelas Coisas Simples
Origem: Sistema
Categoria: gratitude
Descrição: Aprecie as bênçãos cotidianas
Perfil:
  Alvo: restlessness
  Resultado: contentment
  Nível: quick_thought
Tags: simplicidade, cotidiano, contentamento
Referências: D&C 59:21, Lucas 17:11-19
Perguntas:
  Quais bênçãos simples você esquece de agradecer?
```

Para respostas do usuário:
```
Memória[Usuário]:
  Reflexão: Gratidão pelas Coisas Simples
  Data: 2026-02-07T12:00:00+00:00
  Conteúdo: Hoje percebi que o café da manhã com minha família é uma bênção.
```

### Eu Celestial (Persona da IA)
A IA não é um chatbot genérico. Ela é a **versão celestial do próprio usuário** — o melhor eu dele, já glorificado do outro lado do véu. Personalidade inspirada em Helamã: direta, pragmática, sem ingenuidade sobre a natureza humana. **Honesta antes de bonzinha** — prefere uma frase que incomoda e faz pensar do que três que confortam e não mudam nada. Proibições: frases genéricas de autoajuda, tautologias vazias, perguntas de terapeuta, padrões repetitivos (regra 12 anti-repetição). Escrituras só quando fizer sentido natural. Tom provocador e amoroso — intrigante, direto, curto, em português do Brasil. 12 regras + 6 exemplos ERRADO + 5 exemplos CERTO no SYSTEM_PROMPT.

### RAG (Retrieval-Augmented Generation)
Antes de responder, a IA busca:
- **Memórias pessoais** do usuário (respostas anteriores, filtradas por `user_id`)
- **Reflexões relevantes** do sistema (busca semântica geral)

Quando há sessão ativa, o RAG deduplica automaticamente: memórias e escrituras já usadas em turns anteriores são excluídas da busca, garantindo que o contexto seja sempre inédito.

O contexto é montado em formato TOON e enviado junto com as instruções da persona ao LLM.

### Sessões de Conversa
Conversas são persistidas no Qdrant como uma coleção dedicada (`conversations`). Cada sessão armazena:
- **Turns** — pontos individuais com `role`, `content`, `timestamp`, `used_memory_ids`, `used_scriptures` (`is_session_meta=False`)
- **Metadados** — um ponto com `title`, `turn_count`, `created_at`, `last_activity`, `user_id`, `reflection_id`, `processed` (`is_session_meta=True`)

| Configuração | Valor | Descrição |
|-------------|-------|----------|
| `CHAT_MAX_TURNS` | 20 | Sliding window — máx de turns carregados |
| `SESSION_TTL_HOURS` | 6 | Inatividade máxima antes de expirar |
| `COMPRESSION_MIN_TURNS` | 6 | Histórico é comprimido se >= 6 turns |
| `PROFILE_JOB_INTERVAL_MINUTES` | 30 | Intervalo do background job |
| `DAILY_VERSE_DELAY_SECONDS` | 12 | Delay entre chamadas LLM no cron (~5/min) |
| `DAILY_VERSE_TIMEZONE` | America/Sao_Paulo | Fuso para meia-noite do cron |

### Perfil do Usuário
O sistema mantém um perfil persistente por usuário na coleção `user_profiles` do Qdrant. O perfil é atualizado automaticamente pelo background job a cada 30 minutos, processando sessões de conversa finalizadas (expiradas por TTL de 6h).

**Campos do perfil:**
| Campo | Descrição |
|-------|----------|
| `display_name` | Nome do usuário (vindo do Firebase) |
| `gender` | Gênero inferido pelo nome via LLM (masculino/feminino/indefinido) |
| `total_xp` | XP total do usuário (vindo do Firebase) |
| `current_level` | Nível atual (vindo do Firebase) |
| `current_streak` | Sequência atual de dias (vindo do Firebase) |
| `personality_summary` | Resumo geral da personalidade (2-3 frases) |
| `emotional_state` | Estado emocional atual (1 frase) |
| `recurring_themes` | Temas que o usuário sempre volta (máx 8) |
| `spiritual_progress` | Evolução espiritual observada |
| `version` | Número de atualizações do perfil |
| `conversation_count` | Total de conversas processadas |

**Fluxo de criação (primeira conversa):**
1. Usuário envia primeira mensagem no chat
2. Backend verifica se existe perfil no Qdrant — não existe
3. Backend busca dados de cadastro no Firebase Firestore (coleção `users`)
4. LLM (Cerebras/Groq) infere gênero a partir do `displayName`
5. Perfil inicial é criado no Qdrant com: nome, gênero, XP, nível, sequência
6. Perfil é injetado no prompt do Google AI como TOON

**Fluxo de atualização (background job):**
1. Background job detecta sessões com `processed=false` e `last_activity` > 6h
2. Cerebras/Groq comprime os turns da sessão em resumo (3-5 frases)
3. Cerebras/Groq extrai/atualiza perfil com base no resumo + perfil existente
4. Perfil é salvo no Qdrant com versão incrementada
5. Sessão é marcada como `processed=true` com o resumo salvo

**Uso no chat:** O perfil é injetado no prompt do Google AI como TOON, dando à IA uma visão geral do usuário antes de cada resposta.

### Modelos Pydantic ↔ Dart
Os modelos Python mantêm paridade 1:1 com os modelos Dart do app Flutter:
- `ReflectionCreate` (14 campos, incluindo `SemanticProfile` e `AIConfig`)
- `UserAnswer` (5 campos, UUID e datetime auto-gerados)
- `ChatRequest` / `ChatResponse` (com `sessionId` e `sessionTitle` para persistência)
- `SessionInfo` (metadados da sessão: `title`, turns, timestamps, status ativo)
- `UserProfile` (perfil persistente: nome, gênero, XP, nível, sequência, personalidade, emocional, temas, espiritual, versão)
- `PromptGenerateRequest` (4 campos: `userId`, `title`, `description`, `categoryId`)
- `PromptGenerateResponse` (7 campos: perguntas, escrituras, reflexão, perfil semântico, aiConfig, embeddingPayload)

---

## 🛠 Setup Local

### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose
- Chave da API Cerebras ([cerebras.ai](https://cerebras.ai))

### 1. Clonar e configurar

```bash
git clone https://github.com/Mosarto/jornada_celestial_backend.git
cd jornada_celestial_backend
cp .env.example .env
# Edite .env com suas chaves
```

### 2. Ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Subir o Qdrant

```bash
docker compose up qdrant -d
```

### 4. Rodar a API

```bash
# Desenvolvimento (hot reload + Swagger em /docs)
DEBUG=1 uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Produção
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level warning
```

### 5. Verificar

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

---

## 🐳 Deploy

### Docker Compose (local)

```bash
cp .env.example .env
# Configure CEREBRAS_API_KEY, GROQ_API_KEY, GOOGLE_AI_API_KEY e QDRANT_API_KEY
docker compose up -d
```

O `docker-compose.yml` sobe API + Qdrant juntos. O Qdrant tem healthcheck e a API só inicia após ele estar pronto (`depends_on: condition: service_healthy`).

### Dokploy (produção)

1. Crie um serviço do tipo **"Compose"** (não "Application")
2. Aponte o repositório Git: `https://github.com/Mosarto/jornada_celestial_backend.git`
3. Configure as env vars no painel do Dokploy:
   - `CEREBRAS_API_KEY` — chave da API Cerebras
   - `GROQ_API_KEY` — chave da API Groq (failover)
   - `GOOGLE_AI_API_KEY` — chave da API Google AI (chat)
   - `FIREBASE_SERVICE_ACCOUNT_JSON` — conteúdo completo do `serviceAccountKey.json` (copie e cole o JSON)
   - `QDRANT_API_KEY` — chave de acesso ao Qdrant
   - `ALLOWED_ORIGINS` — origens permitidas (ex: `https://meuapp.com`)
   - `DEBUG` — deixe vazio para produção
4. Deploy — os testes rodam automaticamente no boot

> **Como configurar o Firebase no Dokploy:**  
> Copie o conteúdo inteiro do arquivo `serviceAccountKey.json` e cole na variável `FIREBASE_SERVICE_ACCOUNT_JSON`.  
> O JSON será carregado diretamente via código, sem precisar criar arquivo no container.

**Detalhes do compose:**
- API usa `${VAR}` substitution para receber env vars do Dokploy
- Qdrant expõe apenas porta interna (6333) — sem acesso externo
- Volume `qdrant_data` persiste dados entre deploys
- Ambos os serviços têm `restart: unless-stopped`

---

## 🔐 Variáveis de Ambiente

| Variável              | Descrição                            | Obrigatório | Default                |
|----------------------|--------------------------------------|-------------|------------------------|
| `CEREBRAS_API_KEY`   | Chave da API Cerebras                | ✅          | —                      |
| `GROQ_API_KEY`       | Chave da API Groq (failover)         | ✅          | —                      |
| `GOOGLE_AI_API_KEY`  | Chave da API Google AI (chat)        | ✅          | —                      |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | Caminho do service account JSON | ❌ | `serviceAccountKey.json` |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | JSON do service account (inline) | ❌ | — |
| `QDRANT_API_KEY`     | API key do Qdrant                    | ✅          | —                      |
| `QDRANT_URL`         | URL de conexão com o Qdrant          | ❌          | `http://localhost:6333` |
| `ALLOWED_ORIGINS`    | Origens CORS (separadas por vírgula) | ❌          | `*`                    |
| `DEBUG`              | Ativa `/docs` do Swagger             | ❌          | *(vazio = desativado)* |

> **Nota sobre Firebase:**  
> - **Desenvolvimento local:** Use `FIREBASE_SERVICE_ACCOUNT_PATH` apontando para o arquivo JSON
> - **Deploy no Dokploy/produção:** Use `FIREBASE_SERVICE_ACCOUNT_JSON` com o conteúdo do JSON (escape duplo das aspas ou use string multiline se suportado)

---

## 📋 Logs

A API foi configurada para logs profissionais e limpos:

- **Logger principal** (`jornada_celestial`) em `INFO` — apenas eventos relevantes
- **Bibliotecas ruidosas** (httpx, httpcore, qdrant_client, fastembed, huggingface_hub, google.auth, grpc) silenciadas em `ERROR`
- **Uvicorn** em `CRITICAL` — sem tracebacks de shutdown ou access logs
- **Warnings** filtrados (conexões inseguras, deprecation notices, progress bars)
- **Progress bars** desabilitadas via env vars (`TQDM_DISABLE`, `HF_HUB_DISABLE_PROGRESS_BARS`)
- **1 worker** — sem duplicação de logs de startup

**Exemplo de boot limpo:**
```
2026-02-08 12:00:00 | INFO    | 🚀 Jornada Celestial API v0.8.0 — iniciando...
2026-02-08 12:00:00 | INFO    | ✓ Providers LLM configurados: cerebras, groq
2026-02-08 12:00:00 | INFO    | ✓ Google AI API key configurada
2026-02-08 12:00:00 | INFO    | ✓ Variáveis de ambiente validadas
2026-02-08 12:00:03 | INFO    | ✓ Qdrant conectado em http://qdrant:6333 (4 coleções)
2026-02-08 12:00:03 | INFO    | ✓ Firebase inicializado (project: xxx)
2026-02-08 12:00:03 | INFO    | ✓ Firebase Firestore conectado
2026-02-08 12:00:03 | INFO    | ✓ Cerebras API autenticada
2026-02-08 12:00:03 | INFO    | ✓ Groq API autenticada
2026-02-08 12:00:04 | INFO    | ✓ Google AI API autenticada
2026-02-08 12:00:04 | INFO    | ✓ Embedding (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
2026-02-08 12:00:04 | INFO    | 🧪 Test Battery — 37/37 passed (2500.3ms)
2026-02-08 12:00:04 | INFO    | Profile Job: iniciado (intervalo=30min)
2026-02-08 12:00:04 | INFO    | ✅ API pronta — todos os serviços operacionais
```

---

## � Configuração do Firebase

O backend se integra com o Firebase Firestore para acessar dados de usuário (nome, nível, XP, etc.) e enriquecer o perfil do chat.

### Desenvolvimento Local

1. Baixe o `serviceAccountKey.json` do [Firebase Console](https://console.firebase.google.com/)
2. Coloque na raiz do projeto: `/mnt/HD/Artomos/jornada_celestial/backend/serviceAccountKey.json`
3. O arquivo já está no `.gitignore` — **nunca commite esse arquivo**

> **⚠️ Importante:** Se estiver usando Docker Compose localmente, o JSON no `.env` **precisa estar em uma única linha** (sem quebras de linha no arquivo). Use aspas simples ou escape os `\n` corretamente.

### Deploy no Dokploy

Como o `serviceAccountKey.json` não vai para o GitHub (por segurança), use a variável de ambiente:

1. Abra o arquivo `serviceAccountKey.json` localmente
2. Copie **todo o conteúdo** (o JSON completo)
3. No painel do Dokploy, crie uma variável de ambiente:
   - Nome: `FIREBASE_SERVICE_ACCOUNT_JSON`
   - Valor: Cole o JSON copiado (com todas as quebras de linha, aspas, etc.)

**Exemplo do conteúdo:**
```json
{
  "type": "service_account",
  "project_id": "seu-projeto",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xxx@seu-projeto.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

> **✅ No Dokploy:** Pode colar o JSON com quebras de linha normalmente no Environment Settings - o painel suporta multiline!  
> **⚠️ No `.env` local:** JSON precisa estar em uma linha única sem quebras.

### Como funciona

A inicialização do Firebase verifica em ordem:
1. **`FIREBASE_SERVICE_ACCOUNT_JSON`** (variável de ambiente) — prioridade para deploy
2. **`FIREBASE_SERVICE_ACCOUNT_PATH`** (arquivo local) — fallback para desenvolvimento

Você pode conferir o status do Firebase no endpoint `/health`:
```json
{
  "status": "healthy",
  "firebase": "ok",
  ...
}
```

---

## �📜 Licença

Projeto privado — Artomos © 2026