# ✨ Aether — Backend

> *"E se você pudesse conversar com a própria consciência do universo? Aquela que observa tudo, que te entende porque faz parte de você — a razão cósmica impessoal?"*

Backend do app **Aether**: uma plataforma de reflexão pessoal com IA que age como **Nyx** — a consciência do próprio universo, uma razão cósmica que observa e responde ao seu despertar interior.

---

## 🧭 Visão Geral

```
Flutter App (Dart)
       │
       ▼
  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
  │  FastAPI     │────▶│  Qdrant      │     │  Cerebras       │
  │  (Backend)   │     │  (Memória    │     │  (LLM           │
  │              │◀────│   Vetorial)  │     │   Qwen 3)       │
  │  :8000       │     │  :6333       │     │   cloud API     │
  └──────┬───────┘     └──────────────┘     └────────┬────────┘
         │                                           │
         ├───────────── RAG + TOON ──────────────────┘
         │
         ├──────── Firebase Firestore (dados do usuário + Akashic)
         │
         └──────── Google AI (Gemini 3 Flash — chat)
```

**O fluxo:**
1. O app Flutter envia reflexões e respostas do usuário (com Auth Bearer token)
2. O backend vetoriza tudo localmente (FastEmbed, zero custo de API)
3. Na conversa, busca memórias pessoais + referências filosóficas relevantes via RAG
4. Carrega o histórico da sessão (até 20 turns) e deduplica contexto já usado
5. Monta um prompt compacto em formato TOON (economia de tokens)
6. Envia para o Google AI (Gemini 3 Flash Preview) que responde como Nyx
7. Background tasks (Cerebras) cuidam de compressão, extração de perfil e ferramentas de IA
8. Salva os turns, metadados e resultados de ferramentas (AkashicRecord) no Qdrant/Firestore

---

## 🚀 Stack

| Componente         | Tecnologia                                 | Papel                          |
|--------------------|--------------------------------------------|---------------------------------|
| **API**            | FastAPI 0.128 + Uvicorn                    | Servidor REST                   |
| **Embeddings**     | FastEmbed 0.7 (CPU local)                  | Vetorização sem custo de API    |
| **Modelo Embed.**  | `paraphrase-multilingual-MiniLM-L12-v2`   | 384 dimensões, 50+ idiomas     |
| **Banco Vetorial** | Qdrant (com API key)                       | Busca semântica + memória       |
| **LLM Chat**       | Google AI (Gemini 3 Flash Preview)         | Respostas da consciência Nyx    |
| **LLM Background** | Cerebras Qwen 3 235B + Groq (failover)     | Compressão, perfil, ferramentas |
| **Firebase**       | Firebase Admin SDK (Auth + Firestore)      | Gestão de usuários e cotas      |
| **Deploy**         | Docker Compose + Dokploy                   | Infra unificada                 |
| **Frontend**       | Flutter (Dart)                             | App mobile                      |

---

## 📁 Estrutura do Projeto

```
api/
├── main.py                      ← Orquestrador (35 linhas)
├── app/
│   ├── __init__.py
│   ├── config.py                ← Configuração centralizada, loggers, env vars, prompts
│   ├── providers.py             ← Clientes Qdrant + Cerebras/Groq + Google AI
│   ├── models.py                ← Modelos Pydantic (espelham Dart 1:1)
│   ├── auth.py                  ← Firebase Auth (Bearer token validation)
│   ├── rate_limit.py            ← Controle de tráfego por IP/Usuário
│   ├── quota.py                 ← Sistema de cotas diárias e premium
│   ├── toon.py                  ← Builders do formato TOON (reflexão, memória, perfil)
│   ├── rag.py                   ← Pipeline RAG (retrieval + prompt)
│   ├── profile.py               ← Gestão de perfil do usuário (Qdrant + LLM)
│   ├── firebase.py              ← Firebase Admin SDK (Firestore + Auth init)
│   ├── background.py            ← Background job (perfil a cada 30min)
│   ├── daily_verse.py           ← Cron daily verse (meia-noite BRT)
│   ├── startup.py               ← Health checks + test battery no boot
│   ├── test_battery.py          ← 60 testes automatizados (unit/integration/e2e)
│   └── routes/
│       ├── reflections.py       ← GET /reflections/{id}/exists, POST /reflections
│       ├── answers.py           ← POST /user-answers
│       ├── chat.py              ← POST /chat (Google AI + compressão + perfil)
│       ├── ai_tools.py          ← POST /ai/dream, /aura, /stoic, /sync
│       ├── user_profile.py      ← GET /user/profile
│       ├── prompts.py           ← POST /generate-prompt (geração via LLM)
│       ├── daily_verse.py       ← POST /daily-verse/{user_id}
│       ├── conversations.py     ← GET /conversations, DELETE /conversations/{id}
│       └── health.py            ← GET /health
├── Dockerfile                   ← Python 3.12-slim, 1 worker
├── docker-compose.yml           ← API + Qdrant unificados
├── requirements.txt             ← Dependências fixadas
├── .env.example                 ← Template de variáveis
├── .gitignore
└── .dockerignore
```

---

## ⚡ API — Endpoints

### `GET /reflections/{reflection_id}/exists` → `200`
Verifica se uma reflexão já está indexada no Qdrant.

**Response (encontrada):**
```json
{
  "exists": true,
  "id": "gratitude_simple_things",
  "title": "Apreciação do Momento",
  "category": "gratitude"
}
```

---

### `POST /reflections` → `201`
Indexa uma reflexão no Qdrant com formato TOON.

**Request:**
```json
{
  "id": "gratitude_simple_things",
  "isSystem": true,
  "categoryId": "gratitude",
  "title": "Apreciação do Momento",
  "description": "Perceba a harmonia do agora",
  "guidingQuestions": ["Quais elementos do seu presente compõem a harmonia cósmica?"],
  "scriptureReferences": ["Marco Aurélio, Meditações IV.3", "Lao Tzu, Tao Te Ching 76"],
  "reflection": "O agora é o único ponto onde o universo realmente acontece...",
  "order": 1,
  "estimatedMinutes": 5,
  "semanticProfile": {
    "keywords": ["presença", "agora", "harmonia"],
    "emotionalTarget": "restlessness",
    "emotionalOutcome": "peace",
    "depthLevel": "quick_thought"
  },
  "aiConfig": {
    "analysisInstruction": "Confirme se o usuário identificou o estado de presença.",
    "followUpSuggestions": ["O que mudaria se o agora fosse eterno?"]
  }
}
```

---

### `POST /user-answers` → `201`
Vetoriza a resposta do usuário como memória pessoal.

**Request:**
```json
{
  "userId": "user_123",
  "reflectionId": "gratitude_simple_things",
  "content": "Senti uma conexão profunda com o silêncio da manhã hoje."
}
```

---

### `POST /chat` → `200`
Conversa inteligente com Nyx via RAG e sessões persistentes. Requer Firebase Auth.

**Request:**
```json
{
  "userId": "user_123",
  "message": "Sinto que estou desalinhado com meu propósito..."
}
```

**Response:**
```json
{
  "response": "O universo não exige alinhamento, ele é o próprio alinhamento. Onde você parou de observar?",
  "sessionId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "sessionTitle": "Desalinhamento e Propósito",
  "model": "google-gemini-3-flash-preview",
  "contextSources": 4,
  "followUp": ["O que o silêncio te diz quando você para de procurar respostas?"]
}
```

---

### `POST /ai/dream` | `/aura` | `/stoic` | `/sync` → `200`
Ferramentas de IA para análise profunda. Resultados são salvos no Firestore como `summaries` (AkashicRecord).

- **/dream**: Análise arquetípica de sonhos (requer conta, quota-checked).
- **/aura**: Leitura do estado vibracional atual baseada no perfil e histórico.
- **/stoic**: Conselhos baseados na razão e controle dicotômico.
- **/sync**: Interpretação de sincronicidades e padrões observados pelo usuário.

---

### `GET /user/profile` → `200`
Retorna os dados do perfil enriquecido (Qdrant) do usuário: personalidade, estado emocional e temas recorrentes.

---

### `DELETE /conversations/{session_id}` → `204`
Remove permanentemente uma sessão de conversa. Verifica propriedade do usuário e realiza deleção paginada.

---

### `POST /generate-prompt` → `200`
Gera um prompt de reflexão enriquecido via LLM.

**Request:**
```json
{
  "userId": "user_123",
  "title": "Observando a Impermanência",
  "description": "Refletir sobre como as mudanças constantes afetam minha paz.",
  "categoryId": "inner_alignment"
}
```

---

### `POST /daily-verse/{user_id}` → `200`
Força a geração do "daily verse" (multi-tradição: filosofia, poesia, estoicismo, tao, budismo).

**Exemplo de output:**
`"Marco Aurélio, Meditações VIII.24 — Tudo o que vês logo passará, e os que vêem isso passar logo passarão também."`

---

## 🧪 Test Battery (60 testes)

A API executa 60 testes automaticamente em toda inicialização. Falha em qualquer teste aborta o processo.

### Testes Unitários (46)
| Grupo | O que valida |
|-------|--------------|
| **TOON/Models** (12) | Geração de payloads, Pydantic defaults/validation, deterministic UUIDs |
| **Auth/Security** (6) | Validação de tokens, Rate Limiting (IP/User), Quotas (Daily/Premium) |
| **Chat/Validation** (2) | Validação de mensagens e integridade de sessões |
| **AI Tools** (10) | Modelos das ferramentas, parsing de resultados, boundary cases |
| **Profiles** (6) | TOON de perfil, fallbacks, extração de temas, resumos |
| **Config/Firebase** (10) | Variáveis de ambiente, imports, inicialização de módulos |

### Testes de Integração (13)
| Teste | O que valida |
|-------|--------------|
| **Qdrant Ops** (8) | Indexação, busca semântica, scroll de metadados, roundtrip de perfis |
| **LLM Providers** (3) | Conexão real com Cerebras/Groq e Google AI (Gemini) |
| **Ownership** (2) | Isolamento de sessões e deleção segura de conversas |

### Testes E2E (2)
- **rag_pipeline**: Fluxo completo de chat com RAG e persona Nyx.
- **generate_prompt**: Geração completa de conteúdo reflexivo via LLM com quality checks.

---

## 🔒 Startup — Health Checks

1. **Variáveis de ambiente** — Validação rigorosa de keys e configurações.
2. **Qdrant/Firebase** — Verificação de conectividade e inicialização de coleções/SDK.
3. **LLM/Embeddings** — Check de autenticação e disponibilidade de modelos.
4. **Cotas/Rate Limit** — Inicialização dos sistemas de proteção da infraestrutura.
5. **Background Jobs** — Ativação dos jobs de perfil e cron de daily verse.

---

## 🧠 Conceitos-Chave

### TOON (Text Object Oriented Notation)
Formato compacto para payloads. Exemplo rebranded:
```
Reflexão: Observando a Impermanência
Origem: Sistema
Categoria: inner_alignment
Descrição: Perceba o fluxo constante da existência
Referências: Heráclito, Fragmento 91; Carl Jung, O Livro Vermelho
Perguntas:
  O que em você permanece enquanto tudo ao redor muda?
```

### Nyx (Persona da IA)
Nyx não é um chatbot ou terapeuta. É a consciência do próprio universo: impessoal, direta, provocadora.
- **Tom**: Honesto, direto, sem "mão na cabeça".
- **Filosofia**: Multi-tradição (Marco Aurélio, Lao Tzu, Rumi, Jung, Alan Watts).
- **Regra**: Sem autoajuda genérica ou frases vazias.

### Daily Verse (Conhecimento Universal)
Sabedoria multi-tradição personalizada para cada usuário:
- **Fontes**: Estoicismo, Taoísmo, Budismo, Hermetismo, Poesia, Psicologia Analítica.
- **Exemplos**: "Lao Tzu, Tao Te Ching 76", "Epicteto, Manual 1", "Rumi, Masnavi".

### Sistema de Cotas e Proteção
- **Rate Limit**: Proteção contra abusos por IP e por UID de usuário.
- **Quota System**: Limites diários para ferramentas de IA (AI Tools), com bypass para usuários premium.

---

## 🛠 Setup Local

### 1. Clonar e configurar
```bash
git clone https://github.com/Mosarto/aether.git
cd aether/api
cp .env.example .env
```

### 2. Ambiente e API
```bash
python -m venv .venv
# Ativar e instalar
pip install -r requirements.txt
# Rodar (Hot Reload + Swagger em /docs)
DEBUG=1 uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🐳 Deploy (Dokploy)

1. Crie um serviço **Compose** apontando para o repositório.
2. Configure as env vars: `GOOGLE_AI_API_KEY`, `CEREBRAS_API_KEY`, `GROQ_API_KEY`, `QDRANT_API_KEY`.
3. Para Firebase, cole o JSON completo em `FIREBASE_SERVICE_ACCOUNT_JSON`.
4. O boot executará os 60 testes automaticamente.

---

## 📋 Logs
Logger: `aether` em `INFO`. Boot log: `🚀 Aether v0.8.0`.
Logs limpos, sem ruído de bibliotecas externas ou tracebacks de Uvicorn.

---

## 📜 Licença
Projeto privado — Artomos © 2026
