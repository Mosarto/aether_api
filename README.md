# Aether API

Backend for [Aether](https://github.com/Mosarto/aether), a reflective journal that combines persistent conversations, semantic memory, and structured personal insights.

The API authenticates Firebase users, retrieves relevant memories from Qdrant, builds compact RAG context, generates responses through OpenRouter, and returns data shaped for the Flutter client.

## What the service does

- Authenticates every user operation with Firebase ID tokens.
- Stores reflections, answers, profiles, sessions, and semantic memories in Qdrant.
- Builds retrieval-augmented prompts with multilingual local embeddings.
- Maintains conversation context with a 20-turn sliding window.
- Compresses longer sessions and updates enriched user profiles in background jobs.
- Generates Akashic Records and specialized dream, aura, stoic, and synchronicity insights.
- Enforces per-user quotas, premium bypass rules, and request rate limits.
- Synchronizes user-facing summaries and account fields with Firestore.

## Request flow

```mermaid
flowchart LR
    A["Flutter client"] -->|"Firebase bearer token"| B["FastAPI"]
    B --> C["Firebase Auth"]
    B --> D["Qdrant memory"]
    D --> E["RAG context"]
    E --> F["OpenRouter"]
    F --> B
    B --> G["Firestore summaries and quota"]
    B --> A
```

For a chat request, the API validates identity and quota, loads recent turns, retrieves semantic context, builds a compact TOON prompt, requests a model completion, persists both turns, and returns remaining quota with the response.

## Stack

| Component | Purpose |
|---|---|
| FastAPI and Uvicorn | HTTP application and lifecycle management |
| Qdrant | Vector storage for reflections, memories, conversations, and profiles |
| FastEmbed | Local multilingual embeddings, 384 dimensions |
| OpenRouter | LLM gateway for chat and background generation |
| Firebase Admin | Token verification, Firestore profiles, summaries, and quotas |
| Docker Compose | Local and production service orchestration |

## Quick start with Docker

### Prerequisites

- Docker Engine with Compose
- OpenRouter API key
- Firebase service account
- Qdrant API key for the Compose stack

### 1. Clone and create the environment file

```bash
git clone https://github.com/Mosarto/aether_api.git
cd aether_api
cp .env.example .env
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

At minimum, configure:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_API_KEY=local-development-key
FIREBASE_SERVICE_ACCOUNT_JSON=<complete-minified-service-account-json>
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

For Docker, place the complete minified Firebase service-account JSON in `FIREBASE_SERVICE_ACCOUNT_JSON`. Keep it only in `.env`; the file is ignored by Git.

### 2. Start the local stack

```bash
docker compose -f docker-compose.local.yml up --build
```

Services become available at:

- API: `http://localhost:8000`
- Health: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`
- Qdrant: `http://localhost:6333`

The first start downloads the embedding model and runs the complete startup battery, so it takes longer than later starts.

## Run without Docker

Start Qdrant separately, then create a Python 3.12 environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_openrouter_key"
export QDRANT_API_KEY="your_qdrant_key"
export FIREBASE_SERVICE_ACCOUNT_PATH="serviceAccountKey.json"
export DEBUG=1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENROUTER_API_KEY = "your_openrouter_key"
$env:QDRANT_API_KEY = "your_qdrant_key"
$env:FIREBASE_SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
$env:DEBUG = "1"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The default Qdrant URL is `http://localhost:6333`.

## Configuration

| Variable | Required | Default | Description |
|---|---:|---|---|
| `OPENROUTER_API_KEY` | Yes | — | LLM gateway credential. Startup stops when absent. |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant endpoint. Compose overrides it with the service hostname. |
| `QDRANT_API_KEY` | Compose/production | Empty | Qdrant authentication key shared by the API and Qdrant service. |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | One Firebase option | `serviceAccountKey.json` | Local service-account file. |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | One Firebase option | Empty | Full service-account JSON for deployment platforms. Takes precedence over the path. |
| `ALLOWED_ORIGINS` | No | Localhost origins | Comma-separated CORS allowlist. Set explicitly in production. |
| `DEBUG` | No | Disabled | Enables Swagger and ReDoc routes. |
| `OPENROUTER_TIMEOUT_SECONDS` | No | `45` | Completion timeout. |
| `OPENROUTER_MAX_RETRIES` | No | `2` | Retries for transient 429 and 503 responses. |
| `OPENROUTER_HTTP_REFERER` | No | Empty | Optional OpenRouter attribution URL. |
| `OPENROUTER_APP_TITLE` | No | `Aether` | Optional OpenRouter application title. |

Do not commit `.env` or `serviceAccountKey.json`.

## Firebase setup

1. Open Firebase Console and create or select a project.
2. Enable Authentication and Firestore.
3. Create a service account under **Project settings > Service accounts**.
4. For local Python development, save the JSON as `serviceAccountKey.json`.
5. For Docker or deployment, set `FIREBASE_SERVICE_ACCOUNT_JSON` to the complete JSON value.

The service account must be able to verify users and read/write the Firestore documents used for profiles, summaries, and quota tracking.

## Authentication

All product endpoints require a Firebase ID token:

```http
Authorization: Bearer <firebase-id-token>
```

The API derives the user ID from the verified token. Request bodies must not supply or override ownership fields.

`GET /health` is the only unauthenticated route. It performs non-billable readiness checks and does not invoke LLM completions.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | API, Qdrant, OpenRouter configuration, embedding, and Firebase readiness |
| `GET` | `/reflections/{reflection_id}/exists` | Check whether a reflection is indexed |
| `POST` | `/reflections` | Index a structured reflection |
| `POST` | `/user-answers` | Store a reflection answer as user memory |
| `POST` | `/chat` | Run the authenticated Nyx chat pipeline |
| `GET` | `/conversations` | List the current user's sessions |
| `GET` | `/conversations/{session_id}` | Load an owned session and its turns |
| `DELETE` | `/conversations/{session_id}` | Delete an owned session |
| `POST` | `/generate-prompt` | Generate a structured reflection prompt |
| `POST` | `/dream` | Analyze dream content |
| `POST` | `/aura` | Generate an emotional-state reading |
| `POST` | `/stoic` | Generate stoic guidance |
| `POST` | `/sync` | Interpret recurring patterns or synchronicities |
| `GET` | `/user/profile` | Return the enriched semantic profile |
| `GET` | `/user/quota` | Return current quota state |

### Chat example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $FIREBASE_ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I keep repeating the same decision pattern. Help me examine it.",
    "sessionId": null,
    "reflectionId": null
  }'
```

Example response:

```json
{
  "response": "...",
  "sessionId": "9cb6203f-2e3c-4bc5-aee0-8a456d9c392c",
  "sessionTitle": "Recurring decisions",
  "model": "openrouter-deepseek/deepseek-v4-pro",
  "contextSources": 4,
  "followUp": ["Which part of the pattern feels familiar?"],
  "remaining": 8
}
```

## Memory and data model

Qdrant uses four logical collections:

- Reflections: system-authored and generated reflection material.
- User memories: answers and durable personal context.
- Conversations: session metadata and individual turns.
- User profiles: LLM-enriched personality, emotional state, recurring themes, and progress.

Firestore remains the source for account-facing fields, subscription state, quota state, and generated summaries consumed by the mobile app.

## Startup and test battery

Startup is intentionally strict:

1. Validate required configuration.
2. Connect to Qdrant.
3. Initialize Firebase when configured.
4. Probe configured OpenRouter models once.
5. Load or verify the embedding model.
6. Run 73 registered tests: 58 unit, 13 integration, and 2 end-to-end.
7. Start profile and daily-insight background jobs.

The process exits when a required dependency or non-soft test fails. Integration tests use real Qdrant and OpenRouter services; budget and network access are required on first boot.

## Project structure

```text
.
├── main.py                    FastAPI app, CORS, routers
├── app/
│   ├── auth.py               Firebase bearer authentication
│   ├── background.py         Expired-session profile processing
│   ├── config.py             Environment, constants, and prompts
│   ├── daily_verse.py        Scheduled personalized insight job
│   ├── firebase.py           Firebase Admin and Firestore access
│   ├── models.py             Pydantic request and response contracts
│   ├── profile.py            Enriched profile lifecycle
│   ├── providers.py          Qdrant, embeddings, and OpenRouter
│   ├── quota.py              Daily and premium quota rules
│   ├── rag.py                Retrieval and prompt assembly
│   ├── rate_limit.py         Per-IP and per-user protection
│   ├── startup.py            Readiness checks and lifespan
│   ├── test_battery.py       Boot-time test registry
│   ├── toon.py               Compact text serialization
│   └── routes/               HTTP endpoint groups
├── Dockerfile
├── docker-compose.yml
├── docker-compose.local.yml
├── requirements.txt
└── .env.example
```

## Production deployment

The production Compose file expects the external `dokploy-network` network and does not expose Qdrant publicly.

Before deployment:

1. Set all required environment variables in the platform secret store.
2. Set a restrictive `ALLOWED_ORIGINS` value.
3. Keep `DEBUG` disabled.
4. Use a strong `QDRANT_API_KEY`.
5. Provide Firebase through `FIREBASE_SERVICE_ACCOUNT_JSON`.
6. Persist the `qdrant_data` volume and configure backups.
7. Confirm `/health` returns `200` after startup completes.

## Troubleshooting

| Symptom | Check |
|---|---|
| Startup reports missing OpenRouter key | Confirm `.env` is loaded and `OPENROUTER_API_KEY` is non-empty. |
| Qdrant is unreachable | Confirm the container is healthy and `QDRANT_URL` matches the execution mode. |
| Firebase is not configured | Verify the service-account path or JSON environment variable. |
| `/docs` returns 404 | Set `DEBUG=1` for local development. |
| Container repeatedly restarts | Read startup logs; strict checks stop the process on dependency or test failure. |
| First boot is slow | FastEmbed may be downloading and loading the multilingual model. |

## License

Released under the [MIT License](LICENSE).
