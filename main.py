from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import ALLOWED_ORIGINS, DEBUG
from app.startup import lifespan
from app.rate_limit import RateLimitExceeded
from app.routes import reflections, answers, chat, health, conversations, prompts, ai_tools, user_profile

app = FastAPI(
    title="Jornada Celestial API",
    version="0.6.0",
    docs_url="/docs" if DEBUG else None,
    lifespan=lifespan,
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return top-level JSON body for 429 (not wrapped in 'detail')."""
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded", "retryAfter": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(reflections.router)
app.include_router(answers.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(prompts.router)
app.include_router(ai_tools.router)
app.include_router(health.router)
app.include_router(user_profile.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
