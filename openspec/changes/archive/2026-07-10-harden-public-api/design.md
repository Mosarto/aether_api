## Context

The unauthenticated health endpoint calls two OpenRouter models per request. Health monitors and attackers can therefore consume paid or rate-limited capacity. Public documentation also contains stale test counts and internal QA evidence.

## Goals / Non-Goals

**Goals:**
- Keep `/health` safe for frequent unauthenticated polling.
- Report useful readiness without invoking billable LLM completions.
- Make public documentation match executable test registration.

**Non-Goals:**
- Add a full observability platform.
- Change authenticated product endpoints.
- Alter startup's strict dependency validation.

## Decisions

- Report OpenRouter as configured when an API key exists; do not perform completion probes in request handling.
- Retain Qdrant and Firebase connectivity checks because they are non-billable readiness dependencies.
- Treat Qdrant readiness plus OpenRouter configuration as healthy; detailed startup validation remains authoritative.
- Remove internal `.sisyphus` evidence and ignore future evidence.

## Risks / Trade-offs

- [Health cannot detect a revoked OpenRouter key] -> Startup integration checks retain deep validation; operational errors remain visible through authenticated traffic and logs.
- [Firebase probe still adds latency] -> It is non-billable and preserves current readiness semantics.

## Migration Plan

Deploy the route change normally. Rollback restores deep probes but should only occur behind authentication and caching.

## Open Questions

None.
