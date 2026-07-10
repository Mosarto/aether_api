# operational-health Specification

## Purpose
TBD - created by archiving change harden-public-api. Update Purpose after archive.
## Requirements
### Requirement: Health checks are side-effect free
The public health endpoint MUST NOT invoke LLM completions or other billable generation operations.

#### Scenario: Health endpoint is polled
- **WHEN** any caller requests `/health`
- **THEN** the API reports configuration and dependency readiness without sending prompts to OpenRouter

### Requirement: Health status remains operationally useful
The endpoint SHALL report API, Qdrant, OpenRouter configuration, embedding model, and Firebase readiness using non-sensitive status values.

#### Scenario: OpenRouter key is absent
- **WHEN** `OPENROUTER_API_KEY` is empty
- **THEN** health reports OpenRouter as `not_configured` and overall status as degraded

### Requirement: Public documentation reflects implementation
Repository documentation MUST describe the registered test count and correct clone target accurately.

#### Scenario: Contributor follows setup documentation
- **WHEN** a contributor reads the README
- **THEN** commands and test counts match the current repository
