## Why

The public health endpoint currently performs billable external LLM requests on every unauthenticated call. Repository documentation and tracked internal QA artifacts also misrepresent the current test suite and reduce public-project quality.

## What Changes

- Replace billable health probes with configuration and dependency readiness checks.
- Keep the health endpoint safe for unauthenticated liveness monitoring.
- Correct test-count and repository-clone documentation.
- Remove internal agent evidence from version control and ignore future artifacts.

## Capabilities

### New Capabilities

- `operational-health`: Defines a side-effect-free public health endpoint and accurate operational status reporting.

### Modified Capabilities

None.

## Impact

Affected areas include the health route, provider readiness reporting, API documentation, Git ignores, and tracked internal evidence.
