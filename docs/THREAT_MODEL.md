# Threat model

## Scope and assets

The service handles CV text, job descriptions, uploaded PDFs, vector embeddings, API credentials, and AI-generated career guidance. The primary trust boundaries are the public HTTP API, the LLM provider, PDF parsing, SQLite ownership metadata, local object storage, and ChromaDB.

## Actors

- A legitimate tenant using its own sessions and documents.
- An unauthenticated or incorrectly authenticated client.
- An authenticated but malicious tenant attempting cross-tenant access or resource exhaustion.
- A malicious PDF author attempting parser abuse or indirect prompt injection.
- A compromised dependency, CI action, container image, or deployment secret.

## Priority threats and controls

| Threat | Impact | Current controls | Residual risk / production action |
| --- | --- | --- | --- |
| Cross-tenant document access | CV disclosure or deletion | API-key tenant identity, server-generated sessions, SQLite ownership checks, Chroma session filter | Put authentication at an identity-aware gateway; rotate keys; add authorization audit alerts |
| Direct or indirect prompt injection | Misleading answers or forged sources | Document text marked untrusted, structured outputs, adversarial citation tests, strict retrieval-ID allowlist, no agent tools | Add a larger red-team corpus and provider moderation appropriate to the product |
| Unbounded AI, upload use, or credential guessing | Cost increase, denial of service, or account access | Input/page/upload/text/chunk limits, per-tenant and pre-authentication IP rate limits, output-token cap, timeouts, bounded embedding cache | Use gateway/Redis-backed distributed limits, budget alerts, queue depth limits and per-tenant quotas |
| Ingestion/deletion race | Deleted personal data is recreated by a queued job | Per-session mutation lock, durable ownership/job metadata, cancelled-job check before processing | Use transactional shared state and a durable queue in multi-instance deployments |
| Malicious PDF | CPU or parser exploitation | Signature/extension/size/page checks, temporary storage, non-root container, dependency audit | Parse in an isolated worker with CPU/memory/time limits and malware scanning |
| Credential disclosure | Repository or account compromise | `.env` ignored, secrets kept in environment, redacted settings, restricted CI permissions | Use a managed secret store, short-lived credentials and automatic rotation |
| Supply-chain compromise | Malicious build or dependency | Dependabot, `pip-audit`, Bandit, read-only CI permissions, SBOM/provenance on release images | Pin workflow actions by commit and sign/verify release images |
| Service or AI-provider failure | Incorrect or unavailable guidance | Safe error redaction, structured logs, readiness endpoint, deterministic fallbacks, explicit degraded-mode metadata | Alert on provider-specific failures and define a public degraded-service SLO |

## Privacy decisions

The embedding cache is bounded and keyed by SHA-256; it does not retain raw input text. Local document and metadata storage persists until session deletion. A real deployment must publish retention terms, obtain consent before sending CV content to a model provider, encrypt storage, and define backup deletion behavior.

## Scaling boundary

The included SQLite, local object store, Chroma persistence, background tasks, and rate limiter make one-instance behavior testable. Horizontal production scale requires managed relational metadata, S3-compatible objects, a managed vector store, Redis-backed limiting, and a durable worker queue. The single-replica Kubernetes manifest intentionally does not pretend otherwise.
