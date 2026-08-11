# Architecture

## Request layer

FastAPI and Pydantic reject unknown fields, unsafe session identifiers, oversized text inputs, invalid configuration limits, enum values, and malformed uploads. Middleware maps hashed API keys to tenant identities, enforces a bounded request rate, and leaves health monitoring public. Production refuses to start without configured credentials. API responses include a request ID, `no-store` caching, CSP, MIME-sniffing protection, frame denial, permissions policy, and a restrictive referrer policy.

Synchronous OpenAI and ChromaDB work runs outside the async event loop. The synchronous upload remains available, while the async endpoint stores a bounded object, returns a persistent job resource, and processes it after the response. The worker object is deleted after every terminal outcome. A per-session mutation coordinator serializes ingestion and deletion so a queued task cannot recreate document data after its session is deleted.

## Career intelligence

The career workflows request structured JSON and validate every response against a Pydantic model. OpenAI calls have explicit timeout, retry, and output-token limits. If live AI is unavailable or returns invalid data, deterministic fallbacks keep the demo usable and testable. Every response reports `generation_mode` and `degraded`, preventing fallback content from being confused with a successful provider result. Prompts explicitly prohibit invented experience and fabricated metrics.

## Document RAG

1. A PDF is validated by filename, signature, byte limit, page limit, extracted-character limit, and chunk-count limit.
2. Text is extracted per page and split into overlapping sentence-aware chunks.
3. Stable SHA-256 chunk IDs include session, source, page, and content.
4. Embeddings are generated and upserted in bounded batches.
5. Queries always include an exact `session_id` metadata filter.
6. Retrieved cosine distances are converted to bounded relevance values.
7. Document text is marked as untrusted context in the AI prompt.
8. Model-provided citations are discarded unless their chunk IDs exist in the retrieval set; filenames, pages, snippets, and scores are rebuilt from trusted metadata.
9. Invalid or omitted citations remain explicitly `unsupported`; the server never attaches unrelated context merely to make an answer look grounded.

## Data lifecycle

`POST /api/sessions` creates a cryptographically random identifier owned by the authenticated tenant. SQLite stores ownership separately from vector metadata and authorizes every read, chat, ingestion, job-status, and deletion operation. `GET /api/sessions/{session_id}/documents` exposes only document summaries, never stored chunk text. `DELETE /api/sessions/{session_id}/documents` removes every chunk and the associated metadata while holding the same mutation lock used by ingestion workers.

For horizontal production scale, replace SQLite/local objects/in-process jobs with shared managed services while preserving the same ownership and job APIs.

## Observability and evaluation

Request middleware emits structured JSON logs and Prometheus counters/histograms without CV text, questions, document names, or session IDs as metric labels. CI runs deterministic retrieval cases and rejects changes below the configured recall threshold. Locust provides a reproducible load-test scenario for recording throughput, latency percentiles, and errors in mock mode.
