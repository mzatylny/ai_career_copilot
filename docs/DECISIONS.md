# Architecture decisions

## ADR-001: Server-owned sessions

Clients receive cryptographically random session identifiers from `POST /api/sessions`. Ownership is stored separately from Chroma so vector metadata is never treated as authorization. Local demo mode can still create a legacy client-supplied session for backwards compatibility; authenticated deployments cannot.

## ADR-002: Small local adapters before managed infrastructure

SQLite and `LocalObjectStore` make persistence and ownership behavior executable without cloud accounts. Their interfaces and documented scaling boundary show where PostgreSQL and S3-compatible storage replace them. This keeps the portfolio reproducible while avoiding a false claim that local state scales horizontally.

## ADR-003: BackgroundTasks as an explicit stepping stone

Async ingestion exposes a job resource and durable job status, but execution remains in-process. This is enough to demonstrate the API contract. Production should move the same worker function to a durable queue so jobs survive restarts and can be isolated from the API.

## ADR-004: Cache embeddings, not career responses

Caching complete career responses risks retaining sensitive CV data and returning stale guidance. The service therefore caches only bounded embedding vectors under non-reversible content hashes. Distributed deployments can move the same keys to an encrypted shared cache.

## ADR-005: Observable without personal-data labels

Metrics label only HTTP method, route template, status, and job outcome. Session IDs, document names, queries, and CV content are deliberately excluded from metric labels and normal request logs to avoid sensitive-data leakage and unbounded cardinality.

## ADR-006: Degradation is part of the response contract

Provider failures may use deterministic content to preserve demo availability, but fallback output must never look like a successful live-model result. Generated responses therefore include `generation_mode` and `degraded`. RAG responses additionally expose `grounding_status`, and the server drops unsupported citations instead of attaching unrelated retrieved chunks.

## ADR-007: Serialize local session mutations

In-process ingestion and deletion use one lock per session. A deletion waits for active ingestion, removes its vectors and ownership record, and causes queued workers to observe that their durable job was cancelled before parsing. This closes the local data-resurrection race while preserving the documented requirement for transactional shared services in a multi-instance deployment.
