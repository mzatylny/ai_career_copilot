# AI Career Copilot

[![CI](https://github.com/mzatylny/ai_career_copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mzatylny/ai_career_copilot/actions/workflows/ci.yml)

AI Career Copilot is a production-minded FastAPI service and interactive web demo that turns a CV and target job description into an evidence-based skills analysis, a practical learning roadmap, and realistic interview practice. It also provides tenant-isolated RAG over uploaded PDFs with server-verified source citations.

## Why this project is different

- **Career outputs stay evidence-based** — structured models and explicit prompts prevent invented experience or fabricated metrics.
- **RAG citations are server-verified** — filenames, pages, snippets, and scores are rebuilt from trusted retrieval metadata.
- **It works without paid services** — safe mock defaults make the complete API easy to run and test locally.
- **The engineering is measurable** — Prometheus metrics, structured logs, load tests, RAG evaluation, security scanning, and deployment assets make quality visible.

## Features
- **Grounded document answers** — source metadata is reconstructed from trusted retrieval results, so the model cannot invent citation IDs or filenames.
- **Privacy boundaries** — authenticated tenants own server-generated sessions, and every Chroma query and deletion remains session-filtered.
- **Safe local demo** — deterministic mock LLM and embedding modes run without an API key.
- **Bounded processing** — uploads, PDF page count, request text, retrieval size, and embedding batches have explicit limits.
- **Truthful degradation** — every generated response identifies OpenAI, mock, or fallback execution; provider failures cannot masquerade as live output.
- **Portfolio-ready engineering** — typed API models, tests, CI, non-root Docker runtime, health checks, request IDs, and security headers.

## Capabilities

- CV-to-job skills-gap analysis with evidence, risk flags, and quick wins
- 14–180 day learning roadmaps with milestones and portfolio projects
- PDF ingestion, chunking, embeddings, and session-scoped ChromaDB retrieval
- Grounded document chat with pages, snippets, chunk IDs, and relevance scores
- Session document inventory and privacy-focused deletion
- Background PDF ingestion with persistent job status
- Technical interview question generation and answer evaluation
- Truth-preserving CV bullet rewriting
- Responsive browser demo, Prometheus metrics, structured audit logs, and Kubernetes manifests

## Architecture

```text
Browser / API client
  │ API key → tenant identity → rate limit
  ▼
FastAPI validation, ownership and observability
  ├── Career workflows ──► OpenAI structured output or deterministic mock
  └── Document workflows ─► async job ─► object store ─► PDF parser
                                                  └────► embeddings ─► ChromaDB
SQLite metadata ──► session ownership + ingestion job state
Prometheus/logs ──► latency, status, job outcome and audit events
```

More detail is available in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the [`threat model`](docs/THREAT_MODEL.md), [`architecture decisions`](docs/DECISIONS.md), and [`deployment guide`](docs/DEPLOYMENT.md).

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

For a free local demo, set:

```env
AI_COPILOT_MOCK_LLM=true
AI_COPILOT_MOCK_EMBEDDINGS=true
```

For shared use, prefer tenant keys:

```env
AI_COPILOT_TENANT_KEYS=portfolio-demo:replace-with-at-least-32-random-characters
```

Send the matching value in the `X-API-Key` header. The legacy single `AI_COPILOT_API_KEY` remains supported. Production mode refuses to start without either form of authentication. Production keys must contain at least 32 characters and cannot be shared by tenants.

Start the API:

```bash
./run_dev.sh
```

Open the interactive demo at [http://localhost:8000](http://localhost:8000) or the API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Runtime mode and version |
| `GET` | `/api/ready` | Verify local persistence dependencies |
| `POST` | `/api/sessions` | Create a server-owned tenant session |
| `POST` | `/api/analyze-gap` | Compare a CV with a job description |
| `POST` | `/api/generate-roadmap` | Build a practical learning roadmap |
| `POST` | `/api/upload-document` | Index a PDF for one session |
| `POST` | `/api/upload-document-async` | Queue PDF ingestion and return a job |
| `GET` | `/api/jobs/{job_id}` | Read tenant-owned ingestion status |
| `POST` | `/api/chat` | Ask a grounded question over session documents |
| `GET` | `/api/sessions/{session_id}/documents` | List indexed document summaries |
| `DELETE` | `/api/sessions/{session_id}/documents` | Delete all chunks for a session |
| `POST` | `/api/interview/question` | Generate an interview question |
| `POST` | `/api/interview/evaluate` | Evaluate an interview answer |
| `POST` | `/api/rewrite-resume` | Rewrite CV bullets without inventing facts |
| `GET` | `/metrics` | Prometheus metrics without personal-data labels |

### Example: analyze a gap

```bash
curl -X POST http://localhost:8000/api/analyze-gap \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with FastAPI, SQL, Docker and data projects...",
    "job_description_text": "AI Engineer role requiring Python, RAG, APIs, Docker and SQL...",
    "target_seniority": "junior"
  }'
```

### Example: upload and query a PDF

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/sessions | python -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')

curl -X POST http://localhost:8000/api/upload-document-async \
  -F "session_id=$SESSION_ID" \
  -F "file=@resume.pdf"

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\",\"user_query\":\"Which projects best support an AI Engineer application?\"}"
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Enables live LLM and embeddings |
| `AI_COPILOT_API_KEY` | empty | Optional API key for all data endpoints |
| `AI_COPILOT_TENANT_KEYS` | empty | Comma-separated `tenant:key` identities |
| `LLM_MODEL` | `gpt-4o-mini` | Structured-output model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PATH` | `./chroma_db` | Persistent vector-store path |
| `SESSION_DATABASE_PATH` | `./data/sessions.db` | SQLite ownership and job metadata |
| `OBJECT_STORAGE_PATH` | `./data/objects` | Staged async-upload objects |
| `MAX_UPLOAD_MB` | `12` | Maximum PDF size |
| `MAX_PDF_PAGES` | `250` | Maximum processed pages per PDF |
| `MAX_DOCUMENT_CHARACTERS` | `2000000` | Maximum extracted text per PDF |
| `MAX_DOCUMENT_CHUNKS` | `2000` | Maximum indexed chunks per PDF |
| `MAX_CONTEXT_CHUNKS` | `5` | Retrieval result limit |
| `EMBEDDING_BATCH_SIZE` | `64` | Maximum texts per embedding call |
| `EMBEDDING_CACHE_SIZE` | `512` | Maximum hashed embedding cache entries |
| `REQUESTS_PER_MINUTE` | `60` | Single-instance per-identity request limit |
| `OPENAI_TIMEOUT_SECONDS` | `30` | AI provider timeout |
| `LLM_MAX_OUTPUT_TOKENS` | `1800` | Per-response cost/output bound |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | Optional OTLP/HTTP trace collector |
| `CORS_ORIGINS` | localhost origins | Comma-separated allowed origins |

Generated response bodies expose `generation_mode` and `degraded`. RAG responses also expose
`grounding_status`; unsupported or forged citations are removed instead of being replaced with
unrelated retrieved chunks.

## Tests and quality checks

```bash
ruff check .
bandit -r app -q
pytest --cov=app --cov-report=term-missing
pip-audit --local --skip-editable
python -m scripts.run_rag_eval --minimum-recall 0.75
```

The test suite covers authentication, tenant ownership, configuration bounds, API validation, safe uploads, error redaction, async jobs, session isolation, citation grounding, document inventory, chunking, metrics, and retrieval-quality calculations. GitHub Actions runs lint, security checks, dependency auditing, tests with coverage, and deterministic RAG evaluation on Python 3.11 and 3.12.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The container runs as an unprivileged user and exposes a health check at `/api/health`.

Tagged releases can publish an SBOM- and provenance-enabled image to GitHub Container Registry. A security-hardened single-replica Kubernetes example lives in [`deploy/kubernetes`](deploy/kubernetes); its deliberate scaling boundaries are explained in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Security scope

This repository enforces tenant-owned sessions when authentication is configured, serializes ingestion/deletion per session, and includes a bounded single-instance limiter. A horizontally scaled public deployment must still use an identity-aware gateway, managed relational/object/vector stores, a durable worker queue, distributed rate limits, TLS termination, and managed secrets. See [`SECURITY.md`](SECURITY.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
