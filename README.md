# AI Career Copilot

AI Career Copilot is a production-minded FastAPI service that turns a CV and a target job description into an evidence-based skills analysis, a practical learning roadmap, and realistic interview practice. It also provides session-isolated RAG over uploaded PDFs with server-verified source citations.

## Why this project is different

- **Grounded document answers** — source metadata is reconstructed from trusted retrieval results, so the model cannot invent citation IDs or filenames.
- **Privacy boundaries** — every Chroma query and deletion is filtered by a validated `session_id`.
- **Safe local demo** — deterministic mock LLM and embedding modes run without an API key.
- **Bounded processing** — uploads, PDF page count, request text, retrieval size, and embedding batches have explicit limits.
- **Portfolio-ready engineering** — typed API models, tests, CI, non-root Docker runtime, health checks, request IDs, and security headers.

## Capabilities

- CV-to-job skills-gap analysis with evidence, risk flags, and quick wins
- 14–180 day learning roadmaps with milestones and portfolio projects
- PDF ingestion, chunking, embeddings, and session-scoped ChromaDB retrieval
- Grounded document chat with pages, snippets, chunk IDs, and relevance scores
- Session document inventory and privacy-focused deletion
- Technical interview question generation and answer evaluation
- Truth-preserving CV bullet rewriting

## Architecture

```text
Client
  │
  ▼
FastAPI validation and request limits
  ├── Career workflows ──► OpenAI structured output or deterministic mock
  └── Document workflows ─► PDF parser ─► chunker ─► embeddings ─► ChromaDB
                                                        │
                                                        └── session filter + trusted citations
```

More detail is available in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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

Start the API:

```bash
./run_dev.sh
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Runtime mode and version |
| `POST` | `/api/analyze-gap` | Compare a CV with a job description |
| `POST` | `/api/generate-roadmap` | Build a practical learning roadmap |
| `POST` | `/api/upload-document` | Index a PDF for one session |
| `POST` | `/api/chat` | Ask a grounded question over session documents |
| `GET` | `/api/sessions/{session_id}/documents` | List indexed document summaries |
| `DELETE` | `/api/sessions/{session_id}/documents` | Delete all chunks for a session |
| `POST` | `/api/interview/question` | Generate an interview question |
| `POST` | `/api/interview/evaluate` | Evaluate an interview answer |
| `POST` | `/api/rewrite-resume` | Rewrite CV bullets without inventing facts |

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
curl -X POST http://localhost:8000/api/upload-document \
  -F "session_id=demo_123" \
  -F "file=@resume.pdf"

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo_123","user_query":"Which projects best support an AI Engineer application?"}'
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Enables live LLM and embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Structured-output model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PATH` | `./chroma_db` | Persistent vector-store path |
| `MAX_UPLOAD_MB` | `12` | Maximum PDF size |
| `MAX_PDF_PAGES` | `250` | Maximum processed pages per PDF |
| `MAX_CONTEXT_CHUNKS` | `5` | Retrieval result limit |
| `EMBEDDING_BATCH_SIZE` | `64` | Maximum texts per embedding call |
| `CORS_ORIGINS` | localhost origins | Comma-separated allowed origins |

## Tests and quality checks

```bash
ruff check .
pytest
```

The test suite covers API validation, safe uploads, error redaction, request headers, session isolation, citation grounding, document inventory, chunking, and relevance scoring. GitHub Actions runs lint and tests on Python 3.11 and 3.12.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The container runs as an unprivileged user and exposes a health check at `/api/health`.

## Security scope

This repository demonstrates secure defaults but does not provide user authentication or production rate limiting. Before a public multi-user deployment, place it behind an authenticated gateway, assign server-generated user/session ownership, enable rate limits, and use a managed database and secret store.
