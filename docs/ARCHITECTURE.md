# Architecture

## Request layer

FastAPI and Pydantic reject unknown fields, unsafe session identifiers, oversized text inputs, invalid enum values, and malformed uploads. API responses include a request ID, `no-store` caching, MIME-sniffing protection, frame denial, and a restrictive referrer policy.

Synchronous OpenAI and ChromaDB work runs outside the async event loop. PDF uploads are copied in bounded chunks to a temporary directory and deleted after processing.

## Career intelligence

The career workflows request structured JSON and validate every response against a Pydantic model. If live AI is unavailable or returns invalid data, deterministic fallbacks keep the demo usable and testable. Prompts explicitly prohibit invented experience and fabricated metrics.

## Document RAG

1. A PDF is validated by filename, signature, byte limit, and page limit.
2. Text is extracted per page and split into overlapping sentence-aware chunks.
3. Stable SHA-256 chunk IDs include session, source, page, and content.
4. Embeddings are generated and upserted in bounded batches.
5. Queries always include an exact `session_id` metadata filter.
6. Retrieved cosine distances are converted to bounded relevance values.
7. Document text is marked as untrusted context in the AI prompt.
8. Model-provided citations are discarded unless their chunk IDs exist in the retrieval set; filenames, pages, snippets, and scores are rebuilt from trusted metadata.

## Data lifecycle

`GET /api/sessions/{session_id}/documents` exposes only document summaries, never stored chunk text. `DELETE /api/sessions/{session_id}/documents` removes every chunk under that validated session identifier.

For a true multi-user deployment, session ownership must be tied to authenticated users rather than supplied directly by clients.
