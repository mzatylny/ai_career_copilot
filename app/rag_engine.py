from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:  # type: ignore
    chromadb = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:  # Allows mock embeddings before dependencies are installed.
    OpenAI = None  # type: ignore
from pypdf import PdfReader

from app.config import get_settings
from app.embeddings import hash_embedding
from app.utils import short_snippet

settings = get_settings()
client = (
    OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    if settings.openai_api_key and OpenAI is not None
    else None
)
_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
_embedding_cache_lock = threading.Lock()


def _hash_embedding(text: str, dimensions: int | None = None) -> list[float]:
    """Backwards-compatible wrapper around the dependency-free demo embedding."""
    return hash_embedding(text, dimensions or settings.embedding_dimensions)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.should_use_mock_embeddings or client is None:
        return [_hash_embedding(text, settings.embedding_dimensions) for text in texts]

    cache_keys = [
        hashlib.sha256(
            f"{settings.embedding_model}:{settings.embedding_dimensions}:{text}".encode()
        ).hexdigest()
        for text in texts
    ]
    resolved: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    with _embedding_cache_lock:
        for index, key in enumerate(cache_keys):
            cached = _embedding_cache.get(key)
            if cached is None:
                missing_indices.append(index)
            else:
                _embedding_cache.move_to_end(key)
                resolved[index] = cached

    if missing_indices:
        missing_texts = [texts[index] for index in missing_indices]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=missing_texts,
            dimensions=settings.embedding_dimensions,
        )
        with _embedding_cache_lock:
            for index, item in zip(missing_indices, response.data, strict=True):
                resolved[index] = item.embedding
                if settings.embedding_cache_size:
                    _embedding_cache[cache_keys[index]] = item.embedding
                    _embedding_cache.move_to_end(cache_keys[index])
            while len(_embedding_cache) > settings.embedding_cache_size:
                _embedding_cache.popitem(last=False)

    return [embedding for embedding in resolved if embedding is not None]


def _collection():
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt")
    db = chromadb.PersistentClient(path=settings.chroma_path)
    return db.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )


def extract_pdf_pages(file_path: str | Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(file_path))
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError("Encrypted PDFs are not supported") from exc
        if not unlocked:
            raise ValueError("Encrypted PDFs are not supported")
    if len(reader.pages) > settings.max_pdf_pages:
        raise ValueError(f"PDF exceeds the {settings.max_pdf_pages}-page processing limit")

    pages: list[dict[str, Any]] = []
    extracted_characters = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        compact = re.sub(r"\n{3,}", "\n\n", text).strip()
        if compact:
            extracted_characters += len(compact)
            if extracted_characters > settings.max_document_characters:
                raise ValueError(
                    "PDF exceeds the "
                    f"{settings.max_document_characters:,}-character processing limit"
                )
            pages.append({"page": page_index, "text": compact})
    return pages


def split_text(text: str, *, chunk_size: int = 1100, overlap: int = 160) -> list[str]:
    """Simple dependency-light splitter with overlap."""
    if chunk_size < 20:
        raise ValueError("chunk_size must be at least 20 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer ending at a sentence boundary when possible.
        boundary = max(
            text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end)
        )
        if boundary > start + chunk_size * 0.55:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _stable_chunk_id(session_id: str, source: str, page: int, text: str) -> str:
    digest = hashlib.sha256(f"{session_id}|{source}|{page}|{text[:500]}".encode()).hexdigest()[:20]
    return f"{session_id}-{digest}"


def process_and_store_document(
    file_path: str | Path, session_id: str, original_filename: str | None = None
) -> int:
    """Load a PDF, split it and upsert chunks into ChromaDB."""
    source = original_filename or Path(file_path).name
    pages = extract_pdf_pages(file_path)
    if not pages:
        return 0

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for page in pages:
        for idx, chunk in enumerate(split_text(page["text"]), start=1):
            if len(documents) >= settings.max_document_chunks:
                raise ValueError(
                    f"PDF exceeds the {settings.max_document_chunks:,}-chunk processing limit"
                )
            chunk_id = _stable_chunk_id(session_id, source, page["page"], chunk + str(idx))
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "session_id": session_id,
                    "source": source,
                    "page": page["page"],
                    "chunk_index": idx,
                    "snippet": short_snippet(chunk, 260),
                }
            )

    if not documents:
        return 0

    col = _collection()
    document_filter = {
        "$and": [
            {"session_id": {"$eq": session_id}},
            {"source": {"$eq": source}},
        ]
    }
    existing = col.get(where=document_filter, include=[])
    existing_ids = set(existing.get("ids") or [])

    batch_size = max(1, min(settings.embedding_batch_size, 256))
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = documents[start:end]
        col.upsert(
            ids=ids[start:end],
            documents=batch_documents,
            metadatas=metadatas[start:end],
            embeddings=embed_texts(batch_documents),
        )

    stale_ids = sorted(existing_ids.difference(ids))
    if stale_ids:
        col.delete(ids=stale_ids)
    return len(documents)


def query_documents(query: str, session_id: str, k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for one user/session."""
    top_k = max(1, min(k or settings.max_context_chunks, 20))
    col = _collection()
    query_embedding = embed_texts([query])[0]
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
        if not chunk_id or not doc:
            continue
        relevance = None
        if distance is not None:
            # Chroma cosine distance is 0 for an identical vector and can approach 2.
            relevance = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
        output.append(
            {
                "chunk_id": chunk_id,
                "text": doc,
                "source": meta.get("source", "document") if meta else "document",
                "page": meta.get("page") if meta else None,
                "snippet": meta.get("snippet", short_snippet(doc)) if meta else short_snippet(doc),
                "relevance_score": relevance,
            }
        )
    return output


def list_session_documents(session_id: str) -> list[dict[str, Any]]:
    """Return document-level summaries without exposing stored chunk text."""
    found = _collection().get(where={"session_id": session_id}, include=["metadatas"])
    metadatas = found.get("metadatas") or []
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": 0, "pages": set()})

    for metadata in metadatas:
        if not metadata:
            continue
        source = str(metadata.get("source") or "document")
        grouped[source]["chunks"] += 1
        page = metadata.get("page")
        if isinstance(page, int) and page > 0:
            grouped[source]["pages"].add(page)

    return [
        {
            "source": source,
            "chunks": values["chunks"],
            "pages": sorted(values["pages"]),
        }
        for source, values in sorted(grouped.items())
    ]


def delete_session_documents(session_id: str) -> int:
    """Delete all stored chunks for a session. Useful for privacy and testing."""
    col = _collection()
    found = col.get(where={"session_id": session_id}, include=[])
    ids = found.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)
