from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable

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
from app.utils import short_snippet

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key) if (settings.openai_api_key and OpenAI is not None) else None


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Deterministic lightweight embedding for tests/demos when no API key exists."""
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if settings.should_use_mock_embeddings or client is None:
        return [_hash_embedding(text) for text in texts]

    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def _collection():
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt")
    db = chromadb.PersistentClient(path=settings.chroma_path)
    return db.get_or_create_collection(name=settings.collection_name, metadata={"hnsw:space": "cosine"})


def extract_pdf_pages(file_path: str | Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(file_path))
    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        compact = re.sub(r"\n{3,}", "\n\n", text).strip()
        if compact:
            pages.append({"page": page_index, "text": compact})
    return pages


def split_text(text: str, *, chunk_size: int = 1100, overlap: int = 160) -> list[str]:
    """Simple dependency-light splitter with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer ending at a sentence boundary when possible.
        boundary = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
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
    digest = hashlib.sha1(f"{session_id}|{source}|{page}|{text[:500]}".encode("utf-8")).hexdigest()[:16]
    return f"{session_id}-{digest}"


def process_and_store_document(file_path: str | Path, session_id: str, original_filename: str | None = None) -> int:
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

    embeddings = embed_texts(documents)
    col = _collection()
    col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def query_documents(query: str, session_id: str, k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve relevant chunks for one user/session."""
    top_k = k or settings.max_context_chunks
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

    for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances):
        relevance = None
        if distance is not None:
            relevance = round(1 / (1 + float(distance)), 4)
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


def delete_session_documents(session_id: str) -> int:
    """Delete all stored chunks for a session. Useful for privacy and testing."""
    col = _collection()
    found = col.get(where={"session_id": session_id}, include=[])
    ids = found.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)
