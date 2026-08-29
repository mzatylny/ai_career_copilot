from types import SimpleNamespace

import pytest

import app.rag_engine as rag


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        rag.split_text("text " * 100, chunk_size=100, overlap=100)


def test_split_text_creates_bounded_overlapping_chunks():
    chunks = rag.split_text("Sentence one. " * 30, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_embed_texts_uses_and_bounds_the_live_provider_cache(monkeypatch):
    class Embeddings:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[float(index), 1.0])
                    for index, _ in enumerate(kwargs["input"], start=1)
                ]
            )

    embeddings = Embeddings()
    monkeypatch.setattr(rag.settings, "mock_embeddings", False)
    monkeypatch.setattr(rag.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(rag.settings, "embedding_cache_size", 2)
    monkeypatch.setattr(rag, "client", SimpleNamespace(embeddings=embeddings))
    rag._embedding_cache.clear()

    first = rag.embed_texts(["alpha", "beta", "gamma"])
    cached = rag.embed_texts(["beta"])

    assert first == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert cached == [[2.0, 1.0]]
    assert len(embeddings.calls) == 1
    assert list(rag._embedding_cache.values()) == [[3.0, 1.0], [2.0, 1.0]]
    rag._embedding_cache.clear()


def test_stable_chunk_id_is_deterministic_and_session_scoped():
    first = rag._stable_chunk_id("session-a", "resume.pdf", 1, "Python and FastAPI")
    again = rag._stable_chunk_id("session-a", "resume.pdf", 1, "Python and FastAPI")
    other_session = rag._stable_chunk_id("session-b", "resume.pdf", 1, "Python and FastAPI")

    assert first == again
    assert first != other_session
    assert first.startswith("session-a-")


def test_pdf_extraction_rejects_excessive_text(monkeypatch):
    class Page:
        def extract_text(self):
            return "x" * 60

    class Reader:
        is_encrypted = False
        pages = [Page(), Page()]

    monkeypatch.setattr(rag, "PdfReader", lambda path: Reader())
    monkeypatch.setattr(rag.settings, "max_document_characters", 100)

    with pytest.raises(ValueError, match="character processing limit"):
        rag.extract_pdf_pages("oversized.pdf")


def test_document_chunk_count_is_bounded_before_embedding(monkeypatch):
    monkeypatch.setattr(
        rag,
        "extract_pdf_pages",
        lambda path: [{"page": 1, "text": "Sentence. " * 100}],
    )
    monkeypatch.setattr(rag, "split_text", lambda text: ["one", "two", "three"])
    monkeypatch.setattr(rag.settings, "max_document_chunks", 2)

    with pytest.raises(ValueError, match="chunk processing limit"):
        rag.process_and_store_document("oversized.pdf", "session-a")


def test_document_processing_upserts_batches_and_removes_stale_chunks(monkeypatch):
    class Collection:
        def __init__(self):
            self.upserts = []
            self.deleted = []

        def get(self, **kwargs):
            assert kwargs["where"] == {
                "$and": [
                    {"session_id": {"$eq": "session-a"}},
                    {"source": {"$eq": "resume.pdf"}},
                ]
            }
            return {"ids": ["stale-chunk"]}

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

        def delete(self, **kwargs):
            self.deleted.extend(kwargs["ids"])

    collection = Collection()
    monkeypatch.setattr(
        rag,
        "extract_pdf_pages",
        lambda path: [
            {"page": 1, "text": "Python APIs"},
            {"page": 2, "text": "RAG systems"},
        ],
    )
    monkeypatch.setattr(rag, "split_text", lambda text: [text])
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(rag, "_collection", lambda: collection)
    monkeypatch.setattr(rag.settings, "embedding_batch_size", 1)

    chunks = rag.process_and_store_document("stored.pdf", "session-a", "resume.pdf")

    assert chunks == 2
    assert len(collection.upserts) == 2
    assert [call["documents"] for call in collection.upserts] == [
        ["Python APIs"],
        ["RAG systems"],
    ]
    assert collection.deleted == ["stale-chunk"]


def test_list_session_documents_aggregates_without_returning_text(monkeypatch):
    class Collection:
        def get(self, **kwargs):
            assert kwargs["where"] == {"session_id": "demo_123"}
            return {
                "metadatas": [
                    {"source": "resume.pdf", "page": 1},
                    {"source": "resume.pdf", "page": 2},
                    {"source": "notes.pdf", "page": 1},
                ]
            }

    monkeypatch.setattr(rag, "_collection", lambda: Collection())

    documents = rag.list_session_documents("demo_123")

    assert documents == [
        {"source": "notes.pdf", "chunks": 1, "pages": [1]},
        {"source": "resume.pdf", "chunks": 2, "pages": [1, 2]},
    ]


def test_query_documents_converts_cosine_distance_to_bounded_relevance(monkeypatch):
    class Collection:
        def query(self, **kwargs):
            return {
                "ids": [["chunk-1", "chunk-2"]],
                "documents": [["Strong match", "Opposite match"]],
                "metadatas": [[{"source": "cv.pdf", "page": 1}, {"source": "cv.pdf", "page": 2}]],
                "distances": [[0.1, 1.5]],
            }

    monkeypatch.setattr(rag, "_collection", lambda: Collection())
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[0.0, 1.0]])

    results = rag.query_documents("FastAPI", "demo_123")

    assert results[0]["relevance_score"] == 0.9
    assert results[1]["relevance_score"] == 0.0


def test_delete_session_documents_removes_every_matching_chunk(monkeypatch):
    class Collection:
        def __init__(self):
            self.deleted = []

        def get(self, **kwargs):
            assert kwargs["where"] == {"session_id": "demo_123"}
            return {"ids": ["chunk-1", "chunk-2"]}

        def delete(self, **kwargs):
            self.deleted.extend(kwargs["ids"])

    collection = Collection()
    monkeypatch.setattr(rag, "_collection", lambda: collection)

    deleted = rag.delete_session_documents("demo_123")

    assert deleted == 2
    assert collection.deleted == ["chunk-1", "chunk-2"]
