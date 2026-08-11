import pytest

import app.rag_engine as rag


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        rag.split_text("text " * 100, chunk_size=100, overlap=100)


def test_split_text_creates_bounded_overlapping_chunks():
    chunks = rag.split_text("Sentence one. " * 30, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


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
