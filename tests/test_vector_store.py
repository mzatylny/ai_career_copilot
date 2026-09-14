from concurrent.futures import ThreadPoolExecutor

import pytest

import app.rag_engine as rag
from app.vector_store import VectorCollection, close_collections, get_collection


@pytest.fixture
def collection(tmp_path):
    store = VectorCollection(tmp_path / "vectors", "documents", 3)
    yield store
    store.close()


def put(store, chunk_id, session="alpha", vector=None):
    store.upsert(ids=[chunk_id], documents=[f"Text for {chunk_id}"],
                 metadatas=[{"session_id": session, "source": "resume.pdf", "page": 1}],
                 embeddings=[vector or [1.0, 0.0, 0.0]])


def test_persistence_preserves_original_ids_and_validates_dimensions(tmp_path):
    path = tmp_path / "vectors"
    first = VectorCollection(path, "documents", 3)
    put(first, "alpha-arbitrary-chunk-id")
    first.close()
    with pytest.raises(ValueError, match="dimensions/distance"):
        VectorCollection(path, "documents", 4)
    restored = VectorCollection(path, "documents", 3)
    try:
        found = restored.get(where={"session_id": "alpha"}, include=["documents", "metadatas"])
        assert found["ids"] == ["alpha-arbitrary-chunk-id"]
        assert found["documents"] == ["Text for alpha-arbitrary-chunk-id"]
        assert found["metadatas"][0]["page"] == 1
    finally:
        restored.close()


def test_rag_reads_and_deletes_only_the_requested_session(collection, monkeypatch):
    put(collection, "alpha-match")
    put(collection, "alpha-opposite", vector=[-1.0, 0.0, 0.0])
    put(collection, "beta-match", session="beta")
    monkeypatch.setattr(rag, "_collection", lambda: collection)
    monkeypatch.setattr(rag, "embed_texts", lambda _: [[1.0, 0.0, 0.0]])
    hits = rag.query_documents("skills", "alpha")
    assert [hit["chunk_id"] for hit in hits] == ["alpha-match", "alpha-opposite"]
    assert [hit["relevance_score"] for hit in hits] == [1.0, 0.0]
    assert rag.list_session_documents("alpha") == [
        {"source": "resume.pdf", "chunks": 2, "pages": [1]}
    ]
    assert rag.delete_session_documents("alpha") == 2
    assert rag.query_documents("skills", "alpha") == []
    assert [h["chunk_id"] for h in rag.query_documents("skills", "beta")] == ["beta-match"]


@pytest.mark.parametrize("where", [{}, {"source": "resume.pdf"}, {"session_id": ""},
                                  {"session_id": {"$ne": "alpha"}}])
def test_unscoped_or_unsupported_filters_are_rejected(collection, where):
    with pytest.raises(ValueError):
        collection.get(where=where, include=[])
    with pytest.raises(ValueError):
        collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=5, where=where, include=[])


def test_listing_and_deletion_cover_multiple_scroll_pages(collection, monkeypatch):
    ids = [f"alpha-{i}" for i in range(300)]
    collection.upsert(ids=ids, documents=ids,
                      metadatas=[{"session_id": "alpha"} for _ in ids],
                      embeddings=[[1.0, 0.0, 0.0] for _ in ids])
    put(collection, "other", session="beta")
    monkeypatch.setattr(rag, "_collection", lambda: collection)
    assert len(collection.get(where={"session_id": "alpha"}, include=[])["ids"]) == 300
    assert rag.delete_session_documents("alpha") == 300
    assert collection.count() == 1


def test_workers_share_one_client_and_can_reopen_after_shutdown(tmp_path):
    path = str(tmp_path / "workers")

    def worker(index):
        store = get_collection(path, "documents", 3)
        put(store, f"alpha-{index}")
        assert store.get(where={"session_id": "alpha"}, include=[])["ids"]
        return id(store)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            identities = set(pool.map(worker, range(20)))
        assert len(identities) == 1
        close_collections()
        assert get_collection(path, "documents", 3).count() == 20
    finally:
        close_collections()


@pytest.mark.parametrize("vector", [[1.0], [float("nan"), 0.0, 0.0]])
def test_invalid_vectors_do_not_create_points(collection, vector):
    with pytest.raises(ValueError):
        put(collection, "invalid", vector=vector)
    assert collection.count() == 0


def test_reupload_removes_stale_chunks_without_affecting_other_sessions(collection, monkeypatch):
    monkeypatch.setattr(rag, "_collection", lambda: collection)
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(rag, "extract_pdf_pages", lambda _: [{"page": 1, "text": "a" * 2500}])
    assert rag.process_and_store_document("resume.pdf", "alpha") == 3
    put(collection, "beta-keep", session="beta")
    monkeypatch.setattr(rag, "extract_pdf_pages", lambda _: [{"page": 1, "text": "short update"}])
    assert rag.process_and_store_document("resume.pdf", "alpha") == 1
    assert collection.count() == 2
    assert collection.get(where={"session_id": "beta"}, include=[])["ids"] == ["beta-keep"]


def test_legacy_chroma_path_is_rejected_without_modification(tmp_path):
    (tmp_path / "chroma.sqlite3").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="legacy Chroma"):
        VectorCollection(tmp_path, "documents", 3)
    assert list(tmp_path.iterdir()) == [tmp_path / "chroma.sqlite3"]
