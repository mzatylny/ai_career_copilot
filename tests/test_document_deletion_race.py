import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.main as main
import app.rag_engine as rag
from app.session_locks import SessionMutationCoordinator
from app.session_store import SessionStore
from app.vector_store import VectorCollection


@pytest.mark.parametrize("authenticated", [False, True])
def test_upload_cannot_restore_documents_after_session_deletion(
    tmp_path, monkeypatch, authenticated
):
    key = "test-only-key-for-document-deletion"
    monkeypatch.setattr(main.settings, "api_access_key", None)
    monkeypatch.setattr(
        main.settings, "tenant_api_keys_raw", SecretStr(f"alpha:{key}" if authenticated else "")
    )
    monkeypatch.setattr(main, "session_store", SessionStore(tmp_path / "sessions.db"))
    monkeypatch.setattr(main, "session_mutations", SessionMutationCoordinator())
    collection = VectorCollection(tmp_path / "vectors", "test", rag.settings.embedding_dimensions)
    monkeypatch.setattr(rag, "_collection", lambda: collection)
    monkeypatch.setattr(rag, "extract_pdf_pages", lambda _: [{"page": 1, "text": "Private CV."}])
    monkeypatch.setattr(rag, "embed_texts", lambda texts: [rag._hash_embedding(t) for t in texts])
    uploaded, resume = threading.Event(), threading.Event()
    original_save = main._save_bounded_upload

    async def paused_save(*args, **kwargs):
        count = await original_save(*args, **kwargs)
        uploaded.set()
        if not await asyncio.to_thread(resume.wait, 10):
            raise TimeoutError("test did not resume the upload")
        return count

    monkeypatch.setattr(main, "_save_bounded_upload", paused_save)
    headers = {"X-API-Key": key} if authenticated else {}
    try:
        with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=1) as executor:
            created = client.post("/api/sessions", headers=headers)
            assert created.status_code == 201
            session_id = created.json()["session_id"]
            pending = executor.submit(
                client.post,
                "/api/upload-document",
                headers=headers,
                data={"session_id": session_id},
                files={"file": ("cv.pdf", b"%PDF-1.4\nfixture", "application/pdf")},
            )
            try:
                assert uploaded.wait(10)
                deleted = client.delete(f"/api/sessions/{session_id}/documents", headers=headers)
                assert deleted.status_code == 200
            finally:
                resume.set()
            response = pending.result(timeout=10)
            assert response.status_code == 404
            assert response.json() == {"detail": "Session not found"}
            assert collection.get(where={"session_id": session_id}, include=[])["ids"] == []
            tenant = "alpha" if authenticated else "local-demo"
            assert not main.session_store.ensure_owner(session_id, tenant)
            assert main.session_mutations.active_sessions() == 0
    finally:
        resume.set()
        collection.close()
