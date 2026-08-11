import asyncio
import os

os.environ["AI_COPILOT_MOCK_LLM"] = "true"
os.environ["AI_COPILOT_MOCK_EMBEDDINGS"] = "true"
os.environ["CHROMA_PATH"] = "./test_chroma_db"
os.environ["SESSION_DATABASE_PATH"] = "./test_data/sessions.db"
os.environ["OBJECT_STORAGE_PATH"] = "./test_data/objects"

from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.main as main_module
from app.main import app
from app.models import ChatResponse
from app.rate_limit import SlidingWindowRateLimiter

client = TestClient(app)


def test_health_reports_professional_version_and_security_headers():
    response = client.get("/api/health", headers={"X-Request-ID": "test-request-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "3.0.0"
    assert response.json()["vector_store"] == "chroma"
    assert response.headers["x-request-id"] == "test-request-1"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


def test_readiness_checks_local_dependencies():
    response = client.get("/api/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert all(response.json()["checks"].values())


def test_readiness_returns_503_without_leaking_dependency_error(monkeypatch, caplog):
    def fail_ping():
        raise RuntimeError("secret database location")

    monkeypatch.setattr(main_module.session_store, "ping", fail_ping)
    response = client.get("/api/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service dependencies are not ready"}
    assert "secret database location" not in response.text
    assert "secret database location" not in caplog.text


def test_analyze_gap_mock():
    payload = {
        "resume_text": "I built Python APIs with FastAPI, SQL and Docker for data projects. " * 4,
        "job_description_text": (
            "AI Engineer role requiring Python, FastAPI, RAG, vector databases, SQL, "
            "Docker and testing. "
        )
        * 3,
        "target_seniority": "junior",
    }
    response = client.post("/api/analyze-gap", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["match_score"] <= 100
    assert "missing_skills" in data
    assert "strengths" in data
    assert data["generation_mode"] == "mock"
    assert data["degraded"] is False


def test_api_rejects_unknown_fields():
    response = client.post(
        "/api/analyze-gap",
        json={
            "resume_text": "Python API developer " * 5,
            "job_description_text": "Python AI engineer role " * 5,
            "unexpected": "ignored data should not be accepted",
        },
    )

    assert response.status_code == 422


def test_optional_api_key_protects_data_endpoints_but_not_health(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_access_key", SecretStr("test-secret"))

    unauthorized = client.post(
        "/api/rewrite-resume",
        json={"resume_bullets": ["created a FastAPI app"], "target_role": "AI Engineer"},
    )
    health = client.get("/api/health")

    assert unauthorized.status_code == 401
    assert unauthorized.json() == {"detail": "Invalid or missing API key"}
    assert unauthorized.headers["x-content-type-options"] == "nosniff"
    assert health.status_code == 200


def test_optional_api_key_accepts_valid_header(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_access_key", SecretStr("test-secret"))

    response = client.post(
        "/api/rewrite-resume",
        headers={"X-API-Key": "test-secret"},
        json={"resume_bullets": ["created a FastAPI app"], "target_role": "AI Engineer"},
    )

    assert response.status_code == 200
    assert response.json()["rewritten_bullets"]


def test_invalid_authentication_attempts_are_rate_limited(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_access_key", SecretStr("test-secret"))
    monkeypatch.setattr(main_module, "rate_limiter", SlidingWindowRateLimiter(1))
    first = client.post(
        "/api/rewrite-resume",
        headers={"X-API-Key": "wrong"},
        json={"resume_bullets": ["created a FastAPI app"], "target_role": "AI Engineer"},
    )
    second = client.post(
        "/api/rewrite-resume",
        headers={"X-API-Key": "still-wrong"},
        json={"resume_bullets": ["created a FastAPI app"], "target_role": "AI Engineer"},
    )

    assert first.status_code == 401
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) > 0


def test_tenant_cannot_access_another_tenants_session(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_access_key", None)
    monkeypatch.setattr(
        main_module.settings,
        "tenant_api_keys_raw",
        SecretStr("alpha:key-alpha,beta:key-beta"),
    )

    created = client.post("/api/sessions", headers={"X-API-Key": "key-alpha"})
    session_id = created.json()["session_id"]
    cross_tenant = client.get(
        f"/api/sessions/{session_id}/documents",
        headers={"X-API-Key": "key-beta"},
    )

    assert created.status_code == 201
    assert cross_tenant.status_code == 404


def test_missing_tenant_key_is_rejected(monkeypatch):
    monkeypatch.setattr(main_module.settings, "api_access_key", None)
    monkeypatch.setattr(main_module.settings, "tenant_api_keys_raw", SecretStr("alpha:key-alpha"))

    response = client.post("/api/sessions")

    assert response.status_code == 401


def test_production_startup_rejects_short_api_keys(monkeypatch):
    monkeypatch.setattr(main_module.settings, "environment", "production")
    monkeypatch.setattr(main_module.settings, "api_access_key", None)
    monkeypatch.setattr(
        main_module.settings,
        "tenant_api_keys_raw",
        SecretStr("alpha:short-key"),
    )

    async def start_application():
        async with main_module.lifespan(app):
            pass

    try:
        asyncio.run(start_application())
    except RuntimeError as exc:
        assert "at least 32 characters" in str(exc)
    else:
        raise AssertionError("Production accepted a short API key")


def test_roadmap_mock_deduplicates_skills():
    response = client.post(
        "/api/generate-roadmap",
        json={
            "missing_skills": ["RAG", " rag ", "Docker"],
            "timeframe_days": 90,
            "hours_per_week": 10,
            "target_role": "AI Engineer",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_days"] == 90


def test_interview_question_mock():
    response = client.post(
        "/api/interview/question",
        json={
            "target_role": "AI Engineer",
            "seniority": "junior",
            "focus_skills": ["Python", "RAG"],
        },
    )

    assert response.status_code == 200
    assert "question" in response.json()


def test_rewrite_resume_mock():
    response = client.post(
        "/api/rewrite-resume",
        json={"resume_bullets": ["created a FastAPI app"], "target_role": "AI Engineer"},
    )

    assert response.status_code == 200
    assert response.json()["rewritten_bullets"]


def test_upload_rejects_invalid_session_id():
    response = client.post(
        "/api/upload-document",
        data={"session_id": "../unsafe"},
        files={"file": ("resume.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf_content():
    response = client.post(
        "/api/upload-document",
        data={"session_id": "demo_123"},
        files={"file": ("resume.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_processes_pdf_without_loading_unbounded_data(monkeypatch):
    monkeypatch.setattr(main_module, "process_and_store_document", lambda *args: 4)
    response = client.post(
        "/api/upload-document",
        data={"session_id": "demo_123"},
        files={"file": ("My Resume.pdf", b"%PDF-1.4\nmock", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "My_Resume.pdf"
    assert response.json()["chunks_indexed"] == 4


def test_async_upload_persists_job_status(monkeypatch):
    monkeypatch.setattr(main_module, "process_and_store_document", lambda *args: 3)
    created = client.post("/api/sessions")
    session_id = created.json()["session_id"]

    queued = client.post(
        "/api/upload-document-async",
        data={"session_id": session_id},
        files={"file": ("portfolio.pdf", b"%PDF-1.4\nmock", "application/pdf")},
    )
    status = client.get(f"/api/jobs/{queued.json()['job_id']}")

    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["chunks_indexed"] == 3


def test_deleted_session_cancels_queued_ingestion_and_removes_object(tmp_path, monkeypatch):
    created = client.post("/api/sessions")
    session_id = created.json()["session_id"]
    job = main_module.session_store.create_job(
        session_id,
        "local-demo",
        "portfolio.pdf",
    )
    source = tmp_path / "portfolio.pdf"
    source.write_bytes(b"%PDF-1.4\nmock")
    stored_path = main_module.object_store.put(job.job_id, source)
    monkeypatch.setattr(main_module, "delete_session_documents", lambda value: 0)
    monkeypatch.setattr(
        main_module,
        "process_and_store_document",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("cancelled ingestion reached document processing")
        ),
    )

    deleted = client.delete(f"/api/sessions/{session_id}/documents")
    main_module._process_document_job(
        job.job_id,
        job.job_id,
        stored_path,
        "portfolio.pdf",
        session_id,
    )

    assert deleted.status_code == 200
    assert not stored_path.exists()


def test_upload_hides_internal_processing_errors(monkeypatch):
    def fail_processing(*args):
        raise RuntimeError("secret internal database path")

    monkeypatch.setattr(main_module, "process_and_store_document", fail_processing)
    response = client.post(
        "/api/upload-document",
        data={"session_id": "demo_123"},
        files={"file": ("resume.pdf", b"%PDF-1.4\nmock", "application/pdf")},
    )

    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["detail"].startswith("PDF processing failed")


def test_chat_returns_safe_empty_result(monkeypatch):
    monkeypatch.setattr(main_module, "query_documents", lambda *args: [])
    response = client.post(
        "/api/chat", json={"session_id": "demo_123", "user_query": "What is my best skill?"}
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == "low"
    assert response.json()["sources"] == []
    assert response.json()["generation_mode"] == "not_invoked"
    assert response.json()["grounding_status"] == "no_context"


def test_chat_uses_retrieved_context(monkeypatch):
    chunks = [
        {
            "chunk_id": "trusted-1",
            "text": "The candidate built a FastAPI service.",
            "source": "resume.pdf",
            "page": 1,
            "relevance_score": 0.9,
        }
    ]
    monkeypatch.setattr(main_module, "query_documents", lambda *args: chunks)
    monkeypatch.setattr(
        main_module,
        "answer_with_context",
        lambda *args: ChatResponse(
            answer="FastAPI is a strong skill.",
            sources=[],
            confidence="high",
            follow_up_questions=[],
        ),
    )
    response = client.post(
        "/api/chat", json={"session_id": "demo_123", "user_query": "What is my best skill?"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "FastAPI is a strong skill."


def test_list_session_documents(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "list_session_documents",
        lambda session_id: [{"source": "resume.pdf", "chunks": 3, "pages": [1, 2]}],
    )
    response = client.get("/api/sessions/demo_123/documents")

    assert response.status_code == 200
    assert response.json()["total_chunks"] == 3
    assert response.json()["documents"][0]["pages"] == [1, 2]


def test_delete_session_documents(monkeypatch):
    monkeypatch.setattr(main_module, "delete_session_documents", lambda session_id: 7)
    response = client.delete("/api/sessions/demo_123/documents")

    assert response.status_code == 200
    assert response.json() == {"session_id": "demo_123", "deleted_chunks": 7}


def test_session_path_validation_blocks_unsafe_identifiers():
    response = client.delete("/api/sessions/..%2Funsafe/documents")

    assert response.status_code in {404, 422}
