import os

os.environ["AI_COPILOT_MOCK_LLM"] = "true"
os.environ["AI_COPILOT_MOCK_EMBEDDINGS"] = "true"
os.environ["CHROMA_PATH"] = "./test_chroma_db"

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models import ChatResponse

client = TestClient(app)


def test_health_reports_professional_version_and_security_headers():
    response = client.get("/api/health", headers={"X-Request-ID": "test-request-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"] == "2.1.0"
    assert response.json()["vector_store"] == "chroma"
    assert response.headers["x-request-id"] == "test-request-1"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


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
