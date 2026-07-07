import os

os.environ["AI_COPILOT_MOCK_LLM"] = "true"
os.environ["AI_COPILOT_MOCK_EMBEDDINGS"] = "true"
os.environ["CHROMA_PATH"] = "./test_chroma_db"

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_gap_mock():
    payload = {
        "resume_text": "I built Python APIs with FastAPI, SQL and Docker for data projects. " * 4,
        "job_description_text": "AI Engineer role requiring Python, FastAPI, RAG, vector databases, SQL, Docker and testing. " * 3,
        "target_seniority": "junior",
    }
    response = client.post("/api/analyze-gap", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["match_score"] <= 100
    assert "missing_skills" in data
    assert "strengths" in data


def test_roadmap_mock():
    payload = {
        "missing_skills": ["RAG", "Vector databases", "Docker"],
        "timeframe_days": 90,
        "hours_per_week": 10,
        "target_role": "AI Engineer",
    }
    response = client.post("/api/generate-roadmap", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_days"] == 90
    assert len(data["portfolio_projects"]) >= 1


def test_interview_question_mock():
    response = client.post(
        "/api/interview/question",
        json={"target_role": "AI Engineer", "seniority": "junior", "focus_skills": ["Python", "RAG"]},
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
