import pytest
from pydantic import ValidationError

from app.models import ChatRequest, ResumeRewriteRequest, RoadmapRequest


def test_roadmap_skills_are_trimmed_and_deduplicated():
    request = RoadmapRequest(missing_skills=[" RAG ", "rag", "Docker"])

    assert request.missing_skills == ["RAG", "Docker"]


def test_resume_bullets_reject_empty_items_only():
    with pytest.raises(ValidationError):
        ResumeRewriteRequest(resume_bullets=[" ", "\t"])


def test_chat_session_id_rejects_path_characters():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="../unsafe", user_query="Explain my CV")


def test_models_reject_unexpected_fields():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="demo_123", user_query="Explain my CV", hidden="value")
