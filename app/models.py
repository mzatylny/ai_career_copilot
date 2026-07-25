from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SESSION_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
SessionId = Annotated[str, Field(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)]


def _clean_string_list(values: list[str], *, item_limit: int, list_name: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item:
            continue
        if len(item) > item_limit:
            raise ValueError(f"Each {list_name} item must be at most {item_limit} characters")
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    if not cleaned:
        raise ValueError(f"At least one non-empty {list_name} item is required")
    return cleaned


class APIModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class Seniority(StrEnum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"


class SkillCategory(StrEnum):
    programming = "programming"
    ai_ml = "ai_ml"
    data = "data"
    backend = "backend"
    cloud_devops = "cloud_devops"
    product = "product"
    communication = "communication"
    other = "other"


class JobDescriptionInput(APIModel):
    resume_text: str = Field(..., min_length=40, max_length=20_000, description="Raw resume/CV text")
    job_description_text: str = Field(
        ..., min_length=40, max_length=20_000, description="Raw job description text"
    )
    target_seniority: Seniority = Seniority.junior


class SkillEvidence(APIModel):
    skill: str = Field(..., min_length=1, max_length=120)
    category: SkillCategory = SkillCategory.other
    importance: int = Field(..., ge=1, le=5)
    evidence_from_resume: str = Field(default="", max_length=1_000)
    evidence_from_job: str = Field(default="", max_length=1_000)
    action: str = Field(..., min_length=1, max_length=1_000)


class GapAnalysisResponse(APIModel):
    match_score: int = Field(..., ge=0, le=100)
    summary: str = Field(..., min_length=1, max_length=4_000)
    missing_skills: list[SkillEvidence]
    strengths: list[SkillEvidence]
    risk_flags: list[str] = Field(default_factory=list)
    quick_wins: list[str] = Field(default_factory=list)
    suggested_project_angle: str = Field(..., min_length=1, max_length=4_000)


class RoadmapRequest(APIModel):
    missing_skills: list[str] = Field(..., min_length=1, max_length=20)
    timeframe_days: int = Field(90, ge=14, le=180)
    hours_per_week: int = Field(8, ge=1, le=60)
    target_role: str = Field(default="AI Engineer", min_length=2, max_length=120)

    @field_validator("missing_skills")
    @classmethod
    def clean_missing_skills(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, item_limit=120, list_name="skill")


class RoadmapMilestone(APIModel):
    week_range: str
    focus: str
    deliverables: list[str]
    practice_tasks: list[str]
    success_metric: str


class ProjectRecommendation(APIModel):
    title: str
    why_it_matters: str
    core_features: list[str]
    tech_stack: list[str]
    stretch_feature: str
    github_readme_pitch: str


class RoadmapResponse(APIModel):
    headline: str
    total_days: int
    milestones: list[RoadmapMilestone]
    portfolio_projects: list[ProjectRecommendation]
    weekly_routine: list[str]
    final_demo_script: str


class ChatRequest(APIModel):
    session_id: SessionId
    user_query: str = Field(..., min_length=3, max_length=2_000)


class SourceChunk(APIModel):
    source: str = Field(..., min_length=1, max_length=240)
    page: int | None = Field(default=None, ge=1)
    chunk_id: str = Field(..., min_length=1, max_length=160)
    snippet: str = Field(..., max_length=1_000)
    relevance_score: float | None = Field(default=None, ge=0, le=1)


class ChatResponse(APIModel):
    answer: str = Field(..., min_length=1, max_length=12_000)
    sources: list[SourceChunk] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    follow_up_questions: list[str] = Field(default_factory=list)


class InterviewQuestionRequest(APIModel):
    target_role: str = Field(default="AI Engineer", min_length=2, max_length=120)
    seniority: Seniority = Seniority.junior
    focus_skills: list[str] = Field(
        default_factory=lambda: ["Python", "RAG", "APIs"], min_length=1, max_length=12
    )

    @field_validator("focus_skills")
    @classmethod
    def clean_focus_skills(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, item_limit=120, list_name="focus skill")


class InterviewQuestionResponse(APIModel):
    question: str
    difficulty: str
    expected_signals: list[str]
    follow_up_probe: str


class InterviewAnswer(APIModel):
    question: str = Field(..., min_length=10, max_length=4_000)
    user_answer: str = Field(..., min_length=5, max_length=12_000)
    target_role: str = Field(default="AI Engineer", min_length=2, max_length=120)


class InterviewFeedbackResponse(APIModel):
    score: int = Field(..., ge=0, le=100)
    verdict: str
    strengths: list[str]
    inaccuracies: list[str]
    improved_answer: str
    follow_up_study_topics: list[str]


class ResumeRewriteRequest(APIModel):
    resume_bullets: list[str] = Field(..., min_length=1, max_length=20)
    target_role: str = Field(default="AI Engineer", min_length=2, max_length=120)
    job_description_text: str | None = Field(default=None, max_length=20_000)

    @field_validator("resume_bullets")
    @classmethod
    def clean_resume_bullets(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value, item_limit=2_000, list_name="resume bullet")


class ResumeRewriteResponse(APIModel):
    rewritten_bullets: list[str]
    keywords_added: list[str]
    explanation: str


class HealthResponse(APIModel):
    status: Literal["ok"]
    app: str
    version: str
    environment: str
    ai_mode: Literal["mock", "openai"]
    vector_store: Literal["chroma"]


class UploadResponse(APIModel):
    message: str
    session_id: SessionId
    filename: str
    chunks_indexed: int = Field(..., ge=1)


class DocumentSummary(APIModel):
    source: str
    chunks: int = Field(..., ge=1)
    pages: list[int] = Field(default_factory=list)


class SessionDocumentsResponse(APIModel):
    session_id: SessionId
    total_chunks: int = Field(..., ge=0)
    documents: list[DocumentSummary] = Field(default_factory=list)


class DeleteDocumentsResponse(APIModel):
    session_id: SessionId
    deleted_chunks: int = Field(..., ge=0)


class ErrorResponse(APIModel):
    detail: str
    hint: str | None = None
