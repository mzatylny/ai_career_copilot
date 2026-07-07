from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Seniority(str, Enum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"


class SkillCategory(str, Enum):
    programming = "programming"
    ai_ml = "ai_ml"
    data = "data"
    backend = "backend"
    cloud_devops = "cloud_devops"
    product = "product"
    communication = "communication"
    other = "other"


class JobDescriptionInput(BaseModel):
    resume_text: str = Field(..., min_length=40, description="Raw resume/CV text")
    job_description_text: str = Field(..., min_length=40, description="Raw job description text")
    target_seniority: Seniority = Seniority.junior


class SkillEvidence(BaseModel):
    skill: str
    category: SkillCategory = SkillCategory.other
    importance: int = Field(..., ge=1, le=5)
    evidence_from_resume: str = ""
    evidence_from_job: str = ""
    action: str


class GapAnalysisResponse(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    summary: str
    missing_skills: List[SkillEvidence]
    strengths: List[SkillEvidence]
    risk_flags: List[str] = []
    quick_wins: List[str] = []
    suggested_project_angle: str


class RoadmapRequest(BaseModel):
    missing_skills: List[str] = Field(..., min_length=1)
    timeframe_days: int = Field(90, ge=14, le=180)
    hours_per_week: int = Field(8, ge=1, le=60)
    target_role: str = "AI Engineer"


class RoadmapMilestone(BaseModel):
    week_range: str
    focus: str
    deliverables: List[str]
    practice_tasks: List[str]
    success_metric: str


class ProjectRecommendation(BaseModel):
    title: str
    why_it_matters: str
    core_features: List[str]
    tech_stack: List[str]
    stretch_feature: str
    github_readme_pitch: str


class RoadmapResponse(BaseModel):
    headline: str
    total_days: int
    milestones: List[RoadmapMilestone]
    portfolio_projects: List[ProjectRecommendation]
    weekly_routine: List[str]
    final_demo_script: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_\-]+$")
    user_query: str = Field(..., min_length=3, max_length=2000)


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str
    snippet: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    confidence: str = Field(..., pattern=r"^(low|medium|high)$")
    follow_up_questions: List[str] = []


class InterviewQuestionRequest(BaseModel):
    target_role: str = "AI Engineer"
    seniority: Seniority = Seniority.junior
    focus_skills: List[str] = Field(default_factory=lambda: ["Python", "RAG", "APIs"])


class InterviewQuestionResponse(BaseModel):
    question: str
    difficulty: str
    expected_signals: List[str]
    follow_up_probe: str


class InterviewAnswer(BaseModel):
    question: str = Field(..., min_length=10)
    user_answer: str = Field(..., min_length=5)
    target_role: str = "AI Engineer"


class InterviewFeedbackResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    verdict: str
    strengths: List[str]
    inaccuracies: List[str]
    improved_answer: str
    follow_up_study_topics: List[str]


class ResumeRewriteRequest(BaseModel):
    resume_bullets: List[str] = Field(..., min_length=1, max_length=20)
    target_role: str = "AI Engineer"
    job_description_text: Optional[str] = None

    @field_validator("resume_bullets")
    @classmethod
    def bullets_not_empty(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty bullet is required")
        return cleaned


class ResumeRewriteResponse(BaseModel):
    rewritten_bullets: List[str]
    keywords_added: List[str]
    explanation: str


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    ai_mode: str
    vector_store: str


class UploadResponse(BaseModel):
    message: str
    session_id: str
    filename: str
    chunks_indexed: int


class ErrorResponse(BaseModel):
    detail: str
    hint: Optional[str] = None
