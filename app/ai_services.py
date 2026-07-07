from __future__ import annotations

import json
from typing import Any, Dict, Type, TypeVar

try:
    from openai import OpenAI
except ImportError:  # Allows mock mode/tests before dependencies are installed.
    OpenAI = None  # type: ignore
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.models import (
    ChatResponse,
    GapAnalysisResponse,
    InterviewFeedbackResponse,
    InterviewQuestionResponse,
    ProjectRecommendation,
    ResumeRewriteResponse,
    RoadmapResponse,
)
from app.utils import short_snippet

T = TypeVar("T", bound=BaseModel)
settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key) if (settings.openai_api_key and OpenAI is not None) else None


def _model_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()
    # OpenAI structured outputs supports JSON Schema, but deeply nested Pydantic schemas
    # can contain titles/defaults that are not needed. Keeping schema explicit and strict.
    return schema


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction for fallback responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first : last + 1]
    return json.loads(text)


def _mock_gap_analysis(resume: str, job_desc: str) -> GapAnalysisResponse:
    jd = job_desc.lower()
    resume_l = resume.lower()
    expected = [
        ("Python", "programming"),
        ("FastAPI", "backend"),
        ("SQL", "data"),
        ("RAG", "ai_ml"),
        ("Vector databases", "ai_ml"),
        ("Docker", "cloud_devops"),
        ("Testing", "backend"),
        ("Cloud deployment", "cloud_devops"),
    ]
    missing = []
    strengths = []
    for skill, category in expected:
        skill_l = skill.lower()
        if skill_l in jd or skill_l.replace(" ", "") in jd:
            item = {
                "skill": skill,
                "category": category,
                "importance": 4,
                "evidence_from_resume": "Found in resume" if skill_l in resume_l else "Not clearly evidenced",
                "evidence_from_job": f"The job description appears to value {skill}.",
                "action": f"Add one concrete project bullet proving {skill}." if skill_l not in resume_l else f"Keep {skill} visible in the top third of the CV.",
            }
            if skill_l in resume_l:
                strengths.append(item)
            else:
                missing.append(item)
    score = max(35, min(92, 70 + len(strengths) * 4 - len(missing) * 5))
    return GapAnalysisResponse(
        match_score=score,
        summary="Mock analysis: your CV has a plausible AI-engineering base, but needs sharper evidence tied to the job description.",
        missing_skills=missing[:6],
        strengths=strengths[:6],
        risk_flags=["Some claims may be too generic unless connected to measurable project outcomes."],
        quick_wins=[
            "Add a 2-line AI Career Copilot project with API, RAG, tests and deployment.",
            "Put Python/FastAPI/RAG/Vector DB keywords near the top.",
            "Quantify results: latency, accuracy, dataset size, users, or documents indexed.",
        ],
        suggested_project_angle="Build an AI Career Copilot with gap analysis, RAG over uploaded PDFs, interview simulation and a polished API demo.",
    )


def _mock_roadmap(missing_skills: list[str], days: int, hours_per_week: int, target_role: str) -> RoadmapResponse:
    skills = ", ".join(missing_skills[:6])
    return RoadmapResponse(
        headline=f"{days}-day {target_role} upgrade plan focused on {skills}",
        total_days=days,
        milestones=[
            {
                "week_range": "Weeks 1-2",
                "focus": "Backend and API foundations",
                "deliverables": ["FastAPI service", "Pydantic schemas", "pytest smoke tests"],
                "practice_tasks": ["Build 5 endpoints", "Write curl examples", "Handle validation errors"],
                "success_metric": "API runs locally and tests pass without an OpenAI key in mock mode.",
            },
            {
                "week_range": "Weeks 3-5",
                "focus": "RAG and document intelligence",
                "deliverables": ["PDF upload", "chunking", "vector search", "source citations"],
                "practice_tasks": ["Index 3 PDFs", "Compare retrieved chunks", "Add page metadata"],
                "success_metric": "Chat endpoint answers with at least 3 relevant sources.",
            },
            {
                "week_range": "Weeks 6-9",
                "focus": "Portfolio polish and deployment",
                "deliverables": ["Dockerfile", "README", "demo script", "GitHub screenshots"],
                "practice_tasks": ["Record 2-minute demo", "Deploy backend", "Add roadmap JSON UI"],
                "success_metric": "A recruiter can understand the project in under 60 seconds.",
            },
        ],
        portfolio_projects=[
            {
                "title": "AI Career Copilot",
                "why_it_matters": "It demonstrates AI engineering, product thinking and real user value.",
                "core_features": ["CV/job gap analysis", "RAG over career PDFs", "interview simulator", "roadmap generation"],
                "tech_stack": ["FastAPI", "OpenAI", "ChromaDB", "Pydantic", "Docker"],
                "stretch_feature": "Add job-posting comparison across 5 roles with a ranking dashboard.",
                "github_readme_pitch": "An end-to-end AI engineering app that turns resumes and job descriptions into actionable learning plans.",
            },
            {
                "title": "Natural Language SQL Analyst",
                "why_it_matters": "Shows practical data access and safe query generation.",
                "core_features": ["schema inspection", "SQL generation", "explain query", "chart-ready JSON"],
                "tech_stack": ["FastAPI", "PostgreSQL", "OpenAI", "SQLAlchemy"],
                "stretch_feature": "Add query risk checks before execution.",
                "github_readme_pitch": "Ask questions about a database in natural language and receive safe, explainable SQL.",
            },
        ],
        weekly_routine=[
            f"Spend {hours_per_week} hours/week: 50% build, 30% reading/docs, 20% testing and README.",
            "End each week with one visible GitHub commit and one screenshot.",
            "Write a short technical note explaining one design decision.",
        ],
        final_demo_script="Upload a CV, paste a job description, show gap analysis, upload a PDF, ask a document question, then run the interview simulator.",
    )


def _mock_interview_question(target_role: str, seniority: str, focus_skills: list[str]) -> InterviewQuestionResponse:
    skills = ", ".join(focus_skills)
    return InterviewQuestionResponse(
        question=f"You are building a {target_role} portfolio app using {skills}. How would you design the API and RAG pipeline so answers are grounded in uploaded documents?",
        difficulty="medium",
        expected_signals=["clear endpoint design", "chunking and metadata", "retrieval evaluation", "error handling", "security boundaries"],
        follow_up_probe="How would you prevent one user's uploaded documents from leaking into another user's chat results?",
    )


def _mock_interview_feedback(question: str, answer: str) -> InterviewFeedbackResponse:
    return InterviewFeedbackResponse(
        score=72,
        verdict="Good structure, but needs more concrete engineering detail.",
        strengths=["You addressed the main idea", "You showed awareness of retrieval and API design"],
        inaccuracies=["The answer should mention metadata filtering by session/user", "It should explain how sources are returned to the user"],
        improved_answer=(
            "I would expose upload and chat endpoints, extract text from PDFs, split it into overlapping chunks, "
            "store embeddings with session_id metadata, retrieve the top chunks for each query, and ask the LLM to answer only from those chunks. "
            "The response should include citations, confidence, and fallback behavior when no relevant context is found."
        ),
        follow_up_study_topics=["Vector-store metadata filtering", "RAG evaluation metrics"],
    )


def _mock_resume_rewrite(bullets: list[str], target_role: str) -> ResumeRewriteResponse:
    rewritten = [
        f"Built and documented {bullet.strip().rstrip('.')} with measurable impact for {target_role} roles."
        for bullet in bullets
    ]
    return ResumeRewriteResponse(
        rewritten_bullets=rewritten,
        keywords_added=["FastAPI", "RAG", "API design", "testing", "deployment"],
        explanation="Mock rewrite: bullets were made more outcome-focused and aligned with AI engineering keywords.",
    )


def _ask_structured(system_prompt: str, user_payload: str, response_model: Type[T], *, fallback: T) -> T:
    """Ask the LLM for a strict JSON object and validate it with Pydantic."""
    if settings.should_use_mock_ai or client is None:
        return fallback

    schema = _model_schema(response_model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content or "{}"
        return response_model.model_validate(_extract_json(content))
    except Exception:
        # Second attempt: older SDK/model fallback with plain JSON object.
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages + [{"role": "user", "content": f"Return valid JSON matching this schema: {json.dumps(schema)}"}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return response_model.model_validate(_extract_json(content))
        except (Exception, ValidationError, json.JSONDecodeError):
            return fallback


def analyze_skills_gap(resume: str, job_desc: str, target_seniority: str = "junior") -> GapAnalysisResponse:
    fallback = _mock_gap_analysis(resume, job_desc)
    return _ask_structured(
        system_prompt=(
            "You are an expert AI career coach and technical recruiter. Analyze the resume against the job description. "
            "Be specific, evidence-based and realistic. Do not invent experience. Return only structured JSON."
        ),
        user_payload=(
            f"Target seniority: {target_seniority}\n\n"
            f"RESUME:\n{resume[:9000]}\n\nJOB DESCRIPTION:\n{job_desc[:9000]}"
        ),
        response_model=GapAnalysisResponse,
        fallback=fallback,
    )


def generate_study_plan(missing_skills: list[str], days: int = 90, hours_per_week: int = 8, target_role: str = "AI Engineer") -> RoadmapResponse:
    fallback = _mock_roadmap(missing_skills, days, hours_per_week, target_role)
    return _ask_structured(
        system_prompt=(
            "You are a senior AI engineer mentor. Create a practical study roadmap that leads to a demonstrable portfolio. "
            "Prefer building over passive learning. Return only structured JSON."
        ),
        user_payload=(
            f"Target role: {target_role}\nTimeframe: {days} days\nHours/week: {hours_per_week}\n"
            f"Missing skills: {', '.join(missing_skills)}"
        ),
        response_model=RoadmapResponse,
        fallback=fallback,
    )


def generate_interview_question(target_role: str, seniority: str, focus_skills: list[str]) -> InterviewQuestionResponse:
    fallback = _mock_interview_question(target_role, seniority, focus_skills)
    return _ask_structured(
        system_prompt="You are a senior technical interviewer. Generate one realistic interview question with assessment signals. Return only JSON.",
        user_payload=f"Role: {target_role}\nSeniority: {seniority}\nFocus skills: {', '.join(focus_skills)}",
        response_model=InterviewQuestionResponse,
        fallback=fallback,
    )


def evaluate_interview_answer(question: str, answer: str, target_role: str = "AI Engineer") -> InterviewFeedbackResponse:
    fallback = _mock_interview_feedback(question, answer)
    return _ask_structured(
        system_prompt=(
            "You are a senior AI engineer interviewer. Evaluate the answer honestly and constructively. "
            "Give a stronger model answer. Return only JSON."
        ),
        user_payload=f"Target role: {target_role}\nQuestion: {question}\nCandidate answer: {answer}",
        response_model=InterviewFeedbackResponse,
        fallback=fallback,
    )


def rewrite_resume_bullets(bullets: list[str], target_role: str, job_description_text: str | None = None) -> ResumeRewriteResponse:
    fallback = _mock_resume_rewrite(bullets, target_role)
    return _ask_structured(
        system_prompt=(
            "You rewrite CV bullets for technical roles. Keep them truthful, concise and achievement-oriented. "
            "Do not invent numbers. Return only JSON."
        ),
        user_payload=(
            f"Target role: {target_role}\nJob description: {short_snippet(job_description_text or '', 4000)}\n"
            f"Bullets:\n" + "\n".join(f"- {b}" for b in bullets)
        ),
        response_model=ResumeRewriteResponse,
        fallback=fallback,
    )


def answer_with_context(query: str, context_blocks: list[dict[str, Any]]) -> ChatResponse:
    sources_for_prompt = []
    for idx, block in enumerate(context_blocks, 1):
        sources_for_prompt.append(
            f"[S{idx}] source={block.get('source')} page={block.get('page')} chunk_id={block.get('chunk_id')}\n{block.get('text')}"
        )
    context = "\n\n".join(sources_for_prompt)

    fallback = ChatResponse(
        answer="I found related document chunks, but the AI service is running in mock mode. The strongest available context is included in the sources.",
        sources=[
            {
                "source": block.get("source", "document"),
                "page": block.get("page"),
                "chunk_id": block.get("chunk_id", "chunk"),
                "snippet": short_snippet(block.get("text", "")),
                "relevance_score": block.get("relevance_score"),
            }
            for block in context_blocks[:5]
        ],
        confidence="medium" if context_blocks else "low",
        follow_up_questions=["Do you want a shorter summary?", "Should I extract action items from this document?"],
    )

    return _ask_structured(
        system_prompt=(
            "You are a careful RAG assistant. Answer strictly from the provided context. "
            "If the context is insufficient, say so. Include source chunks in the response. Return only JSON."
        ),
        user_payload=f"Question: {query}\n\nContext:\n{context[:14000]}",
        response_model=ChatResponse,
        fallback=fallback,
    )
