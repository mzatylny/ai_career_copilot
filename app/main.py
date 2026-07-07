from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.ai_services import (
    analyze_skills_gap,
    answer_with_context,
    evaluate_interview_answer,
    generate_interview_question,
    generate_study_plan,
    rewrite_resume_bullets,
)
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    GapAnalysisResponse,
    HealthResponse,
    InterviewAnswer,
    InterviewFeedbackResponse,
    InterviewQuestionRequest,
    InterviewQuestionResponse,
    JobDescriptionInput,
    ResumeRewriteRequest,
    ResumeRewriteResponse,
    RoadmapRequest,
    RoadmapResponse,
    UploadResponse,
)
from app.rag_engine import delete_session_documents, process_and_store_document, query_documents
from app.utils import safe_filename

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="2.0.0-wow",
    description="AI Career Copilot API: CV/job gap analysis, RAG over PDFs, roadmap generation and interview coaching.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """
    <html>
      <head><title>AI Career Copilot</title></head>
      <body style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 850px; margin: 48px auto; line-height: 1.55;">
        <h1>AI Career Copilot API</h1>
        <p><strong>Status:</strong> running. Open <a href="/docs">/docs</a> for the interactive API.</p>
        <h2>Demo flow</h2>
        <ol>
          <li>POST /api/analyze-gap with resume + job description</li>
          <li>POST /api/generate-roadmap with missing skills</li>
          <li>POST /api/upload-document with a PDF and session_id</li>
          <li>POST /api/chat to ask questions over uploaded PDFs</li>
          <li>POST /api/interview/question and /api/interview/evaluate</li>
        </ol>
      </body>
    </html>
    """


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        ai_mode="mock" if settings.should_use_mock_ai else "openai",
        vector_store=settings.chroma_path,
    )


@app.post(f"{settings.api_prefix}/analyze-gap", response_model=GapAnalysisResponse)
async def api_analyze_gap(data: JobDescriptionInput) -> GapAnalysisResponse:
    return analyze_skills_gap(data.resume_text, data.job_description_text, data.target_seniority.value)


@app.post(f"{settings.api_prefix}/generate-roadmap", response_model=RoadmapResponse)
async def api_generate_roadmap(data: RoadmapRequest) -> RoadmapResponse:
    return generate_study_plan(data.missing_skills, data.timeframe_days, data.hours_per_week, data.target_role)


@app.post(f"{settings.api_prefix}/upload-document", response_model=UploadResponse)
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    if not session_id or len(session_id) > 80:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = safe_filename(file.filename)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {settings.max_upload_mb} MB")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like a valid PDF")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / safe_name
        temp_path.write_bytes(content)
        try:
            chunks = process_and_store_document(temp_path, session_id=session_id, original_filename=safe_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {exc}") from exc

    if chunks == 0:
        raise HTTPException(status_code=422, detail="No readable text found in this PDF")

    return UploadResponse(
        message="Document processed successfully",
        session_id=session_id,
        filename=safe_name,
        chunks_indexed=chunks,
    )


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat_with_documents(data: ChatRequest) -> ChatResponse:
    chunks = query_documents(data.user_query, data.session_id)
    if not chunks:
        return ChatResponse(
            answer="I could not find relevant information in uploaded documents for this session.",
            sources=[],
            confidence="low",
            follow_up_questions=["Have you uploaded a PDF for this session_id?", "Should I answer from general knowledge instead?"],
        )
    return answer_with_context(data.user_query, chunks)


@app.delete(f"{settings.api_prefix}/sessions/{{session_id}}/documents")
async def delete_documents(session_id: str) -> dict[str, int | str]:
    deleted = delete_session_documents(session_id)
    return {"session_id": session_id, "deleted_chunks": deleted}


@app.post(f"{settings.api_prefix}/interview/question", response_model=InterviewQuestionResponse)
async def api_interview_question(data: InterviewQuestionRequest) -> InterviewQuestionResponse:
    return generate_interview_question(data.target_role, data.seniority.value, data.focus_skills)


@app.post(f"{settings.api_prefix}/interview/evaluate", response_model=InterviewFeedbackResponse)
async def api_interview_evaluate(data: InterviewAnswer) -> InterviewFeedbackResponse:
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


# Backwards-compatible endpoint name from your first version.
@app.post(f"{settings.api_prefix}/interview-simulator", response_model=InterviewFeedbackResponse)
async def api_interview_simulator(data: InterviewAnswer) -> InterviewFeedbackResponse:
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


@app.post(f"{settings.api_prefix}/rewrite-resume", response_model=ResumeRewriteResponse)
async def api_rewrite_resume(data: ResumeRewriteRequest) -> ResumeRewriteResponse:
    return rewrite_resume_bullets(data.resume_bullets, data.target_role, data.job_description_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
