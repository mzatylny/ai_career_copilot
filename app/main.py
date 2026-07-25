from __future__ import annotations

import logging
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi import Path as PathParameter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from app import __version__
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
    SESSION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    DeleteDocumentsResponse,
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
    SessionDocumentsResponse,
    UploadResponse,
)
from app.rag_engine import (
    delete_session_documents,
    list_session_documents,
    process_and_store_document,
    query_documents,
)
from app.utils import safe_filename

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "AI Career Copilot API: CV/job gap analysis, grounded RAG over PDFs, "
        "roadmap generation and interview coaching."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allows_credentials,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "Authorization", "X-Request-ID"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip()[:80] or secrets.token_hex(12)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith(settings.api_prefix):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return f"""
    <html>
      <head><title>AI Career Copilot</title></head>
      <body style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 850px; margin: 48px auto; line-height: 1.55;">
        <h1>AI Career Copilot API</h1>
        <p><strong>Status:</strong> running (v{__version__}). Open <a href="/docs">/docs</a> for the interactive API.</p>
        <h2>Demo flow</h2>
        <ol>
          <li>Analyze a CV against a job description</li>
          <li>Generate a skills roadmap</li>
          <li>Upload and inspect PDF documents for one session</li>
          <li>Ask grounded questions with trusted source chunks</li>
          <li>Generate and evaluate technical interview answers</li>
        </ol>
      </body>
    </html>
    """


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.environment,
        ai_mode="mock" if settings.should_use_mock_ai else "openai",
        vector_store="chroma",
    )


@app.post(f"{settings.api_prefix}/analyze-gap", response_model=GapAnalysisResponse)
def api_analyze_gap(data: JobDescriptionInput) -> GapAnalysisResponse:
    return analyze_skills_gap(
        data.resume_text, data.job_description_text, data.target_seniority.value
    )


@app.post(f"{settings.api_prefix}/generate-roadmap", response_model=RoadmapResponse)
def api_generate_roadmap(data: RoadmapRequest) -> RoadmapResponse:
    return generate_study_plan(
        data.missing_skills, data.timeframe_days, data.hours_per_week, data.target_role
    )


async def _save_bounded_upload(file: UploadFile, destination: Path, max_bytes: int) -> int:
    total = 0
    header = b""
    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            if not header:
                header = chunk[:5]
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size is {settings.max_upload_mb} MB",
                )
            output.write(chunk)

    if not header.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like a valid PDF")
    return total


@app.post(f"{settings.api_prefix}/upload-document", response_model=UploadResponse)
async def upload_document(
    session_id: Annotated[
        str, Form(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)
    ],
    file: Annotated[UploadFile, File(...)],
) -> UploadResponse:
    if not file.filename or not file.filename.casefold().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = safe_filename(file.filename)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / safe_name
            await _save_bounded_upload(file, temp_path, settings.max_upload_bytes)
            try:
                chunks = await run_in_threadpool(
                    process_and_store_document,
                    temp_path,
                    session_id,
                    safe_name,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("PDF processing failed for request session")
                raise HTTPException(
                    status_code=500, detail="PDF processing failed. Please verify the file and try again."
                ) from exc
    finally:
        await file.close()

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
    chunks = await run_in_threadpool(query_documents, data.user_query, data.session_id)
    if not chunks:
        return ChatResponse(
            answer="I could not find relevant information in uploaded documents for this session.",
            sources=[],
            confidence="low",
            follow_up_questions=[
                "Have you uploaded a PDF for this session_id?",
                "Should I answer from general knowledge instead?",
            ],
        )
    return await run_in_threadpool(answer_with_context, data.user_query, chunks)


@app.get(
    f"{settings.api_prefix}/sessions/{{session_id}}/documents",
    response_model=SessionDocumentsResponse,
)
def get_documents(
    session_id: Annotated[
        str, PathParameter(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)
    ],
) -> SessionDocumentsResponse:
    documents = list_session_documents(session_id)
    return SessionDocumentsResponse(
        session_id=session_id,
        total_chunks=sum(document["chunks"] for document in documents),
        documents=documents,
    )


@app.delete(
    f"{settings.api_prefix}/sessions/{{session_id}}/documents",
    response_model=DeleteDocumentsResponse,
)
def delete_documents(
    session_id: Annotated[
        str, PathParameter(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)
    ],
) -> DeleteDocumentsResponse:
    deleted = delete_session_documents(session_id)
    return DeleteDocumentsResponse(session_id=session_id, deleted_chunks=deleted)


@app.post(f"{settings.api_prefix}/interview/question", response_model=InterviewQuestionResponse)
def api_interview_question(data: InterviewQuestionRequest) -> InterviewQuestionResponse:
    return generate_interview_question(
        data.target_role, data.seniority.value, data.focus_skills
    )


@app.post(f"{settings.api_prefix}/interview/evaluate", response_model=InterviewFeedbackResponse)
def api_interview_evaluate(data: InterviewAnswer) -> InterviewFeedbackResponse:
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


@app.post(f"{settings.api_prefix}/interview-simulator", response_model=InterviewFeedbackResponse)
def api_interview_simulator(data: InterviewAnswer) -> InterviewFeedbackResponse:
    """Backwards-compatible endpoint name from the first version."""
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


@app.post(f"{settings.api_prefix}/rewrite-resume", response_model=ResumeRewriteResponse)
def api_rewrite_resume(data: ResumeRewriteRequest) -> ResumeRewriteResponse:
    return rewrite_resume_bullets(
        data.resume_bullets, data.target_role, data.job_description_text
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
