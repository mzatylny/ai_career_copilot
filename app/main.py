from __future__ import annotations

import logging
import os
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi import Path as PathParameter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
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
from app.auth import APIKeyRegistry, Principal, principal_from_request
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
    JobStatusResponse,
    ReadinessResponse,
    ResumeRewriteRequest,
    ResumeRewriteResponse,
    RoadmapRequest,
    RoadmapResponse,
    SessionCreateResponse,
    SessionDocumentsResponse,
    UploadJobResponse,
    UploadResponse,
)
from app.object_store import LocalObjectStore
from app.observability import configure_logging
from app.rag_engine import (
    delete_session_documents,
    list_session_documents,
    process_and_store_document,
    query_documents,
)
from app.rate_limit import SlidingWindowRateLimiter
from app.session_locks import SessionMutationCoordinator
from app.session_store import JobRecord, SessionStore
from app.tracing import configure_tracing
from app.utils import safe_filename

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

session_store = SessionStore(settings.session_database_path)
object_store = LocalObjectStore(settings.object_storage_path)
rate_limiter = SlidingWindowRateLimiter(settings.requests_per_minute)
session_mutations = SessionMutationCoordinator()

REQUEST_COUNT = Counter(
    "ai_copilot_http_requests_total",
    "HTTP requests handled by the service",
    ("method", "route", "status"),
)
REQUEST_DURATION = Histogram(
    "ai_copilot_http_request_duration_seconds",
    "HTTP request duration",
    ("method", "route"),
)
INGESTION_JOBS = Counter(
    "ai_copilot_ingestion_jobs_total",
    "Asynchronous document ingestion jobs",
    ("status",),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.environment in {"prod", "production"}:
        registry = APIKeyRegistry(settings)
        if not registry.enabled:
            raise RuntimeError("Production requires AI_COPILOT_TENANT_KEYS or AI_COPILOT_API_KEY")
        if not registry.meets_minimum_key_length():
            raise RuntimeError("Production API keys must contain at least 32 characters")
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "AI Career Copilot API: CV/job gap analysis, grounded RAG over PDFs, "
        "roadmap generation and interview coaching."
    ),
    docs_url="/docs" if settings.expose_docs else None,
    redoc_url="/redoc" if settings.expose_docs else None,
    openapi_url="/openapi.json" if settings.expose_docs else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allows_credentials,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
)
configure_tracing(
    app,
    endpoint=settings.otel_exporter_otlp_endpoint,
    service_name="ai-career-copilot",
)


@app.middleware("http")
async def security_observability_middleware(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", "").strip()[:80] or secrets.token_hex(12)
    is_protected_api = request.url.path.startswith(
        settings.api_prefix
    ) and request.url.path not in {f"{settings.api_prefix}/health", f"{settings.api_prefix}/ready"}

    registry = APIKeyRegistry(settings)
    principal = registry.resolve(request.headers.get("X-API-Key"))
    request.state.principal = principal

    identity = (
        principal.tenant_id
        if principal
        else f"unauthenticated:{request.client.host if request.client else 'unknown'}"
    )
    allowed, retry_after = rate_limiter.check(identity) if is_protected_api else (True, 0)
    if not allowed:
        response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        response.headers["Retry-After"] = str(retry_after)
    elif is_protected_api and principal is None:
        response = JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    else:
        response = await call_next(request)

    route = getattr(request.scope.get("route"), "path", request.url.path)
    duration = time.perf_counter() - started
    REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
    REQUEST_DURATION.labels(request.method, route).observe(duration)
    logger.info(
        "request completed",
        extra={
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": route,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1_000, 2),
            "tenant_id": principal.tenant_id if principal else None,
        },
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "img-src 'self' data:; connect-src 'self'"
    )
    if request.url.path.startswith(settings.api_prefix):
        response.headers["Cache-Control"] = "no-store"
    return response


def _require_session_owner(
    session_id: str,
    principal: Principal,
    *,
    allow_local_create: bool = False,
) -> None:
    create = allow_local_create and not principal.authenticated
    if not session_store.ensure_owner(session_id, principal.tenant_id, create=create):
        raise HTTPException(status_code=404, detail="Session not found")


def _job_response(job: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        session_id=job.session_id,
        filename=job.filename,
        status=job.status,
        chunks_indexed=job.chunks_indexed,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/", response_class=FileResponse, include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")


@app.get("/app.js", response_class=FileResponse, include_in_schema=False)
def app_javascript() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "app.js")


@app.get("/styles.css", response_class=FileResponse, include_in_schema=False)
def app_styles() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "styles.css")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


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


def _storage_path_ready(path: str | Path) -> bool:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


@app.get(f"{settings.api_prefix}/ready", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    try:
        metadata_ready = session_store.ping()
    except Exception as exc:
        logger.warning(
            "metadata readiness check failed",
            extra={"event": "readiness_failed", "error_type": type(exc).__name__},
        )
        metadata_ready = False
    checks = {
        "metadata_store": metadata_ready,
        "object_store": object_store.is_ready(),
        "vector_store_path": _storage_path_ready(settings.chroma_path),
    }
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail="Service dependencies are not ready")
    return ReadinessResponse(status="ready", checks=checks)


@app.post(f"{settings.api_prefix}/sessions", response_model=SessionCreateResponse, status_code=201)
def create_session(
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> SessionCreateResponse:
    return SessionCreateResponse(session_id=session_store.create_session(principal.tenant_id))


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


def _validate_pdf_filename(file: UploadFile) -> str:
    if not file.filename or not file.filename.casefold().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return safe_filename(file.filename)


def _process_document_serialized(
    file_path: str | Path,
    session_id: str,
    filename: str,
) -> int:
    with session_mutations.hold(session_id):
        return process_and_store_document(file_path, session_id, filename)


@app.post(f"{settings.api_prefix}/upload-document", response_model=UploadResponse)
async def upload_document(
    session_id: Annotated[str, Form(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)],
    file: Annotated[UploadFile, File(...)],
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> UploadResponse:
    _require_session_owner(session_id, principal, allow_local_create=True)
    safe_name = _validate_pdf_filename(file)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / safe_name
            await _save_bounded_upload(file, temp_path, settings.max_upload_bytes)
            try:
                chunks = await run_in_threadpool(
                    _process_document_serialized, temp_path, session_id, safe_name
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("PDF processing failed", extra={"event": "ingestion_failed"})
                raise HTTPException(
                    status_code=500,
                    detail="PDF processing failed. Please verify the file and try again.",
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


def _process_document_job(
    job_id: str, object_id: str, stored_path: Path, filename: str, session_id: str
) -> None:
    try:
        with session_mutations.hold(session_id):
            if session_store.get_job_for_worker(job_id) is None:
                return
            session_store.update_job(job_id, status="processing")
            try:
                chunks = process_and_store_document(stored_path, session_id, filename)
                if chunks == 0:
                    raise ValueError("No readable text found in this PDF")
                session_store.update_job(job_id, status="completed", chunks_indexed=chunks)
                INGESTION_JOBS.labels("completed").inc()
            except ValueError as exc:
                session_store.update_job(job_id, status="failed", error=str(exc)[:500])
                INGESTION_JOBS.labels("failed").inc()
            except Exception:
                logger.exception(
                    "background ingestion failed",
                    extra={"event": "ingestion_failed", "job_id": job_id},
                )
                session_store.update_job(
                    job_id,
                    status="failed",
                    error="Document processing failed",
                )
                INGESTION_JOBS.labels("failed").inc()
    finally:
        object_store.delete(object_id)


@app.post(
    f"{settings.api_prefix}/upload-document-async",
    response_model=UploadJobResponse,
    status_code=202,
)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    session_id: Annotated[str, Form(min_length=3, max_length=80, pattern=SESSION_ID_PATTERN)],
    file: Annotated[UploadFile, File(...)],
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> UploadJobResponse:
    _require_session_owner(session_id, principal, allow_local_create=True)
    safe_name = _validate_pdf_filename(file)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / safe_name
            await _save_bounded_upload(file, temp_path, settings.max_upload_bytes)
            job = session_store.create_job(session_id, principal.tenant_id, safe_name)
            stored_path = object_store.put(job.job_id, temp_path)
    finally:
        await file.close()

    INGESTION_JOBS.labels("queued").inc()
    background_tasks.add_task(
        _process_document_job,
        job.job_id,
        job.job_id,
        stored_path,
        safe_name,
        session_id,
    )
    return UploadJobResponse(
        job_id=job.job_id,
        session_id=session_id,
        filename=safe_name,
        status="queued",
    )


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}", response_model=JobStatusResponse)
def get_job(
    job_id: Annotated[str, PathParameter(min_length=8, max_length=80)],
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> JobStatusResponse:
    job = session_store.get_job(job_id, principal.tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job)


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat_with_documents(
    data: ChatRequest,
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> ChatResponse:
    _require_session_owner(data.session_id, principal, allow_local_create=True)
    chunks = await run_in_threadpool(query_documents, data.user_query, data.session_id)
    if not chunks:
        return ChatResponse(
            answer="I could not find relevant information in uploaded documents for this session.",
            sources=[],
            confidence="low",
            generation_mode="not_invoked",
            grounding_status="no_context",
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
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> SessionDocumentsResponse:
    _require_session_owner(session_id, principal, allow_local_create=True)
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
    principal: Annotated[Principal, Depends(principal_from_request)],
) -> DeleteDocumentsResponse:
    _require_session_owner(session_id, principal, allow_local_create=True)
    with session_mutations.hold(session_id):
        deleted = delete_session_documents(session_id)
        session_store.delete_session(session_id, principal.tenant_id)
    logger.info(
        "session documents deleted",
        extra={
            "event": "documents_deleted",
            "tenant_id": principal.tenant_id,
            "session_id": session_id,
        },
    )
    return DeleteDocumentsResponse(session_id=session_id, deleted_chunks=deleted)


@app.post(f"{settings.api_prefix}/interview/question", response_model=InterviewQuestionResponse)
def api_interview_question(data: InterviewQuestionRequest) -> InterviewQuestionResponse:
    return generate_interview_question(data.target_role, data.seniority.value, data.focus_skills)


@app.post(f"{settings.api_prefix}/interview/evaluate", response_model=InterviewFeedbackResponse)
def api_interview_evaluate(data: InterviewAnswer) -> InterviewFeedbackResponse:
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


@app.post(f"{settings.api_prefix}/interview-simulator", response_model=InterviewFeedbackResponse)
def api_interview_simulator(data: InterviewAnswer) -> InterviewFeedbackResponse:
    """Backwards-compatible endpoint name from the first version."""
    return evaluate_interview_answer(data.question, data.user_answer, data.target_role)


@app.post(f"{settings.api_prefix}/rewrite-resume", response_model=ResumeRewriteResponse)
def api_rewrite_resume(data: ResumeRewriteRequest) -> ResumeRewriteResponse:
    return rewrite_resume_bullets(data.resume_bullets, data.target_role, data.job_description_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
