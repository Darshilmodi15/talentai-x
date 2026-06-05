"""
Parse API Routes
POST /api/v1/parse          — single resume upload
POST /api/v1/parse/batch    — multiple resumes (async)
GET  /api/v1/jobs/{job_id}  — poll job status
GET  /api/v1/jobs/{job_id}/trace — agent execution trace
"""
import uuid
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db.database import get_db
from app.db.models.models import ParseJob, ProcessingStatus, Candidate, Experience, Education
from app.api.middleware.auth import verify_api_key
from app.agents.orchestrator import run_parse_pipeline
from app.core.config import settings
from app.services.candidate_service import save_candidate_from_state

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────────────────────────

class ParseJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    file_name: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    file_name: str
    created_at: str
    completed_at: Optional[str]
    candidate_id: Optional[str]
    parse_confidence: Optional[float]
    layout_detected: Optional[str]
    resume_language: Optional[str]
    ai_content_probability: Optional[float]
    skills_found: Optional[int]
    error_message: Optional[str]
    partial_result: Optional[dict]


# ──────────────────────────────────────────────────────────────────
# Background Task
# ──────────────────────────────────────────────────────────────────

async def process_resume_task(
    job_id: str,
    file_bytes: bytes,
    file_name: str,
    file_type: str,
):
    """
    Background task: run pipeline and persist results.
    Called after the API has already returned 202.
    """
    from app.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        # Update status to processing
        job = await db.get(ParseJob, uuid.UUID(job_id))
        if not job:
            return
        job.status = ProcessingStatus.PROCESSING
        await db.commit()

        try:
            # Run the full agent pipeline
            state = await run_parse_pipeline(
                raw_file=file_bytes,
                file_name=file_name,
                file_type=file_type,
                job_id=job_id,
            )

            # Save candidate to DB
            candidate_id = await save_candidate_from_state(state, job_id, db)
            state["candidate_id"] = str(candidate_id)

            # Update job record
            job.status = (
                ProcessingStatus.COMPLETED
                if state.get("overall_status") == "completed"
                else ProcessingStatus.PARTIAL
            )
            job.completed_at = datetime.utcnow()
            job.traces = state.get("traces", [])
            await db.commit()

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}


@router.post(
    "/parse",
    response_model=ParseJobResponse,
    status_code=202,
    summary="Upload and parse a single resume",
    description="""
Upload a resume file (PDF, DOCX, or TXT) for async processing.
Returns a job_id immediately. Poll GET /api/v1/jobs/{job_id} for status.

**Processing pipeline:**
1. Layout detection (single-column, two-column, table, image)
2. Parallel text extraction (4 specialized LLM prompts)
3. Skill normalization against 5,000+ skill taxonomy
4. Entity resolution (GitHub, LinkedIn, other platforms)
5. AI content detection
    """,
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def parse_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    # Validate file type
    content_type = file.content_type or ""
    file_type = ALLOWED_EXTENSIONS.get(content_type)

    # Fallback: detect from extension
    if not file_type and file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext in ["pdf", "docx", "txt"]:
            file_type = ext

    if not file_type:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: PDF, DOCX, TXT. Got: {content_type}",
        )

    # Read file bytes
    file_bytes = await file.read()

    # Size check
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Create parse job record
    job_id = str(uuid.uuid4())
    job = ParseJob(
        id=uuid.UUID(job_id),
        status=ProcessingStatus.QUEUED,
        file_name=file.filename or "upload",
        file_type=file_type,
        file_path=f"uploads/{job_id}.{file_type}",
    )
    db.add(job)
    await db.commit()

    # Save file to disk
    os.makedirs("uploads", exist_ok=True)
    with open(f"uploads/{job_id}.{file_type}", "wb") as f:
        f.write(file_bytes)

    # Kick off background processing
    background_tasks.add_task(
        process_resume_task,
        job_id=job_id,
        file_bytes=file_bytes,
        file_name=file.filename or "upload",
        file_type=file_type,
    )

    return ParseJobResponse(
        job_id=job_id,
        status="queued",
        message="Resume queued for processing. Poll /api/v1/jobs/{job_id} for status.",
        file_name=file.filename or "upload",
    )


@router.post(
    "/parse/batch",
    status_code=202,
    summary="Upload multiple resumes for batch processing",
)
@limiter.limit("10/minute")
async def parse_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 files per batch")

    batch_id = str(uuid.uuid4())
    job_ids = []

    for file in files:
        content_type = file.content_type or ""
        file_type = ALLOWED_EXTENSIONS.get(content_type)
        if not file_type and file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext in ["pdf", "docx", "txt"]:
                file_type = ext
        if not file_type:
            continue

        file_bytes = await file.read()
        job_id = str(uuid.uuid4())

        job = ParseJob(
            id=uuid.UUID(job_id),
            status=ProcessingStatus.QUEUED,
            file_name=file.filename or "upload",
            file_type=file_type,
            file_path=f"uploads/{job_id}.{file_type}",
        )
        db.add(job)
        job_ids.append(job_id)

        os.makedirs("uploads", exist_ok=True)
        with open(f"uploads/{job_id}.{file_type}", "wb") as f:
            f.write(file_bytes)

        background_tasks.add_task(
            process_resume_task,
            job_id=job_id,
            file_bytes=file_bytes,
            file_name=file.filename or "upload",
            file_type=file_type,
        )

    await db.commit()

    return {
        "batch_id": batch_id,
        "total_files": len(job_ids),
        "job_ids": job_ids,
        "status": "queued",
        "message": f"{len(job_ids)} resumes queued. Poll each job_id for status.",
    }


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll parse job status",
)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    job = await db.get(ParseJob, job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get candidate if completed
    candidate_id = None
    parse_confidence = None
    layout = None
    lang = None
    ai_prob = None
    skills_count = None

    if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.PARTIAL]:
        from sqlalchemy import select
        result = await db.execute(
            select(Candidate).where(Candidate.parse_job_id == job_uuid)
        )
        candidate = result.scalar_one_or_none()
        if candidate:
            candidate_id = str(candidate.id)
            parse_confidence = candidate.parse_confidence
            lang = candidate.resume_language
            ai_prob = candidate.ai_content_probability
            skills_count = len(candidate.skills_canonical or [])
            parsed = candidate.parsed_data or {}
            layout = parsed.get("layout_type")

    return JobStatusResponse(
        job_id=job_id,
        status=job.status.value,
        file_name=job.file_name,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        candidate_id=candidate_id,
        parse_confidence=parse_confidence,
        layout_detected=layout,
        resume_language=lang,
        ai_content_probability=ai_prob,
        skills_found=skills_count,
        error_message=job.error_message,
        partial_result=None,
    )


@router.get(
    "/jobs/{job_id}/trace",
    summary="Get per-agent execution trace",
    description="Returns latency, quality score, retry count, and status for each agent.",
)
async def get_job_trace(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    job = await db.get(ParseJob, job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    traces = job.traces or []
    total_ms = sum(t.get("duration_ms", 0) for t in traces)

    return {
        "job_id": job_id,
        "total_duration_ms": total_ms,
        "agent_traces": traces,
        "overall_status": job.status.value,
    }
