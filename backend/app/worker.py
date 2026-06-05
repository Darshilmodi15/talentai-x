"""Celery Worker for async batch resume processing"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "talentai_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=120,  # 2 min soft limit per resume
    task_time_limit=180,       # 3 min hard limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(bind=True, max_retries=3)
def process_resume_celery(self, job_id: str, file_path: str, file_name: str, file_type: str):
    """Celery task for batch resume processing."""
    import asyncio

    async def _run():
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        from app.agents.orchestrator import run_parse_pipeline
        from app.db.database import AsyncSessionLocal
        from app.services.candidate_service import save_candidate_from_state
        from app.db.models.models import ParseJob, ProcessingStatus
        from datetime import datetime
        import uuid

        state = await run_parse_pipeline(
            raw_file=file_bytes,
            file_name=file_name,
            file_type=file_type,
            job_id=job_id,
        )

        async with AsyncSessionLocal() as db:
            candidate_id = await save_candidate_from_state(state, job_id, db)
            job = await db.get(ParseJob, uuid.UUID(job_id))
            if job:
                job.status = ProcessingStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.traces = state.get("traces", [])
            await db.commit()

        return {"candidate_id": str(candidate_id), "status": "completed"}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=10)
