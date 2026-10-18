"""Jobs API Routes"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.db.database import get_db
from app.db.models.models import Job
from app.api.middleware.auth import CurrentUser

router = APIRouter()


class CreateJobRequest(BaseModel):
    title: str
    description: str
    company: Optional[str] = None
    experience_years_min: int = 0


@router.post("/jobs", summary="Save a job description for repeated matching")
async def create_job(
    body: CreateJobRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    job = Job(
        title=body.title,
        description=body.description,
        company=body.company,
        experience_years_min=body.experience_years_min,
        owner_user_id=user.id,
    )
    db.add(job)
    await db.commit()
    return {"job_id": str(job.id), "title": job.title}


@router.get("/jobs", summary="List saved jobs")
async def list_jobs(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job).where(Job.owner_user_id == user.id).order_by(Job.created_at.desc()).limit(50)
    )
    jobs = result.scalars().all()
    return {
        "jobs": [
            {"id": str(j.id), "title": j.title, "company": j.company,
             "created_at": j.created_at.isoformat()}
            for j in jobs
        ]
    }


@router.get("/saved-jobs/{job_id}", summary="Get a saved job")
async def get_job(
    job_id: str,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    job = await db.scalar(select(Job).where(Job.id == job_uuid, Job.owner_user_id == user.id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": str(job.id), "title": job.title,
        "description": job.description, "company": job.company,
        "required_skills": job.required_skills,
        "experience_years_min": job.experience_years_min,
    }
