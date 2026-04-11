"""
Candidates API Routes
GET  /api/v1/candidates                 — list all candidates
GET  /api/v1/candidates/{id}            — full candidate profile
GET  /api/v1/candidates/{id}/skills     — normalized skill profile
GET  /api/v1/candidates/{id}/matches    — all match results for candidate
DELETE /api/v1/candidates/{id}          — delete candidate (GDPR)
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.db.models.models import Candidate, MatchResult
from app.api.middleware.auth import verify_api_key

router = APIRouter()


@router.get(
    "/candidates",
    summary="List all parsed candidates",
)
async def list_candidates(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or email"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    offset = (page - 1) * page_size
    stmt = select(Candidate).order_by(Candidate.created_at.desc())

    if search:
        stmt = stmt.where(
            Candidate.name.ilike(f"%{search}%") |
            Candidate.email.ilike(f"%{search}%")
        )

    total_stmt = select(func.count()).select_from(Candidate)
    total = await db.scalar(total_stmt)

    result = await db.execute(stmt.offset(offset).limit(page_size))
    candidates = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "candidates": [
            {
                "id": str(c.id),
                "name": c.name,
                "email": c.email,
                "location": c.location,
                "skills_count": len(c.skills_canonical or []),
                "experience_months": c.experience_months_total,
                "parse_confidence": c.parse_confidence,
                "created_at": c.created_at.isoformat(),
            }
            for c in candidates
        ],
    }


@router.get(
    "/candidates/{candidate_id}",
    summary="Get full candidate profile",
)
async def get_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        cand_uuid = uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id")

    candidate = await db.get(Candidate, cand_uuid)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    parsed = candidate.parsed_data or {}
    return {
        "id": str(candidate.id),
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "location": candidate.location,
        "summary": candidate.summary,
        "resume_language": candidate.resume_language,
        "ai_content_probability": candidate.ai_content_probability,
        "parse_confidence": candidate.parse_confidence,
        "experience_months_total": candidate.experience_months_total,
        "experience": parsed.get("experience", []),
        "education": parsed.get("education", []),
        "certifications": parsed.get("certifications", []),
        "projects": parsed.get("projects", []),
        "skills_canonical": candidate.skills_canonical,
        "enriched_skills": candidate.enriched_skills,
        "platform_profiles": candidate.resolved_platforms,
        "created_at": candidate.created_at.isoformat(),
    }


@router.get(
    "/candidates/{candidate_id}/skills",
    summary="Get normalized skill profile",
    description="""
Returns the full skill profile with canonical names, proficiency levels,
years of experience, source (resume/github/inferred), and inferred skills.
    """,
)
async def get_candidate_skills(
    candidate_id: str,
    include_inferred: bool = Query(True, description="Include skills inferred from the implication graph"),
    source_filter: Optional[str] = Query(None, description="Filter by source: resume, github, linkedin, inferred"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        cand_uuid = uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id")

    candidate = await db.get(Candidate, cand_uuid)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    skills = list(candidate.skills_canonical or [])
    enriched = list(candidate.enriched_skills or [])

    if include_inferred:
        # Merge enriched skills
        existing_canonicals = {s.get("canonical") for s in skills}
        for es in enriched:
            if es.get("canonical") not in existing_canonicals:
                skills.append(es)

    if source_filter:
        skills = [s for s in skills if s.get("source") == source_filter]

    # Group by category
    technical = [s for s in skills if s.get("category") == "technical"]
    soft = [s for s in skills if s.get("category") == "soft"]
    domain = [s for s in skills if s.get("category") == "domain"]

    return {
        "candidate_id": candidate_id,
        "name": candidate.name,
        "total_skills": len(skills),
        "by_category": {
            "technical": len(technical),
            "soft": len(soft),
            "domain": len(domain),
        },
        "by_proficiency": {
            level: len([s for s in skills if s.get("proficiency") == level])
            for level in ["expert", "advanced", "intermediate", "beginner", "inferred"]
        },
        "skills": skills,
        "platform_verification": {
            platform: len([s for s in skills if s.get("source") == platform])
            for platform in ["github", "linkedin", "stackoverflow"]
        },
    }


@router.get(
    "/candidates/{candidate_id}/matches",
    summary="Get all match results for a candidate",
)
async def get_candidate_matches(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        cand_uuid = uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id")

    result = await db.execute(
        select(MatchResult)
        .where(MatchResult.candidate_id == cand_uuid)
        .order_by(MatchResult.created_at.desc())
    )
    matches = result.scalars().all()

    return {
        "candidate_id": candidate_id,
        "total_matches": len(matches),
        "matches": [
            {
                "match_id": str(m.id),
                "match_score": m.match_score,
                "blind_score": m.blind_score,
                "bias_flagged": m.bias_flagged,
                "required_skill_coverage": m.required_skill_coverage,
                "skill_gaps": m.skill_gaps[:5],
                "status": m.status.value,
                "human_reviewed": m.human_reviewed,
                "created_at": m.created_at.isoformat(),
            }
            for m in matches
        ],
    }


@router.delete(
    "/candidates/{candidate_id}",
    status_code=204,
    summary="Delete candidate (GDPR right to erasure)",
)
async def delete_candidate(
    candidate_id: str,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        cand_uuid = uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id")

    candidate = await db.get(Candidate, cand_uuid)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    await db.delete(candidate)
    await db.commit()
    # Returns 204 No Content
