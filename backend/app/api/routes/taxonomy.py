"""
Taxonomy API Routes
GET  /api/v1/skills/taxonomy          — browse full taxonomy
GET  /api/v1/skills/taxonomy/search   — search skills
GET  /api/v1/skills/taxonomy/emerging — list emerging skills for review
POST /api/v1/skills/taxonomy/approve  — approve an emerging skill
GET  /api/v1/skills/taxonomy/categories — list all categories
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models.models import SkillTaxonomy, EmergingSkill
from app.api.middleware.auth import verify_api_key

router = APIRouter()


@router.get(
    "/skills/taxonomy",
    summary="Browse the skill taxonomy",
    description="Returns all skills in the taxonomy, optionally filtered by category or parent.",
)
async def browse_taxonomy(
    category: Optional[str] = Query(None, description="Filter: technical, soft, domain"),
    parent: Optional[str] = Query(None, description="Filter by parent skill"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    stmt = select(SkillTaxonomy).order_by(SkillTaxonomy.category, SkillTaxonomy.name)

    if category:
        stmt = stmt.where(SkillTaxonomy.category == category)
    if parent:
        stmt = stmt.where(SkillTaxonomy.parent == parent)

    total = await db.scalar(select(func.count()).select_from(SkillTaxonomy))
    offset = (page - 1) * page_size
    result = await db.execute(stmt.offset(offset).limit(page_size))
    skills = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "skills": [
            {
                "id": str(s.id),
                "name": s.name,
                "canonical_name": s.canonical_name,
                "category": s.category,
                "parent": s.parent,
                "synonyms": s.synonyms,
                "source": s.source,
            }
            for s in skills
        ],
    }


@router.get(
    "/skills/taxonomy/search",
    summary="Search skill taxonomy",
)
async def search_taxonomy(
    q: str = Query(..., min_length=2, description="Search query"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    stmt = select(SkillTaxonomy).where(
        SkillTaxonomy.name.ilike(f"%{q}%") |
        SkillTaxonomy.canonical_name.ilike(f"%{q}%")
    ).limit(20)

    result = await db.execute(stmt)
    skills = result.scalars().all()

    return {
        "query": q,
        "results": [
            {
                "name": s.name,
                "canonical_name": s.canonical_name,
                "category": s.category,
                "parent": s.parent,
                "synonyms": s.synonyms,
            }
            for s in skills
        ],
    }


@router.get(
    "/skills/taxonomy/categories",
    summary="Get all skill categories and counts",
)
async def get_categories(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    result = await db.execute(
        select(SkillTaxonomy.category, func.count(SkillTaxonomy.id))
        .group_by(SkillTaxonomy.category)
    )
    rows = result.all()
    return {
        "categories": [{"name": cat, "count": count} for cat, count in rows]
    }


@router.get(
    "/skills/taxonomy/emerging",
    summary="List emerging skills pending review",
    description="Skills found in resumes that are not yet in the taxonomy. Agent 5 logs these.",
)
async def list_emerging_skills(
    status: str = Query("pending", description="Filter: pending, approved, rejected"),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    result = await db.execute(
        select(EmergingSkill)
        .where(EmergingSkill.status == status)
        .order_by(EmergingSkill.count.desc())
        .limit(100)
    )
    skills = result.scalars().all()

    return {
        "status_filter": status,
        "count": len(skills),
        "emerging_skills": [
            {
                "id": str(s.id),
                "raw_skill": s.raw_skill,
                "occurrences": s.count,
                "proposed_canonical": s.proposed_canonical,
                "status": s.status,
                "first_seen": s.created_at.isoformat(),
            }
            for s in skills
        ],
    }


class ApproveSkillRequest(BaseModel):
    emerging_skill_id: str
    canonical_name: str
    category: str
    parent: Optional[str] = None
    synonyms: list[str] = []


@router.post(
    "/skills/taxonomy/approve",
    summary="Approve an emerging skill and add to taxonomy",
)
async def approve_emerging_skill(
    body: ApproveSkillRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    import uuid
    try:
        skill_uuid = uuid.UUID(body.emerging_skill_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID")

    emerging = await db.get(EmergingSkill, skill_uuid)
    if not emerging:
        raise HTTPException(status_code=404, detail="Emerging skill not found")

    # Add to taxonomy
    new_skill = SkillTaxonomy(
        name=body.canonical_name,
        canonical_name=body.canonical_name,
        category=body.category,
        parent=body.parent,
        synonyms=body.synonyms + [emerging.raw_skill],
        source="auto_discovered",
    )
    db.add(new_skill)

    # Mark as approved
    emerging.status = "approved"
    emerging.proposed_canonical = body.canonical_name
    await db.commit()

    return {
        "message": f"Skill '{body.canonical_name}' added to taxonomy",
        "skill_id": str(new_skill.id),
    }
