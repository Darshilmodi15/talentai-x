"""Candidate Service - persists parsed pipeline state to PostgreSQL"""
import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.models import Candidate, Experience, Education, EmergingSkill
from app.core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)


async def save_candidate_from_state(
    state: PipelineState,
    job_id: str,
    db: AsyncSession,
) -> uuid.UUID:
    """
    Take the completed pipeline state and persist everything to the DB.
    Returns the candidate UUID.

    Raises ValueError if parsed data is completely empty (defense in depth).
    """
    parsed = state.get("parsed") or {}

    # ── Pre-save validation ──
    key_fields = ["name", "email", "skills", "experience", "education"]
    filled = [f for f in key_fields if parsed.get(f)]
    logger.info(
        f"save_candidate_from_state: job_id={job_id}, "
        f"filled_fields={filled}, "
        f"name={parsed.get('name')}, "
        f"email={parsed.get('email')}, "
        f"phone={parsed.get('phone')}, "
        f"skills_count={len(parsed.get('skills', []))}, "
        f"experience_count={len(parsed.get('experience', []))}, "
        f"education_count={len(parsed.get('education', []))}"
    )

    if not filled:
        raise ValueError(
            f"Cannot save candidate with zero extracted data for job {job_id}. "
            f"All key fields (name, email, skills, experience, education) are empty. "
            f"This should not happen — the caller should check overall_status first."
        )
    candidate_id = uuid.uuid4()

    # Main candidate record
    candidate = Candidate(
        id=candidate_id,
        parse_job_id=uuid.UUID(job_id),
        name=parsed.get("name"),
        email=parsed.get("email"),
        phone=parsed.get("phone"),
        location=parsed.get("location"),
        summary=parsed.get("summary"),
        parsed_data={
            **parsed,
            "layout_type": state.get("layout_type"),
        },
        skills_canonical=state.get("skills_canonical", []),
        github_url=parsed.get("github_url"),
        linkedin_url=parsed.get("linkedin_url"),
        resolved_platforms=state.get("platform_profiles", {}),
        enriched_skills=state.get("enriched_skills", []),
        ai_content_probability=state.get("ai_content_probability", 0.0),
        parse_confidence=state.get("parse_confidence", 0.0),
        resume_language=state.get("resume_language", "en"),
        experience_months_total=state.get("experience_months_total", 0),
    )
    db.add(candidate)

    # Experience records
    for exp_data in parsed.get("experience", []):
        exp = Experience(
            candidate_id=candidate_id,
            company=exp_data.get("company"),
            role=exp_data.get("role"),
            start_date=exp_data.get("start"),
            end_date=exp_data.get("end"),
            duration_months=exp_data.get("duration_months", 0),
            bullets=exp_data.get("bullets", []),
            skills_mentioned=exp_data.get("skills_mentioned", []),
        )
        db.add(exp)

    # Education records
    for edu_data in parsed.get("education", []):
        edu = Education(
            candidate_id=candidate_id,
            institution=edu_data.get("institution"),
            degree=edu_data.get("degree"),
            field=edu_data.get("field"),
            year=edu_data.get("year"),
            gpa=edu_data.get("gpa"),
        )
        db.add(edu)

    # Log emerging skills (Agent 5 input)
    for raw_skill in state.get("emerging_skills_found", []):
        # Check if already logged
        from sqlalchemy import select
        existing = await db.execute(
            select(EmergingSkill).where(EmergingSkill.raw_skill == raw_skill.lower())
        )
        existing_record = existing.scalar_one_or_none()
        if existing_record:
            existing_record.count += 1
        else:
            emerging = EmergingSkill(
                raw_skill=raw_skill.lower(),
                count=1,
                contexts=[parsed.get("name", "unknown")],
                status="pending",
            )
            db.add(emerging)

    await db.flush()
    return candidate_id
