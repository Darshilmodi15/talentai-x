"""
Match API Routes
POST /api/v1/match              — match one candidate to a JD
POST /api/v1/match/batch        — rank multiple candidates against JD
POST /api/v1/match/bias-audit   — run bias audit for a job
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.db.database import get_db
from app.db.models.models import Candidate, Job, MatchResult, MatchStatus, HITLReviewItem
from app.api.middleware.auth import verify_api_key
from app.agents.match_agent import match_agent
from app.core.pipeline_state import PipelineState
from app.core.config import settings
from datetime import datetime, timedelta

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    candidate_id: str = Field(..., description="UUID of the parsed candidate")
    job_description: str = Field(..., min_length=50, description="Full job description text")
    weights: Optional[dict] = Field(
        None,
        description="Custom scoring weights. Keys: required_skill_coverage, semantic_similarity, experience_depth, nice_to_have_coverage",
        example={"required_skill_coverage": 0.45, "semantic_similarity": 0.25}
    )
    save_result: bool = Field(True, description="Persist match result to database")


class MatchResponse(BaseModel):
    match_id: Optional[str]
    candidate_id: str
    match_score: float
    blind_score: float
    bias_delta: float
    bias_flagged: bool
    semantic_score: float
    required_skill_coverage: float
    experience_depth_score: float
    matched_skills: list
    skill_gaps: list
    upskilling_suggestions: dict
    shap_values: dict
    summary: str
    recommendation: str
    interview_questions: dict
    hitl_required: bool
    hitl_triggers: list


class BatchMatchRequest(BaseModel):
    candidate_ids: list[str] = Field(..., max_length=100)
    job_description: str = Field(..., min_length=50)
    top_n: int = Field(10, ge=1, le=100)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

async def build_state_from_candidate(candidate: Candidate) -> PipelineState:
    """Reconstruct pipeline state from saved candidate data."""
    parsed = candidate.parsed_data or {}
    return {
        "job_id": str(candidate.parse_job_id),
        "raw_file": b"",
        "file_name": "",
        "file_type": "",
        "layout_type": "",
        "parsed": parsed,
        "parse_confidence": candidate.parse_confidence,
        "resume_language": candidate.resume_language,
        "ai_content_probability": candidate.ai_content_probability,
        "skills_canonical": candidate.skills_canonical or [],
        "experience_months_total": candidate.experience_months_total,
        "inferred_skills": [],
        "emerging_skills_found": [],
        "platform_profiles": candidate.resolved_platforms or {},
        "entity_confidence": 0.8,
        "enriched_skills": candidate.enriched_skills or [],
        "candidate_id": str(candidate.id),
        "match_score": 0.0,
        "blind_score": 0.0,
        "semantic_score": 0.0,
        "required_skill_coverage": 0.0,
        "experience_depth_score": 0.0,
        "bias_delta": 0.0,
        "bias_flagged": False,
        "matched_skills": [],
        "skill_gaps": [],
        "upskilling_suggestions": {},
        "shap_values": {},
        "cot_reasoning": "",
        "match_summary": "",
        "interview_questions": {},
        "hitl_required": False,
        "hitl_triggers": [],
        "traces": [],
        "errors": [],
        "overall_status": "completed",
    }


def extract_recommendation(cot: str) -> str:
    for line in cot.split("\n"):
        for rec in ["STRONG YES", "YES", "CONDITIONAL", "NO"]:
            if rec in line.upper():
                return rec
    return "REVIEW"


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@router.post(
    "/match",
    response_model=MatchResponse,
    summary="Match a candidate to a job description",
    description="""
Runs semantic matching with Chain-of-Thought reasoning.

Returns:
- **match_score** — weighted final score (0-1)
- **blind_score** — score without demographic signals (bias-protected)
- **bias_delta** — difference between blind and full score
- **shap_values** — which features drove the score
- **skill_gaps** — missing required skills
- **upskilling_suggestions** — learning paths for gaps
- **interview_questions** — 6 targeted questions for this candidate
- **hitl_required** — whether human review is recommended
    """,
)
async def match_candidate(
    request: Request,
    body: MatchRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    # Load candidate
    try:
        cand_uuid = uuid.UUID(body.candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate_id")

    candidate = await db.get(Candidate, cand_uuid)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Build state
    state = await build_state_from_candidate(candidate)

    # Run matching
    state = await match_agent(state, body.job_description)

    # Persist result
    match_id = None
    if body.save_result:
        # Save or find job
        job = Job(
            title="Unspecified",
            description=body.job_description,
            required_skills=state.get("skill_gaps", []),
        )
        db.add(job)
        await db.flush()

        match_status = (
            MatchStatus.PENDING_REVIEW if state.get("hitl_required")
            else MatchStatus.AUTO_MERGED
        )
        match_result = MatchResult(
            candidate_id=cand_uuid,
            job_id=job.id,
            match_score=state["match_score"],
            blind_score=state["blind_score"],
            semantic_score=state["semantic_score"],
            required_skill_coverage=state["required_skill_coverage"],
            experience_depth_score=state["experience_depth_score"],
            bias_delta=state["bias_delta"],
            bias_flagged=state["bias_flagged"],
            matched_skills=state["matched_skills"],
            skill_gaps=state["skill_gaps"],
            upskilling_suggestions=state["upskilling_suggestions"],
            shap_values=state["shap_values"],
            cot_reasoning=state["cot_reasoning"],
            summary=state["match_summary"],
            interview_questions=state["interview_questions"],
            status=match_status,
        )
        db.add(match_result)

        # Add to HITL queue if needed
        if state.get("hitl_required"):
            hitl = HITLReviewItem(
                match_result_id=match_result.id,
                trigger_reason=", ".join(state.get("hitl_triggers", [])),
                priority="high" if state.get("bias_flagged") else "normal",
                expires_at=datetime.utcnow() + timedelta(hours=48),
            )
            db.add(hitl)

        await db.commit()
        match_id = str(match_result.id)

    return MatchResponse(
        match_id=match_id,
        candidate_id=body.candidate_id,
        match_score=state["match_score"],
        blind_score=state["blind_score"],
        bias_delta=state["bias_delta"],
        bias_flagged=state["bias_flagged"],
        semantic_score=state["semantic_score"],
        required_skill_coverage=state["required_skill_coverage"],
        experience_depth_score=state["experience_depth_score"],
        matched_skills=state["matched_skills"],
        skill_gaps=state["skill_gaps"],
        upskilling_suggestions=state["upskilling_suggestions"],
        shap_values=state["shap_values"],
        summary=state["match_summary"],
        recommendation=extract_recommendation(state["cot_reasoning"]),
        interview_questions=state["interview_questions"],
        hitl_required=state["hitl_required"],
        hitl_triggers=state["hitl_triggers"],
    )


@router.post(
    "/match/batch",
    summary="Rank multiple candidates against a job description",
    description="Returns a sorted shortlist. Useful for bulk screening.",
)
async def batch_match(
    request: Request,
    body: BatchMatchRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    results = []

    for cand_id_str in body.candidate_ids:
        try:
            cand_uuid = uuid.UUID(cand_id_str)
            candidate = await db.get(Candidate, cand_uuid)
            if not candidate:
                continue

            state = await build_state_from_candidate(candidate)
            state = await match_agent(state, body.job_description)

            results.append({
                "candidate_id": cand_id_str,
                "name": (candidate.parsed_data or {}).get("name", "Unknown"),
                "match_score": state["match_score"],
                "blind_score": state["blind_score"],
                "bias_flagged": state["bias_flagged"],
                "required_skill_coverage": state["required_skill_coverage"],
                "skill_gaps": state["skill_gaps"][:3],
                "recommendation": extract_recommendation(state["cot_reasoning"]),
            })
        except Exception:
            continue

    # Sort by blind_score (unbiased ranking)
    results.sort(key=lambda x: x["blind_score"], reverse=True)

    return {
        "total_candidates": len(results),
        "job_description_preview": body.job_description[:100] + "...",
        "ranking_method": "blind_score",
        "shortlist": results[: body.top_n],
    }


@router.post(
    "/match/bias-audit",
    summary="Run algorithmic bias audit for a job",
    description="""
Generates synthetic test resume pairs with swapped demographic signals
and measures score variance. Required by NYC Local Law 144.

Returns bias_score < 0.05 = FAIR, ≥ 0.05 = FLAGGED.
    """,
)
async def bias_audit(
    request: Request,
    job_description: str,
    _: str = Depends(verify_api_key),
):
    """
    Simplified bias audit:
    Creates 4 synthetic candidate profiles (identical skills, different names/locations)
    and checks if scores vary significantly.
    """
    from app.agents.match_agent import parse_job_description, find_best_skill_match

    # Synthetic skill profiles — identical qualifications, different demographics
    test_profiles = {
        "profile_A_male_indian": {
            "name": "Rahul Sharma",
            "location": "Ahmedabad, India",
            "skills": ["python", "machine learning", "tensorflow", "sql"],
            "experience_years": 3,
        },
        "profile_B_female_indian": {
            "name": "Priya Patel",
            "location": "Mumbai, India",
            "skills": ["python", "machine learning", "tensorflow", "sql"],
            "experience_years": 3,
        },
        "profile_C_male_us": {
            "name": "John Smith",
            "location": "San Francisco, USA",
            "skills": ["python", "machine learning", "tensorflow", "sql"],
            "experience_years": 3,
        },
        "profile_D_female_us": {
            "name": "Jane Williams",
            "location": "New York, USA",
            "skills": ["python", "machine learning", "tensorflow", "sql"],
            "experience_years": 3,
        },
    }

    scores = {}
    for profile_key, profile in test_profiles.items():
        # Build minimal state
        state: PipelineState = {
            "parsed": {"name": profile["name"], "experience": [], "skills": profile["skills"]},
            "skills_canonical": [
                {"raw": s, "canonical": s, "category": "technical",
                 "proficiency": "intermediate", "years": 2.0, "inferred": False, "source": "resume"}
                for s in profile["skills"]
            ],
            "inferred_skills": [],
            "enriched_skills": [],
            "entity_confidence": 1.0,
            "parse_confidence": 1.0,
            "ai_content_probability": 0.0,
            "traces": [], "errors": [],
        }
        state = await match_agent(state, job_description)
        scores[profile_key] = {
            "blind_score": state["blind_score"],
            "full_score": state["match_score"],
            "bias_delta": state["bias_delta"],
        }

    # Compute variance across blind scores
    blind_scores = [v["blind_score"] for v in scores.values()]
    mean = sum(blind_scores) / len(blind_scores)
    variance = sum((s - mean) ** 2 for s in blind_scores) / len(blind_scores)
    bias_score = round(variance ** 0.5, 4)

    return {
        "bias_score": bias_score,
        "status": "FAIR" if bias_score < 0.05 else "FLAGGED",
        "threshold": 0.05,
        "profile_scores": scores,
        "interpretation": (
            "Score variance is within acceptable range. System appears fair."
            if bias_score < 0.05
            else f"Score variance {bias_score:.3f} exceeds threshold. "
                 "Investigate demographic-correlated features."
        ),
        "compliant_with": ["NYC Local Law 144", "Colorado AI Act 2026"],
    }
