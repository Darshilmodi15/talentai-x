"""Admin API Routes - API key management, HITL queue, system stats"""
import uuid
import hashlib
import secrets
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db
from app.db.models.models import APIKey, HITLReviewItem, MatchResult, Candidate, ParseJob
from app.api.middleware.auth import verify_api_key

router = APIRouter()


# ── API Key Management ────────────────────────────────────────────

class CreateAPIKeyRequest(BaseModel):
    name: str


@router.post("/admin/api-keys", summary="Create a new API key", tags=["Admin"])
async def create_api_key(
    body: CreateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
):
    """Creates a new API key. The raw key is shown ONCE — store it securely."""
    raw_key = f"tai_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = APIKey(name=body.name, key_hash=key_hash)
    db.add(api_key)
    await db.commit()

    return {
        "api_key": raw_key,
        "key_id": str(api_key.id),
        "name": body.name,
        "warning": "Store this key securely. It will not be shown again.",
    }


@router.get("/admin/api-keys", summary="List all API keys", tags=["Admin"])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    result = await db.execute(select(APIKey).order_by(APIKey.created_at.desc()))
    keys = result.scalars().all()
    return {
        "api_keys": [
            {"id": str(k.id), "name": k.name, "is_active": k.is_active,
             "created_at": k.created_at.isoformat(), "last_used": k.last_used.isoformat() if k.last_used else None}
            for k in keys
        ]
    }


# ── HITL Queue ────────────────────────────────────────────────────

@router.get("/admin/hitl-queue", summary="Get HITL review queue", tags=["Admin"])
async def get_hitl_queue(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    result = await db.execute(
        select(HITLReviewItem)
        .where(HITLReviewItem.resolved == False)
        .order_by(HITLReviewItem.created_at.desc())
        .limit(50)
    )
    items = result.scalars().all()
    return {
        "pending_count": len(items),
        "items": [
            {
                "id": str(i.id),
                "match_result_id": str(i.match_result_id),
                "trigger_reason": i.trigger_reason,
                "priority": i.priority,
                "expires_at": i.expires_at.isoformat(),
                "created_at": i.created_at.isoformat(),
            }
            for i in items
        ],
    }


@router.post("/admin/hitl-queue/{item_id}/resolve", summary="Resolve a HITL review item", tags=["Admin"])
async def resolve_hitl_item(
    item_id: str,
    decision: str,
    notes: str = "",
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    try:
        item_uuid = uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid item_id")

    item = await db.get(HITLReviewItem, item_uuid)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item.resolved = True
    match = await db.get(MatchResult, item.match_result_id)
    if match:
        match.human_reviewed = True
        match.reviewer_notes = f"Decision: {decision}. {notes}"

    await db.commit()
    return {"message": "Resolved", "decision": decision}


# ── System Stats ──────────────────────────────────────────────────

@router.get("/admin/stats", summary="System statistics", tags=["Admin"])
async def system_stats(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    total_candidates = await db.scalar(select(func.count()).select_from(Candidate))
    total_jobs = await db.scalar(select(func.count()).select_from(ParseJob))
    total_matches = await db.scalar(select(func.count()).select_from(MatchResult))
    hitl_pending = await db.scalar(
        select(func.count()).select_from(HITLReviewItem)
        .where(HITLReviewItem.resolved == False)
    )
    bias_flagged = await db.scalar(
        select(func.count()).select_from(MatchResult)
        .where(MatchResult.bias_flagged == True)
    )

    return {
        "total_candidates_parsed": total_candidates,
        "total_parse_jobs": total_jobs,
        "total_matches_run": total_matches,
        "hitl_queue_pending": hitl_pending,
        "bias_flags_raised": bias_flagged,
        "system_status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }
