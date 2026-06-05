"""API Key Authentication Middleware"""
import hashlib
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from X-API-Key header.
    For development: accepts the hardcoded dev key 'dev_key_change_in_production'.
    In production: validates against hashed keys in database.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    # Development bypass
    if settings.ENVIRONMENT == "development" and api_key == "dev_key_change_in_production":
        return api_key

    # Production: validate against DB
    from app.db.database import AsyncSessionLocal
    from app.db.models.models import APIKey

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True,
            )
        )
        key_record = result.scalar_one_or_none()

        if not key_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API key.",
            )

        # Update last_used
        key_record.last_used = datetime.utcnow()
        await db.commit()

    return api_key
