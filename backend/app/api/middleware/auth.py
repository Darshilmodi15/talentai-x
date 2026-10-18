"""Authentication & Authorization Middleware

Provides:
- Supabase JWT validation (signature, issuer, audience, subject, and expiry)
- Database-backed user profiles with server-controlled roles ('individual', 'recruiter', 'admin')
- API key verification for automated and administrative workflows
- Role-based access control (RBAC) dependencies
"""
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTClaimsError, JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.db.models.models import APIKey, UserProfile, UserRole

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    id: uuid.UUID
    email: Optional[str] = None
    role: str = UserRole.INDIVIDUAL.value
    auth_method: str = "jwt"  # "jwt", "api_key", or "dev_bypass"

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN.value

    @property
    def is_recruiter(self) -> bool:
        return self.role in [UserRole.RECRUITER.value, UserRole.ADMIN.value]


async def _get_or_create_user_profile(
    db: AsyncSession,
    user_id: uuid.UUID,
    email: Optional[str] = None,
    default_role: str = UserRole.INDIVIDUAL.value,
) -> UserProfile:
    """Load or create a user profile in PostgreSQL."""
    profile = await db.get(UserProfile, user_id)
    if not profile:
        profile = UserProfile(
            id=user_id,
            email=email,
            role=default_role,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    elif email and profile.email != email:
        profile.email = email
        await db.commit()
    return profile


async def get_current_user(
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> AuthenticatedUser:
    """
    Authenticate request via Supabase JWT Bearer token or server-side API Key.

    1. Checks 'Authorization: Bearer <token>' for Supabase JWT:
       - Validates HS256 signature using SUPABASE_JWT_SECRET
       - Validates audience ('authenticated'), issuer, expiration, and UUID subject
       - Retrieves or provisions server-controlled UserProfile record (defaulting to 'individual')
    2. Fallback to 'X-API-Key' for machine-to-machine integrations:
       - In production, validates against SHA-256 hashed keys in database
       - In development only, allows local dev key with deterministic admin profile
    """
    # ── 1. Supabase JWT Authentication ────────────────────────────
    if auth_credentials and auth_credentials.credentials:
        token = auth_credentials.credentials.strip()

        if not settings.SUPABASE_JWT_SECRET:
            logger.error("SUPABASE_JWT_SECRET is not configured on the backend.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication misconfiguration: SUPABASE_JWT_SECRET is not configured on the backend.",
            )

        try:
            # Supabase tokens use HS256 symmetric signature with SUPABASE_JWT_SECRET
            options = {
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "verify_sub": True,
            }
            payload = jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options=options,
            )
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token has expired. Please log in again.",
            )
        except JWTClaimsError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid authentication token claims: {str(exc)}",
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token signature or malformed JWT.",
            )

        # Validate issuer if SUPABASE_URL is configured
        if settings.SUPABASE_URL:
            expected_iss_prefix = settings.SUPABASE_URL.rstrip("/")
            token_iss = payload.get("iss", "")
            if token_iss and not (
                token_iss.startswith(expected_iss_prefix)
                or token_iss == f"{expected_iss_prefix}/auth/v1"
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token issuer mismatch. Expected issuer for {expected_iss_prefix}.",
                )

        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token: missing 'sub' claim.",
            )

        try:
            user_uuid = uuid.UUID(sub)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token: 'sub' must be a valid UUID.",
            )

        email = payload.get("email")
        profile = await _get_or_create_user_profile(
            db=db,
            user_id=user_uuid,
            email=email,
            default_role=UserRole.INDIVIDUAL.value,
        )

        return AuthenticatedUser(
            id=profile.id,
            email=profile.email,
            role=profile.role,
            auth_method="jwt",
        )

    # ── 2. API Key Authentication ─────────────────────────────────
    if api_key:
        # Development bypass: strictly limited to ENVIRONMENT == "development"
        if settings.ENVIRONMENT == "development" and api_key == "dev_key_change_in_production":
            dev_user_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
            dev_profile = await _get_or_create_user_profile(
                db=db,
                user_id=dev_user_id,
                email="dev-admin@talentai.local",
                default_role=UserRole.ADMIN.value,
            )
            return AuthenticatedUser(
                id=dev_profile.id,
                email=dev_profile.email,
                role=dev_profile.role,
                auth_method="dev_bypass",
            )

        # Production database key verification
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
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

        # Update last_used timestamp
        key_record.last_used = datetime.now(timezone.utc)
        await db.commit()

        if key_record.user_id:
            profile = await db.get(UserProfile, key_record.user_id)
            if profile:
                return AuthenticatedUser(
                    id=profile.id,
                    email=profile.email,
                    role=profile.role,
                    auth_method="api_key",
                )

        # Standalone API key defaults to server-side admin role
        admin_system_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        admin_profile = await _get_or_create_user_profile(
            db=db,
            user_id=admin_system_id,
            email="system-api-key@talentai.local",
            default_role=UserRole.ADMIN.value,
        )
        return AuthenticatedUser(
            id=admin_system_id,
            email=admin_profile.email,
            role=admin_profile.role,
            auth_method="api_key",
        )

    # ── 3. Missing Credentials ────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide 'Authorization: Bearer <supabase_jwt>' or 'X-API-Key' header.",
    )


def require_role(*allowed_roles: str):
    """Factory dependency to enforce server-side role restrictions."""
    async def role_checker(
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required role in {list(allowed_roles)}; current role is '{current_user.role}'.",
            )
        return current_user

    return role_checker


require_recruiter_or_admin = require_role(
    UserRole.RECRUITER.value,
    UserRole.ADMIN.value,
)
require_admin = require_role(UserRole.ADMIN.value)


async def verify_api_key(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> str:
    """Backward compatibility helper returning the authenticated user ID as string."""
    return str(current_user.id)
