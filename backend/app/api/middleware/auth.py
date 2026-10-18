"""Supabase JWT authentication and server-side role authorization."""
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
try:
    from jose.exceptions import JWTError
except ImportError:
    from jose import JWTError
from sqlalchemy import select

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.db.models.models import UserProfile

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: UUID
    email: str | None
    role: str


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
) -> AuthenticatedUser:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not settings.SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase JWT verification is not configured on the server.",
        )

    try:
        claims = jwt.decode(
            credentials.credentials,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience=settings.SUPABASE_JWT_AUDIENCE,
            issuer=f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1" if settings.SUPABASE_URL else None,
            options={"verify_iss": bool(settings.SUPABASE_URL)},
        )
        user_id = UUID(str(claims["sub"]))
    except (JWTError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    async with AsyncSessionLocal() as db:
        profile = await db.get(UserProfile, user_id)
        if not profile:
            profile = UserProfile(
                id=user_id,
                email=claims.get("email"),
                role="individual",
            )
            db.add(profile)
            await db.commit()
        return AuthenticatedUser(id=user_id, email=profile.email, role=profile.role)


async def require_recruiter(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> AuthenticatedUser:
    if user.role not in {"recruiter", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recruiter access is required.")
    return user


async def require_admin(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> AuthenticatedUser:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access is required.")
    return user


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
RecruiterUser = Annotated[AuthenticatedUser, Depends(require_recruiter)]
AdminUser = Annotated[AuthenticatedUser, Depends(require_admin)]
