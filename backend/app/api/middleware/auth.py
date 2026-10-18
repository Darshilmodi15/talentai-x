"""Supabase JWT authentication and server-side role authorization."""
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from jose import jwk, jwt
try:
    from jose.exceptions import JWTError
except ImportError:
    from jose import JWTError
from sqlalchemy import select

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.db.models.models import UserProfile

bearer_scheme = HTTPBearer(auto_error=False)
_jwks_cache: dict[str, object] | None = None


async def _get_jwks() -> dict[str, object]:
    global _jwks_cache
    if _jwks_cache is None:
        if not settings.SUPABASE_JWKS_URL:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase JWT signing keys are not configured on the server.",
            )
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(settings.SUPABASE_JWKS_URL)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("keys"), list):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase returned an invalid JWT signing-key document.",
            )
        _jwks_cache = payload
    return _jwks_cache


async def _decode_token(token: str) -> dict[str, object]:
    global _jwks_cache
    header = jwt.get_unverified_header(token)
    algorithm = header.get("alg")
    issuer = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1" if settings.SUPABASE_URL else None
    options = {"verify_iss": bool(issuer)}

    if algorithm == "HS256":
        if not settings.SUPABASE_JWT_SECRET:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase legacy JWT secret is not configured on the server.",
            )
        return jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience=settings.SUPABASE_JWT_AUDIENCE,
            issuer=issuer,
            options=options,
        )

    if algorithm == "ES256":
        key_id = header.get("kid")
        if not isinstance(key_id, str):
            raise JWTError("Supabase token is missing its signing-key id.")
        jwks = await _get_jwks()
        keys = jwks["keys"]
        signing_key = next(
            (key for key in keys if isinstance(key, dict) and key.get("kid") == key_id),
            None,
        )
        if not isinstance(signing_key, dict):
            _jwks_cache = None
            jwks = await _get_jwks()
            signing_key = next(
                (key for key in jwks["keys"] if isinstance(key, dict) and key.get("kid") == key_id),
                None,
            )
        if not isinstance(signing_key, dict):
            raise JWTError("Supabase signing key was not found.")
        return jwt.decode(
            token,
            jwk.construct(signing_key),
            algorithms=["ES256"],
            audience=settings.SUPABASE_JWT_AUDIENCE,
            issuer=issuer,
            options=options,
        )

    raise JWTError(f"Unsupported Supabase token algorithm: {algorithm}")


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
    if not settings.SUPABASE_JWT_SECRET and not settings.SUPABASE_JWKS_URL:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase JWT verification is not configured on the server.",
        )

    try:
        claims = await _decode_token(credentials.credentials)
        user_id = UUID(str(claims["sub"]))
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve Supabase JWT signing keys.",
        ) from exc
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
