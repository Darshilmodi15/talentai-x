from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "TalentAI-X"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SQL_ECHO: bool = False

    SECRET_KEY: str = "change_this_in_production"
    SUPABASE_URL: str = ""
    SUPABASE_JWT_SECRET: str = ""
    SUPABASE_JWT_AUDIENCE: str = "authenticated"
    SUPABASE_JWKS_URL: Optional[str] = None
    ALLOW_DEV_API_KEY: bool = False
    INDIVIDUAL_UPLOAD_LIMIT: int = 3
    RECRUITER_UPLOAD_LIMIT: int = 25
    UPLOAD_LIMIT_WINDOW_HOURS: int = 24
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://talentai-x.vercel.app",
    ]
    ALLOWED_ORIGIN_REGEX: Optional[str] = r"https://.*|http://localhost:\d+|http://127\.0\.0\.1:\d+"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    def parse_allowed_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = value.strip("[]\"'")
            if isinstance(value, str):
                value = value.split(",")
            return [item.strip().rstrip("/") for item in value]
        if isinstance(value, list):
            return [item.rstrip("/") if isinstance(item, str) else item for item in value]
        return value

    DATABASE_URL: Optional[str] = None
    DATABASE_URL_SYNC: Optional[str] = None
    DATABASE_USER: str = "talentai"
    DATABASE_PASSWORD: str = "talentai_secret"
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5433
    DATABASE_NAME: str = "talentai_db"

    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_API_KEY: str = ""
    CHROMA_TENANT: str = ""
    CHROMA_DATABASE: str = "talentai_db"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # Frontend Supabase values are included here because the root .env is shared
    # by the Vite build and backend services.
    VITE_SUPABASE_URL: Optional[str] = None
    VITE_SUPABASE_ANON_KEY: Optional[str] = None

    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_MODEL_FALLBACKS: List[str] = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    MATCH_THRESHOLD_AUTO: float = 0.85
    MATCH_THRESHOLD_REVIEW: float = 0.60
    MAX_UPLOAD_SIZE_MB: int = 10
    VITE_API_URL: Optional[str] = None
    ALLOWED_MIME_TYPES: List[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]
    RATE_LIMIT_PER_MINUTE: int = 100

    @model_validator(mode="after")
    def build_database_urls(self):
        if not self.SUPABASE_URL and self.VITE_SUPABASE_URL:
            self.SUPABASE_URL = self.VITE_SUPABASE_URL
        if not self.SUPABASE_JWKS_URL and self.SUPABASE_URL:
            self.SUPABASE_JWKS_URL = f"{self.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"

        if not self.DATABASE_URL or not self.DATABASE_URL.startswith("postgresql"):
            self.DATABASE_URL = (
                f"postgresql+asyncpg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
                f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
            )
        if not self.DATABASE_URL_SYNC or not self.DATABASE_URL_SYNC.startswith("postgresql"):
            self.DATABASE_URL_SYNC = (
                f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
                f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
            )
        return self

    class Config:
        env_file = (Path(__file__).resolve().parents[3] / ".env", ".env")
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
