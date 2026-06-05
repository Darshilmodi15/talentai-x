from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TalentAI-X"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # API Security
    SECRET_KEY: str = "change_this_in_production"
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://talentai:talentai_secret@localhost:5432/talentai_db"
    DATABASE_URL_SYNC: str = "postgresql://talentai:talentai_secret@localhost:5432/talentai_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # Anthropic
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Matching thresholds
    MATCH_THRESHOLD_AUTO: float = 0.85
    MATCH_THRESHOLD_REVIEW: float = 0.60

    # File upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_MIME_TYPES: List[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
