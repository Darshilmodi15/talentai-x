from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=getattr(settings, "SQL_ECHO", False),
    pool_size=5,
    max_overflow=2,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

SCHEMA_UPGRADE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        id UUID PRIMARY KEY,
        email VARCHAR(255),
        role VARCHAR(20) NOT NULL DEFAULT 'individual',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON user_profiles(role);
    """,
    """
    ALTER TABLE parse_jobs
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES user_profiles(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_parse_jobs_user_id ON parse_jobs(user_id);
    """,
    """
    ALTER TABLE candidates
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES user_profiles(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_candidates_user_id ON candidates(user_id);
    """,
    """
    ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES user_profiles(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
    """,
    """
    ALTER TABLE match_results
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES user_profiles(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_match_results_user_id ON match_results(user_id);
    """,
    """
    ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES user_profiles(id) ON DELETE SET NULL;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
    """,
]


async def init_db():
    """Create all tables on startup and apply incremental schema upgrades."""
    # Ensure all models are registered in Base.metadata
    from app.db.models import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in SCHEMA_UPGRADE_STATEMENTS:
            try:
                await conn.execute(text(stmt.strip()))
            except Exception as e:
                logger.debug(f"Schema upgrade notice: {e}")
