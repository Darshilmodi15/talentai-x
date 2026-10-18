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


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.url.get_backend_name() == "postgresql":
            await conn.exec_driver_sql(
                "ALTER TABLE user_profiles DROP CONSTRAINT IF EXISTS fk_user_profiles_auth_users"
            )
            await conn.exec_driver_sql(
                "ALTER TABLE parse_jobs ADD COLUMN IF NOT EXISTS user_id UUID"
            )
            await conn.exec_driver_sql(
                "ALTER TABLE candidates ADD COLUMN IF NOT EXISTS user_id UUID"
            )
            await conn.exec_driver_sql(
                "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS user_id UUID"
            )
            await conn.exec_driver_sql(
                "ALTER TABLE match_results ADD COLUMN IF NOT EXISTS user_id UUID"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_parse_jobs_user_id ON parse_jobs(user_id)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_candidates_user_id ON candidates(user_id)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_jobs_user_id ON jobs(user_id)"
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_match_results_user_id ON match_results(user_id)"
            )
