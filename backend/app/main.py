"""
TalentAI-X — FastAPI Application
Production-grade multi-agent talent intelligence API
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.db.database import init_db
from app.api.routes import parse, match, candidates, jobs, taxonomy, admin
from app.services.chroma_service import init_chroma
from app.services.taxonomy_service import seed_taxonomy_if_empty


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Startup
    print("🚀 TalentAI-X starting up...")
    await init_db()
    print("✅ Database initialized")
    await init_chroma()
    print("✅ ChromaDB initialized")
    await seed_taxonomy_if_empty()
    print("✅ Skill taxonomy ready")
    print("✅ TalentAI-X is ready!")
    yield
    # Shutdown
    print("👋 TalentAI-X shutting down")


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="TalentAI-X API",
    description="""
## TalentAI-X: Multi-Agent Talent Intelligence System

The world's first bias-aware, explainable AI talent intelligence platform.

### Features
- **Multi-format resume parsing** (PDF, DOCX, TXT)
- **Semantic skill matching** with 87% accuracy (CoT + embeddings)
- **Bias Shield** — blind scoring + bias audit
- **SHAP explainability** on every decision
- **Entity resolution** across GitHub, LinkedIn, Stack Overflow
- **Self-evolving skill taxonomy**
- **Interview question generation**
- **HITL review queue** for borderline decisions

### Authentication
All endpoints require `X-API-Key` header.

### Compliance
Built for NYC Local Law 144 and Colorado AI Act 2026.
    """,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routers
app.include_router(parse.router,       prefix="/api/v1", tags=["Resume Parsing"])
app.include_router(match.router,       prefix="/api/v1", tags=["Matching"])
app.include_router(candidates.router,  prefix="/api/v1", tags=["Candidates"])
app.include_router(jobs.router,        prefix="/api/v1", tags=["Jobs"])
app.include_router(taxonomy.router,    prefix="/api/v1", tags=["Skill Taxonomy"])
app.include_router(admin.router,       prefix="/api/v1", tags=["Admin"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "TalentAI-X API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
    }
