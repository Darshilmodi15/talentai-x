"""ChromaDB Service - vector store initialization and skill embedding"""
import typing
import logging
from app.core.config import settings

import google.generativeai as genai

logger = logging.getLogger(__name__)

_client = None
_chroma_available = False

COLLECTIONS = {
    "skill_taxonomy": "TalentAI skill taxonomy embeddings",
    "candidate_profiles": "Candidate profile vectors for semantic search",
    "job_descriptions": "Job description vectors",
    "emerging_skills": "Unknown skills found in resumes",
}


def get_chroma_client() -> "typing.Any | None":
    """Return a Chroma client.

    The project runs a local Chroma server in docker-compose, but the previous
    implementation always tried Chroma Cloud and silently disabled vector
    features unless cloud credentials existed. Prefer Cloud only when an API key
    is configured; otherwise connect to the local HTTP server.
    """
    global _client, _chroma_available
    if _client is None and not _chroma_available:
        try:
            import chromadb
            if settings.CHROMA_API_KEY:
                logger.info(
                    f"Connecting to Chroma Cloud "
                    f"(tenant={settings.CHROMA_TENANT}, db={settings.CHROMA_DATABASE}, "
                    f"chromadb_version={chromadb.__version__})"
                )
                _client = chromadb.CloudClient(
                    api_key=settings.CHROMA_API_KEY,
                    tenant=settings.CHROMA_TENANT,
                    database=settings.CHROMA_DATABASE,
                )
                logger.info("✅ Chroma Cloud connected")
            else:
                logger.info(
                    f"Connecting to local Chroma at {settings.CHROMA_HOST}:{settings.CHROMA_PORT} "
                    f"(chromadb_version={chromadb.__version__})"
                )
                _client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                )
                logger.info("✅ Local Chroma connected")
            _chroma_available = True
        except Exception as e:
            logger.error(f"❌ Chroma connection failed: {e}", exc_info=True)
            _client = None
            _chroma_available = False
    return _client




async def init_chroma():
    """Create collections on startup if they don't exist."""
    client = get_chroma_client()
    if client is None:
        logger.warning("Chroma Cloud unavailable — skipping collection init. "
                        "Embedding features will be disabled.")
        return
    for name, description in COLLECTIONS.items():
        try:
            client.get_or_create_collection(
                name=name,
                metadata={"description": description, "hnsw:space": "cosine"},
            )
            logger.info(f"  ✅ Collection '{name}' ready")
        except Exception as e:
            logger.warning(f"Warning: Could not init ChromaDB collection '{name}': {e}")


async def embed_candidate_profile(candidate_id: str, skills: list[dict], summary: str = ""):
    """Embed candidate's skill profile into ChromaDB for fast semantic search."""
    try:
        client = get_chroma_client()
        if client is None:
            logger.warning(f"Chroma unavailable — skipping embed for candidate {candidate_id}")
            return

        collection = client.get_collection("candidate_profiles")

        skill_text = " ".join(s.get("canonical", "") for s in skills)
        combined_text = f"{summary} {skill_text}".strip()[:2000]

        BACKOFFS = [5, 15, 30]
        res = None
        for attempt in range(4):
            try:
                logger.info(f"GEMINI CALL: chroma_service.py | embed_candidate_profile | job_id=unknown | attempt={attempt+1}")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                res = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=combined_text,
                    task_type="retrieval_document",
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                is_quota = any(pattern in error_str for pattern in ["too many requests", "resource exhausted", "quota exceeded", "rate limit", "generaterequestsperday", "429"])
                if is_quota:
                    logger.error("Gemini quota exceeded. Aborting without retry.")
                    raise ValueError("Gemini quota exceeded")

                is_not_found = any(pattern in error_str for pattern in ["404", "not found", "is no longer available"])
                if is_not_found:
                    logger.error("Gemini model not found. Aborting without retry.")
                    raise ValueError("Gemini model not found")
                
                is_network = any(p in error_str for p in ["connection", "timeout", "transient", "network", "ssl"])
                if is_network and attempt < len(BACKOFFS):
                    import asyncio
                    logger.warning(f"Network error. Retrying in {BACKOFFS[attempt]}s...")
                    await asyncio.sleep(BACKOFFS[attempt])
                    continue
                else:
                    logger.error(f"Failed to embed candidate profile: {e}")
                    return

        if not res or 'embedding' not in res:
            return
            
        embedding = res['embedding']

        collection.upsert(
            ids=[candidate_id],
            embeddings=[embedding],  # type: ignore
            metadatas=[{
                "skills": ",".join(s.get("canonical", "") for s in skills[:20]),
                "skill_count": len(skills),
            }],
        )
    except Exception as e:
        logger.warning(f"Warning: Could not embed candidate {candidate_id}: {e}")


async def embed_skill_in_taxonomy(skill_id: str, skill_name: str, category: str, parent: str = ""):
    """Add a skill to the taxonomy ChromaDB collection."""
    try:
        client = get_chroma_client()
        if client is None:
            logger.warning(f"Chroma unavailable — skipping embed for skill '{skill_name}'")
            return

        collection = client.get_collection("skill_taxonomy")

        text = f"{skill_name} {category} {parent}".strip()
        
        BACKOFFS = [5, 15, 30]
        res = None
        for attempt in range(4):
            try:
                logger.info(f"GEMINI CALL: chroma_service.py | add_skills_batch | job_id=unknown | attempt={attempt+1}")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                res = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document",
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                is_quota = any(pattern in error_str for pattern in ["too many requests", "resource exhausted", "quota exceeded", "rate limit", "generaterequestsperday", "429"])
                if is_quota:
                    logger.error("Gemini quota exceeded. Aborting without retry.")
                    raise ValueError("Gemini quota exceeded")

                is_not_found = any(pattern in error_str for pattern in ["404", "not found", "is no longer available"])
                if is_not_found:
                    logger.error("Gemini model not found. Aborting without retry.")
                    raise ValueError("Gemini model not found")
                
                is_network = any(p in error_str for p in ["connection", "timeout", "transient", "network", "ssl"])
                if is_network and attempt < len(BACKOFFS):
                    import asyncio
                    logger.warning(f"Network error. Retrying in {BACKOFFS[attempt]}s...")
                    await asyncio.sleep(BACKOFFS[attempt])
                    continue
                else:
                    logger.error(f"Failed to embed skill: {e}")
                    return
                    
        if not res or 'embedding' not in res:
            return
            
        embedding = res['embedding']

        collection.upsert(
            ids=[skill_id],
            embeddings=[embedding],  # type: ignore
            metadatas=[{
                "canonical_name": skill_name,
                "category": category,
                "parent": parent,
            }],
        )
    except Exception as e:
        logger.warning(f"Warning: Could not embed skill '{skill_name}': {e}")
