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
    global _client, _chroma_available
    if _client is None and not _chroma_available:
        try:
            import chromadb
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
            _chroma_available = True
            logger.info("✅ Chroma Cloud connected")
        except Exception as e:
            logger.error(f"❌ Chroma Cloud connection failed: {e}", exc_info=True)
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

        genai.configure(api_key=settings.GEMINI_API_KEY)
        res = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=combined_text,
            task_type="retrieval_document",
        )
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
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        res = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        embedding = res['embedding']

        collection.upsert(
            ids=[skill_id],
            embeddings=[embedding],  # type: ignore
            metadatas={
                "canonical_name": skill_name,
                "category": category,
                "parent": parent,
            },
        )
    except Exception as e:
        logger.warning(f"Warning: Could not embed skill '{skill_name}': {e}")
