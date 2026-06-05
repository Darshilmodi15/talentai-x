"""ChromaDB Service - vector store initialization and skill embedding"""
import chromadb
from app.core.config import settings

_client = None
_model = None

COLLECTIONS = {
    "skill_taxonomy": "TalentAI skill taxonomy embeddings",
    "candidate_profiles": "Candidate profile vectors for semantic search",
    "job_descriptions": "Job description vectors",
    "emerging_skills": "Unknown skills found in resumes",
}


def get_chroma_client() -> chromadb.CloudClient:
    global _client
    if _client is None:
        _client = chromadb.CloudClient(
            api_key=settings.CHROMA_API_KEY,
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
        )
    return _client


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


async def init_chroma():
    """Create collections on startup if they don't exist."""
    client = get_chroma_client()
    for name, description in COLLECTIONS.items():
        try:
            client.get_or_create_collection(
                name=name,
                metadata={"description": description, "hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"Warning: Could not init ChromaDB collection '{name}': {e}")


async def embed_candidate_profile(candidate_id: str, skills: list[dict], summary: str = ""):
    """Embed candidate's skill profile into ChromaDB for fast semantic search."""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        collection = client.get_collection("candidate_profiles")

        skill_text = " ".join(s.get("canonical", "") for s in skills)
        combined_text = f"{summary} {skill_text}".strip()[:2000]

        embedding = model.encode(combined_text).tolist()

        collection.upsert(
            ids=[candidate_id],
            embeddings=[embedding],
            metadatas=[{
                "skills": ",".join(s.get("canonical", "") for s in skills[:20]),
                "skill_count": len(skills),
            }],
        )
    except Exception as e:
        print(f"Warning: Could not embed candidate {candidate_id}: {e}")


async def embed_skill_in_taxonomy(skill_id: str, skill_name: str, category: str, parent: str = ""):
    """Add a skill to the taxonomy ChromaDB collection."""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        collection = client.get_collection("skill_taxonomy")

        text = f"{skill_name} {category} {parent}".strip()
        embedding = model.encode(text).tolist()

        collection.upsert(
            ids=[skill_id],
            embeddings=[embedding],
            metadatas={
                "canonical_name": skill_name,
                "category": category,
                "parent": parent,
            },
        )
    except Exception as e:
        print(f"Warning: Could not embed skill '{skill_name}': {e}")
