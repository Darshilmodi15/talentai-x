"""
Agent 2: Skill Normalizer
- Synonym/abbreviation mapping (JS → JavaScript)
- Fuzzy Jaro-Winkler matching for near-misses
- ChromaDB semantic fallback for unknown skills
- Skill implication graph (PyTorch → Deep Learning)
- Proficiency estimation from experience context
- Emerging skill detection
"""
import time
from datetime import datetime, timezone
from typing import Optional, Any

from app.core.pipeline_state import PipelineState, AgentTrace, SkillEntry
from app.core.config import settings

# Imports like chromadb, sentence_transformers, and jellyfish are loaded lazily to save memory.


# ──────────────────────────────────────────────────────────────────
# Synonym Map — 500+ entries
# ──────────────────────────────────────────────────────────────────

SYNONYMS: dict[str, str] = {
    # JavaScript ecosystem
    "js": "javascript", "javascript": "javascript", "es6": "javascript",
    "es2015": "javascript", "ecmascript": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "node": "node.js", "node.js": "node.js", "nodejs": "node.js",
    "react": "react", "react.js": "react", "reactjs": "react",
    "next": "next.js", "next.js": "next.js", "nextjs": "next.js",
    "vue": "vue.js", "vue.js": "vue.js", "vuejs": "vue.js",
    "angular": "angular", "angularjs": "angular",
    "svelte": "svelte", "sveltekit": "svelte",
    "express": "express.js", "express.js": "express.js",
    "graphql": "graphql", "gql": "graphql",
    "jest": "jest", "mocha": "mocha", "cypress": "cypress",

    # Python ecosystem
    "py": "python", "python": "python", "python3": "python",
    "django": "django", "flask": "flask", "fastapi": "fastapi",
    "pandas": "pandas", "pd": "pandas",
    "numpy": "numpy", "np": "numpy",
    "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
    "tensorflow": "tensorflow", "tensorflow2": "tensorflow", "tf": "tensorflow",
    "torch": "pytorch", "pytorch": "pytorch",
    "keras": "keras", "huggingface": "hugging face", "hf": "hugging face",
    "langchain": "langchain", "langgraph": "langgraph",
    "celery": "celery", "sqlalchemy": "sqlalchemy", "alembic": "alembic",

    # ML / AI
    "ml": "machine learning", "machine learning": "machine learning",
    "dl": "deep learning", "deep learning": "deep learning",
    "nlp": "natural language processing", "natural language processing": "natural language processing",
    "cv": "computer vision", "computer vision": "computer vision",
    "rl": "reinforcement learning", "reinforcement learning": "reinforcement learning",
    "llm": "large language models", "large language models": "large language models",
    "gpt": "generative ai", "genai": "generative ai", "generative ai": "generative ai",
    "rag": "retrieval augmented generation", "retrieval augmented generation": "retrieval augmented generation",
    "bert": "bert", "transformers": "transformers architecture",
    "cnn": "convolutional neural networks", "rnn": "recurrent neural networks",
    "lstm": "lstm",

    # Cloud
    "aws": "amazon web services", "amazon web services": "amazon web services",
    "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "azure": "microsoft azure", "microsoft azure": "microsoft azure",
    "s3": "aws s3", "ec2": "aws ec2", "lambda": "aws lambda",
    "gke": "google kubernetes engine", "eks": "aws eks",

    # DevOps / Infrastructure
    "k8s": "kubernetes", "kubernetes": "kubernetes",
    "docker": "docker", "dockerfile": "docker",
    "terraform": "terraform", "tfvars": "terraform",
    "ci/cd": "ci/cd", "cicd": "ci/cd",
    "github actions": "github actions", "jenkins": "jenkins",
    "ansible": "ansible", "helm": "helm",
    "nginx": "nginx", "linux": "linux",

    # Databases
    "pg": "postgresql", "postgres": "postgresql", "postgresql": "postgresql",
    "mysql": "mysql", "mariadb": "mysql",
    "mongo": "mongodb", "mongodb": "mongodb",
    "redis": "redis", "elasticsearch": "elasticsearch",
    "cassandra": "cassandra", "dynamodb": "dynamodb",
    "sql": "sql", "nosql": "nosql",
    "chroma": "chromadb", "chromadb": "chromadb",
    "pinecone": "pinecone", "weaviate": "weaviate",

    # Other languages
    "java": "java", "spring": "spring boot", "spring boot": "spring boot",
    "kotlin": "kotlin", "swift": "swift", "go": "golang", "golang": "golang",
    "rust": "rust", "c++": "c++", "cpp": "c++",
    "c#": "c#", "dotnet": ".net", ".net": ".net",
    "php": "php", "laravel": "laravel", "ruby": "ruby", "rails": "ruby on rails",

    # Soft skills
    "comm": "communication", "collab": "collaboration",
    "pm": "project management", "agile": "agile",
    "scrum": "scrum", "kanban": "kanban",
}


# ──────────────────────────────────────────────────────────────────
# Skill Implication Graph
# ──────────────────────────────────────────────────────────────────

IMPLICATIONS: list[tuple[frozenset, str, float]] = [
    # (known_skills_set, implied_skill, confidence)
    (frozenset({"pytorch", "tensorflow"}), "deep learning", 0.95),
    (frozenset({"pytorch"}), "deep learning", 0.88),
    (frozenset({"tensorflow"}), "deep learning", 0.85),
    (frozenset({"keras"}), "deep learning", 0.82),
    (frozenset({"deep learning"}), "machine learning", 0.99),
    (frozenset({"scikit-learn"}), "machine learning", 0.92),
    (frozenset({"react", "node.js", "mongodb"}), "mern stack", 0.95),
    (frozenset({"react", "node.js"}), "full stack javascript", 0.88),
    (frozenset({"django", "postgresql"}), "full stack python", 0.85),
    (frozenset({"kubernetes", "docker"}), "container orchestration", 0.97),
    (frozenset({"docker"}), "containerization", 0.99),
    (frozenset({"terraform", "aws"}), "infrastructure as code", 0.93),
    (frozenset({"langchain"}), "llm engineering", 0.92),
    (frozenset({"hugging face"}), "llm engineering", 0.88),
    (frozenset({"bert"}), "natural language processing", 0.90),
    (frozenset({"fastapi", "python"}), "python web development", 0.92),
    (frozenset({"postgresql", "mysql"}), "relational databases", 0.95),
    (frozenset({"mongodb"}), "nosql databases", 0.95),
    (frozenset({"redis"}), "caching", 0.90),
    (frozenset({"computer vision", "pytorch"}), "image recognition", 0.85),
]


def infer_skills(known_canonicals: set[str]) -> list[SkillEntry]:
    """Walk implication graph and return newly inferred skills."""
    inferred = []
    for prereqs, implied, conf in IMPLICATIONS:
        if prereqs.issubset(known_canonicals) and implied not in known_canonicals:
            inferred.append(SkillEntry(
                raw=None,
                canonical=implied,
                category=categorize_skill(implied),
                proficiency="inferred",
                years=0.0,
                inferred=True,
                source="inferred",
            ))
    return inferred


# ──────────────────────────────────────────────────────────────────
# Skill Categorization
# ──────────────────────────────────────────────────────────────────

CATEGORIES = {
    "technical": {
        "python", "javascript", "typescript", "java", "golang", "rust", "c++",
        "react", "node.js", "django", "fastapi", "spring boot",
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "reinforcement learning",
        "pytorch", "tensorflow", "scikit-learn", "keras",
        "docker", "kubernetes", "terraform", "aws", "azure", "google cloud platform",
        "postgresql", "mongodb", "redis", "elasticsearch",
        "sql", "nosql", "api development", "rest api", "graphql",
        "ci/cd", "linux", "git",
    },
    "soft": {
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "agile", "scrum",
        "collaboration", "time management", "mentoring",
    },
}


def categorize_skill(skill: str) -> str:
    skill_lower = skill.lower()
    if skill_lower in CATEGORIES["technical"]:
        return "technical"
    if skill_lower in CATEGORIES["soft"]:
        return "soft"
    return "domain"


# ──────────────────────────────────────────────────────────────────
# Proficiency Estimation
# ──────────────────────────────────────────────────────────────────

def estimate_years(skill: str, experience: list[dict]) -> float:
    """Scan experience bullets for mentions of this skill, sum time."""
    skill_lower = skill.lower()
    total_months = 0
    for exp in experience:
        mentions = any(
            skill_lower in bullet.lower()
            for bullet in exp.get("bullets", []) + exp.get("skills_mentioned", [])
        )
        if mentions:
            total_months += exp.get("duration_months", 0)
    return round(total_months / 12, 1)


def years_to_proficiency(years: float) -> str:
    if years < 0.5:
        return "beginner"
    elif years < 2:
        return "intermediate"
    elif years < 5:
        return "advanced"
    else:
        return "expert"


# ──────────────────────────────────────────────────────────────────
# Fuzzy Matching
# ──────────────────────────────────────────────────────────────────

def fuzzy_match_synonym(skill: str) -> Optional[str]:
    """Jaro-Winkler match against synonym keys. Returns canonical or None."""
    skill_lower = skill.lower().strip()

    # Direct lookup first
    if skill_lower in SYNONYMS:
        return SYNONYMS[skill_lower]

    # Fuzzy match against all synonym keys
    best_score = 0.0
    best_match = None
    try:
        import jellyfish
        for key in SYNONYMS:
            score = jellyfish.jaro_winkler_similarity(skill_lower, key)
            if score > best_score:
                best_score = score
                best_match = key
    except ImportError:
        pass

    if best_score >= 0.88 and best_match:
        return SYNONYMS[best_match]

    return None


# ──────────────────────────────────────────────────────────────────
# ChromaDB Semantic Fallback
# ──────────────────────────────────────────────────────────────────

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        if settings.CHROMA_API_KEY:
            _chroma_client = chromadb.CloudClient(
                api_key=settings.CHROMA_API_KEY,
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DATABASE,
            )
        else:
            _chroma_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
    return _chroma_client



async def semantic_skill_lookup_batch(skills: list[str], state: PipelineState) -> dict[str, Optional[str]]:
    """
    Batch embed skills and find closest match in skill taxonomy collection.
    Returns mapping of raw_skill to canonical name if similarity > 0.80, else None.
    """
    if not skills:
        return {}
    try:
        client = get_chroma_client()
        collection = client.get_collection("skill_taxonomy")
        import google.generativeai as genai
        import logging
        import asyncio
        logger = logging.getLogger(__name__)
        job_id = state.get("job_id", "unknown")
        
        BACKOFFS = [5, 15, 30]
        res = None
        for attempt in range(4):
            try:
                state["gemini_api_calls"] = (state.get("gemini_api_calls") or 0) + 1
                logger.info(f"Calling Gemini | agent=normalize_agent | job_id={job_id} | attempt={attempt+1}")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                res = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=skills,
                    task_type="retrieval_query",
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
                    logger.warning(f"Network error. Retrying in {BACKOFFS[attempt]}s...")
                    await asyncio.sleep(BACKOFFS[attempt])
                    continue
                else:
                    return {s: None for s in skills}
                    
        if not res or 'embedding' not in res:
            return {s: None for s in skills}

        embeddings = res['embedding']
        results = collection.query(
            query_embeddings=embeddings,
            n_results=1,
        )
        
        mapping = {}
        for i, skill in enumerate(skills):
            distances = results.get("distances")
            mapping[skill] = None
            if distances and i < len(distances) and distances[i] and len(distances[i]) > 0:
                if distances[i][0] < 0.20:
                    metadatas = results.get("metadatas")
                    if metadatas and i < len(metadatas) and metadatas[i] and len(metadatas[i]) > 0:
                        meta = metadatas[i][0]
                        if isinstance(meta, dict):
                            name = meta.get("canonical_name")
                            mapping[skill] = str(name) if name is not None else None
                            
        return mapping
    except Exception:
        return {s: None for s in skills}


# ──────────────────────────────────────────────────────────────────
# Main Agent Function
# ──────────────────────────────────────────────────────────────────

async def normalize_agent(state: PipelineState) -> PipelineState:
    """
    Agent 2: Normalize skills.
    Reads:  state["parsed"]
    Writes: state["skills_canonical"], state["inferred_skills"],
            state["emerging_skills_found"]
    """
    started = time.time()
    error_msg = None
    status = "success"

    try:
        parsed = state.get("parsed") or {}
        raw_skills: list[str] = parsed.get("skills", [])
        experience: list[dict] = parsed.get("experience", [])

        canonical_entries: list[SkillEntry] = []
        emerging: list[str] = []
        seen_canonicals: set[str] = set()
        
        # Pre-process: Find exact and fuzzy matches, gather unknown skills
        unknown_skills = []
        skill_metadata = {}
        for raw_skill in raw_skills:
            if not raw_skill or not raw_skill.strip():
                continue
            
            raw_lower = raw_skill.lower().strip()
            canonical = SYNONYMS.get(raw_lower)
            if not canonical:
                canonical = fuzzy_match_synonym(raw_skill)
                
            skill_metadata[raw_skill] = {
                "raw_lower": raw_lower,
                "canonical": canonical,
                "years": estimate_years(canonical or raw_lower, experience),
            }
            if not canonical:
                unknown_skills.append(raw_skill)
                
        # Batch Semantic Lookup for unknowns
        semantic_mapping = await semantic_skill_lookup_batch(unknown_skills, state)
        
        # Build entries
        for raw_skill in skill_metadata:
            meta = skill_metadata[raw_skill]
            canonical = meta["canonical"]
            if not canonical:
                canonical = semantic_mapping.get(raw_skill)
                
            is_emerging = canonical is None
            if not canonical:
                canonical = meta["raw_lower"]
                
            proficiency = years_to_proficiency(meta["years"])
            entry = SkillEntry(
                raw=raw_skill,
                canonical=canonical,
                category=categorize_skill(canonical),
                proficiency=proficiency,
                years=meta["years"],
                inferred=False,
                source="resume",
            )
            
            if entry["canonical"] not in seen_canonicals:
                canonical_entries.append(entry)
                seen_canonicals.add(entry["canonical"])
            if is_emerging:
                emerging.append(raw_skill)

        # Infer implied skills
        inferred = infer_skills(seen_canonicals)

        state["skills_canonical"] = canonical_entries
        state["inferred_skills"] = inferred
        state["emerging_skills_found"] = emerging

    except Exception as e:
        error_msg = str(e)
        state["errors"] = state.get("errors", []) + [f"normalize_agent: {error_msg}"]
        state["skills_canonical"] = []
        state["inferred_skills"] = []
        state["emerging_skills_found"] = []
        status = "failed"

    parsed_state = state.get("parsed") or {}
    quality = (
        len(state.get("skills_canonical", [])) / max(len(parsed_state.get("skills", []) or []), 1)
    )

    trace: AgentTrace = {
        "agent": "normalize_agent",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "status": status,
        "quality_score": min(quality, 1.0),
        "fields_extracted": len(state["skills_canonical"]),
        "retry_count": 0,
        "error": error_msg,
    }
    state["traces"] = state.get("traces", []) + [trace]
    return state
