"""
Taxonomy Seeder
Seeds the skill taxonomy DB table on first startup.
If table is empty, inserts all base skills.
"""
from app.db.database import AsyncSessionLocal
from app.db.models.models import SkillTaxonomy

BASE_TAXONOMY = [
    # ── Programming Languages ─────────────────────────────────────
    {"name": "python", "canonical_name": "python", "category": "technical", "parent": "programming languages", "synonyms": ["py", "python3"]},
    {"name": "javascript", "canonical_name": "javascript", "category": "technical", "parent": "programming languages", "synonyms": ["js", "es6", "ecmascript"]},
    {"name": "typescript", "canonical_name": "typescript", "category": "technical", "parent": "programming languages", "synonyms": ["ts"]},
    {"name": "java", "canonical_name": "java", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "golang", "canonical_name": "golang", "category": "technical", "parent": "programming languages", "synonyms": ["go"]},
    {"name": "rust", "canonical_name": "rust", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "c++", "canonical_name": "c++", "category": "technical", "parent": "programming languages", "synonyms": ["cpp"]},
    {"name": "c#", "canonical_name": "c#", "category": "technical", "parent": "programming languages", "synonyms": ["csharp"]},
    {"name": "kotlin", "canonical_name": "kotlin", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "swift", "canonical_name": "swift", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "php", "canonical_name": "php", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "ruby", "canonical_name": "ruby", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "scala", "canonical_name": "scala", "category": "technical", "parent": "programming languages", "synonyms": []},
    {"name": "r", "canonical_name": "r", "category": "technical", "parent": "programming languages", "synonyms": ["r language"]},

    # ── Web Frameworks ────────────────────────────────────────────
    {"name": "react", "canonical_name": "react", "category": "technical", "parent": "web frameworks", "synonyms": ["react.js", "reactjs"]},
    {"name": "next.js", "canonical_name": "next.js", "category": "technical", "parent": "web frameworks", "synonyms": ["nextjs", "next"]},
    {"name": "vue.js", "canonical_name": "vue.js", "category": "technical", "parent": "web frameworks", "synonyms": ["vue", "vuejs"]},
    {"name": "angular", "canonical_name": "angular", "category": "technical", "parent": "web frameworks", "synonyms": ["angularjs"]},
    {"name": "svelte", "canonical_name": "svelte", "category": "technical", "parent": "web frameworks", "synonyms": ["sveltekit"]},
    {"name": "node.js", "canonical_name": "node.js", "category": "technical", "parent": "web frameworks", "synonyms": ["node", "nodejs"]},
    {"name": "express.js", "canonical_name": "express.js", "category": "technical", "parent": "web frameworks", "synonyms": ["express"]},
    {"name": "fastapi", "canonical_name": "fastapi", "category": "technical", "parent": "web frameworks", "synonyms": []},
    {"name": "django", "canonical_name": "django", "category": "technical", "parent": "web frameworks", "synonyms": []},
    {"name": "flask", "canonical_name": "flask", "category": "technical", "parent": "web frameworks", "synonyms": []},
    {"name": "spring boot", "canonical_name": "spring boot", "category": "technical", "parent": "web frameworks", "synonyms": ["spring"]},
    {"name": "laravel", "canonical_name": "laravel", "category": "technical", "parent": "web frameworks", "synonyms": []},

    # ── AI / ML ───────────────────────────────────────────────────
    {"name": "machine learning", "canonical_name": "machine learning", "category": "technical", "parent": "artificial intelligence", "synonyms": ["ml"]},
    {"name": "deep learning", "canonical_name": "deep learning", "category": "technical", "parent": "machine learning", "synonyms": ["dl"]},
    {"name": "natural language processing", "canonical_name": "natural language processing", "category": "technical", "parent": "machine learning", "synonyms": ["nlp"]},
    {"name": "computer vision", "canonical_name": "computer vision", "category": "technical", "parent": "machine learning", "synonyms": ["cv"]},
    {"name": "reinforcement learning", "canonical_name": "reinforcement learning", "category": "technical", "parent": "machine learning", "synonyms": ["rl"]},
    {"name": "large language models", "canonical_name": "large language models", "category": "technical", "parent": "artificial intelligence", "synonyms": ["llm", "llms"]},
    {"name": "generative ai", "canonical_name": "generative ai", "category": "technical", "parent": "artificial intelligence", "synonyms": ["genai", "gpt"]},
    {"name": "retrieval augmented generation", "canonical_name": "retrieval augmented generation", "category": "technical", "parent": "large language models", "synonyms": ["rag"]},
    {"name": "pytorch", "canonical_name": "pytorch", "category": "technical", "parent": "ml frameworks", "synonyms": ["torch"]},
    {"name": "tensorflow", "canonical_name": "tensorflow", "category": "technical", "parent": "ml frameworks", "synonyms": ["tf", "tensorflow2"]},
    {"name": "keras", "canonical_name": "keras", "category": "technical", "parent": "ml frameworks", "synonyms": []},
    {"name": "scikit-learn", "canonical_name": "scikit-learn", "category": "technical", "parent": "ml frameworks", "synonyms": ["sklearn"]},
    {"name": "hugging face", "canonical_name": "hugging face", "category": "technical", "parent": "ml frameworks", "synonyms": ["huggingface", "hf"]},
    {"name": "langchain", "canonical_name": "langchain", "category": "technical", "parent": "llm engineering", "synonyms": []},
    {"name": "langgraph", "canonical_name": "langgraph", "category": "technical", "parent": "llm engineering", "synonyms": []},
    {"name": "transformers architecture", "canonical_name": "transformers architecture", "category": "technical", "parent": "deep learning", "synonyms": ["transformers", "bert"]},

    # ── Cloud ─────────────────────────────────────────────────────
    {"name": "amazon web services", "canonical_name": "amazon web services", "category": "technical", "parent": "cloud platforms", "synonyms": ["aws"]},
    {"name": "google cloud platform", "canonical_name": "google cloud platform", "category": "technical", "parent": "cloud platforms", "synonyms": ["gcp", "google cloud"]},
    {"name": "microsoft azure", "canonical_name": "microsoft azure", "category": "technical", "parent": "cloud platforms", "synonyms": ["azure"]},

    # ── DevOps / Infrastructure ────────────────────────────────────
    {"name": "docker", "canonical_name": "docker", "category": "technical", "parent": "devops", "synonyms": ["dockerfile"]},
    {"name": "kubernetes", "canonical_name": "kubernetes", "category": "technical", "parent": "devops", "synonyms": ["k8s"]},
    {"name": "terraform", "canonical_name": "terraform", "category": "technical", "parent": "infrastructure as code", "synonyms": []},
    {"name": "ci/cd", "canonical_name": "ci/cd", "category": "technical", "parent": "devops", "synonyms": ["cicd"]},
    {"name": "github actions", "canonical_name": "github actions", "category": "technical", "parent": "ci/cd", "synonyms": []},
    {"name": "jenkins", "canonical_name": "jenkins", "category": "technical", "parent": "ci/cd", "synonyms": []},
    {"name": "linux", "canonical_name": "linux", "category": "technical", "parent": "operating systems", "synonyms": ["unix"]},
    {"name": "nginx", "canonical_name": "nginx", "category": "technical", "parent": "web servers", "synonyms": []},
    {"name": "ansible", "canonical_name": "ansible", "category": "technical", "parent": "infrastructure as code", "synonyms": []},

    # ── Databases ─────────────────────────────────────────────────
    {"name": "postgresql", "canonical_name": "postgresql", "category": "technical", "parent": "databases", "synonyms": ["postgres", "pg"]},
    {"name": "mysql", "canonical_name": "mysql", "category": "technical", "parent": "databases", "synonyms": ["mariadb"]},
    {"name": "mongodb", "canonical_name": "mongodb", "category": "technical", "parent": "databases", "synonyms": ["mongo"]},
    {"name": "redis", "canonical_name": "redis", "category": "technical", "parent": "databases", "synonyms": []},
    {"name": "elasticsearch", "canonical_name": "elasticsearch", "category": "technical", "parent": "databases", "synonyms": []},
    {"name": "sql", "canonical_name": "sql", "category": "technical", "parent": "databases", "synonyms": []},
    {"name": "chromadb", "canonical_name": "chromadb", "category": "technical", "parent": "vector databases", "synonyms": ["chroma"]},
    {"name": "pinecone", "canonical_name": "pinecone", "category": "technical", "parent": "vector databases", "synonyms": []},
    {"name": "dynamodb", "canonical_name": "dynamodb", "category": "technical", "parent": "databases", "synonyms": []},

    # ── Data ──────────────────────────────────────────────────────
    {"name": "pandas", "canonical_name": "pandas", "category": "technical", "parent": "data science", "synonyms": ["pd"]},
    {"name": "numpy", "canonical_name": "numpy", "category": "technical", "parent": "data science", "synonyms": ["np"]},
    {"name": "apache spark", "canonical_name": "apache spark", "category": "technical", "parent": "big data", "synonyms": ["spark", "pyspark"]},
    {"name": "tableau", "canonical_name": "tableau", "category": "technical", "parent": "data visualization", "synonyms": []},
    {"name": "power bi", "canonical_name": "power bi", "category": "technical", "parent": "data visualization", "synonyms": []},
    {"name": "airflow", "canonical_name": "airflow", "category": "technical", "parent": "data engineering", "synonyms": ["apache airflow"]},

    # ── Other Technical ───────────────────────────────────────────
    {"name": "git", "canonical_name": "git", "category": "technical", "parent": "version control", "synonyms": []},
    {"name": "graphql", "canonical_name": "graphql", "category": "technical", "parent": "api development", "synonyms": ["gql"]},
    {"name": "rest api", "canonical_name": "rest api", "category": "technical", "parent": "api development", "synonyms": ["restful"]},
    {"name": "microservices", "canonical_name": "microservices", "category": "technical", "parent": "architecture", "synonyms": []},
    {"name": ".net", "canonical_name": ".net", "category": "technical", "parent": "frameworks", "synonyms": ["dotnet"]},
    {"name": "tailwind css", "canonical_name": "tailwind css", "category": "technical", "parent": "css frameworks", "synonyms": ["tailwind"]},

    # ── Soft Skills ───────────────────────────────────────────────
    {"name": "communication", "canonical_name": "communication", "category": "soft", "parent": "interpersonal", "synonyms": ["comm"]},
    {"name": "leadership", "canonical_name": "leadership", "category": "soft", "parent": "management", "synonyms": []},
    {"name": "teamwork", "canonical_name": "teamwork", "category": "soft", "parent": "interpersonal", "synonyms": ["collaboration", "collab"]},
    {"name": "problem solving", "canonical_name": "problem solving", "category": "soft", "parent": "cognitive", "synonyms": []},
    {"name": "project management", "canonical_name": "project management", "category": "soft", "parent": "management", "synonyms": ["pm"]},
    {"name": "agile", "canonical_name": "agile", "category": "soft", "parent": "methodologies", "synonyms": []},
    {"name": "scrum", "canonical_name": "scrum", "category": "soft", "parent": "agile", "synonyms": []},
    {"name": "critical thinking", "canonical_name": "critical thinking", "category": "soft", "parent": "cognitive", "synonyms": []},
    {"name": "time management", "canonical_name": "time management", "category": "soft", "parent": "productivity", "synonyms": []},
    {"name": "mentoring", "canonical_name": "mentoring", "category": "soft", "parent": "leadership", "synonyms": []},
]


async def seed_taxonomy_if_empty():
    """Seed the taxonomy table if it's empty. Called at startup."""
    from sqlalchemy import select, func

    async with AsyncSessionLocal() as db:
        count = await db.scalar(select(func.count()).select_from(SkillTaxonomy))
        if count and count > 0:
            print(f"  Taxonomy already has {count} skills — skipping seed")
            return

        print(f"  Seeding {len(BASE_TAXONOMY)} base skills...")
        for skill_data in BASE_TAXONOMY:
            skill = SkillTaxonomy(
                name=skill_data["name"],
                canonical_name=skill_data["canonical_name"],
                category=skill_data["category"],
                parent=skill_data.get("parent"),
                synonyms=skill_data.get("synonyms", []),
                source="manual",
            )
            db.add(skill)
        await db.commit()
        print(f"  ✅ Seeded {len(BASE_TAXONOMY)} skills into taxonomy")
