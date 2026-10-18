# TalentAI-X — Multi-Agent Talent Intelligence System

> The world's first **bias-aware, explainable** multi-agent AI platform for talent intelligence.  
> Built for Tic Tech Toe '26 · Domain: Multi-Agent AI & Talent Intelligence

---

## What makes this different from every other ATS

| Feature | Traditional ATS | TalentAI-X |
|---|---|---|
| Matching method | Keyword matching (40–55% accuracy) | CoT + semantic embeddings (87% accuracy) |
| Bias detection | None | Dual blind/sighted scoring with audit API |
| Explainability | Black box score | SHAP attribution on every decision |
| Skill normalization | Exact match | Synonym map + fuzzy + semantic + implication graph |
| Cross-platform identity | None | Anchor-first entity resolution (GitHub, LinkedIn, SO) |
| Skill taxonomy | Static | Self-evolving via Agent 5 |
| Human review | Never triggered | HITL queue for borderline + biased decisions |
| Interview prep | None | 6 tailored interview questions per candidate |
| Legal compliance | None | NYC Local Law 144 · Colorado AI Act 2026 · GDPR |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (Port 8000)                  │
│  /parse  /match  /candidates  /skills/taxonomy  /admin      │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼─────────┐
              │  LangGraph       │
              │  Orchestrator    │
              └────────┬─────────┘
        ┌──────────────┼──────────────┐──────────────┐
        ▼              ▼              ▼               ▼
   ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐
   │ Agent 1 │  │  Agent 2  │  │ Agent 3  │  │   Agent 4    │
   │  Parse  │→ │ Normalize │→ │  Match   │  │   Resolve    │
   │         │  │           │  │ + Bias   │  │   Identity   │
   └─────────┘  └───────────┘  └──────────┘  └──────────────┘
                                      │
                              ┌───────▼──────────┐
                              │    Agent 5       │
                              │  Skill Intel     │
                              │ (nightly cron)   │
                              └──────────────────┘

Storage:
  PostgreSQL  — structured data (candidates, jobs, matches)
  ChromaDB    — skill + profile embeddings (semantic search)
  Redis       — job queue, caching, rate limiting
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- Docker + Docker Compose
- Gemini API key

### 1. Clone and configure
```bash
git clone <repo>
cd talentai-x
cp .env.example .env
# Edit .env — set GEMINI_API_KEY
```

### 2. One-command setup
```bash
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

### 3. Start everything with Docker
```bash
docker compose up
```

### 4. Or run locally (3 terminals)
```bash
# Terminal 1 — Backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Celery worker (for batch processing)
cd backend && source venv/bin/activate
celery -A app.worker.celery_app worker --loglevel=info

# Terminal 3 — Frontend
cd frontend && npm run dev
```

### URLs
| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |

---

## Authentication & Role-Based Access Control (RBAC)

TalentAI-X protects all routes using **Supabase JWT validation** with server-side role management in `user_profiles` and strict ownership scoping to eliminate Insecure Direct Object References (IDOR):

- **Authentication Header**: `Authorization: Bearer <supabase_jwt_access_token>` (or `X-API-Key` for provisioned service keys).
- **Roles & Permissions**:
  - `individual`: Can upload resumes, view their own parsed candidates, search saved jobs, and run individual matches.
  - `recruiter`: Can perform batch candidate ranking, shortlists, and NYC Local Law 144 algorithmic bias audits.
  - `admin`: Full system access, HITL review queue resolution, skill taxonomy approvals, system statistics, and API key provisioning.
- **Ownership Scoping**: Candidates, saved jobs, parse jobs, and matches are strictly isolated by `user_id`. Non-admin accounts can only access records they own.

### Resume Parsing
```bash
# Upload single resume (authenticated)
curl -X POST http://localhost:8000/api/v1/parse \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -F "file=@resume.pdf"
# Returns: { "job_id": "uuid", "status": "queued" }

# Poll status
curl http://localhost:8000/api/v1/jobs/{job_id} \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Get agent trace
curl http://localhost:8000/api/v1/jobs/{job_id}/trace \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Batch upload
curl -X POST http://localhost:8000/api/v1/parse/batch \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -F "files=@resume1.pdf" -F "files=@resume2.docx"
```

### Candidate Intelligence
```bash
# List candidates (scoped to user)
curl http://localhost:8000/api/v1/candidates \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Full profile (with ownership validation)
curl http://localhost:8000/api/v1/candidates/{id} \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Normalized skill profile
curl http://localhost:8000/api/v1/candidates/{id}/skills \
  -H "Authorization: Bearer $SUPABASE_JWT"
```

### Matching
```bash
# Match candidate to JD
curl -X POST http://localhost:8000/api/v1/match \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "uuid",
    "job_description": "Senior ML Engineer, Python, PyTorch required..."
  }'

# Batch rank (recruiter or admin role required)
curl -X POST http://localhost:8000/api/v1/match/batch \
  -H "Authorization: Bearer $SUPABASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_ids": ["uuid1", "uuid2"],
    "job_description": "...",
    "top_n": 10
  }'

# Bias audit (recruiter or admin role required - NYC Local Law 144 compliance)
curl -X POST "http://localhost:8000/api/v1/match/bias-audit?job_description=..." \
  -H "Authorization: Bearer $SUPABASE_JWT"
```

### Skill Taxonomy
```bash
# Browse taxonomy
curl "http://localhost:8000/api/v1/skills/taxonomy?category=technical" \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Search skills
curl "http://localhost:8000/api/v1/skills/taxonomy/search?q=pytorch" \
  -H "Authorization: Bearer $SUPABASE_JWT"

# Emerging skills (pending proposals)
curl http://localhost:8000/api/v1/skills/taxonomy/emerging \
  -H "Authorization: Bearer $SUPABASE_JWT"
```

---

## The 5 Agents

### Agent 1 — Resume Parser
- Layout detection: single-column, two-column, table, image/OCR
- Parallelized extraction: 4 specialized LLM prompts run concurrently
- Multilingual: detect and route non-English resumes
- AI content detection via burstiness analysis

### Agent 2 — Skill Normalizer
- 500+ synonym mappings (JS→JavaScript, K8s→Kubernetes)
- Jaro-Winkler fuzzy matching for typos/abbreviations
- ChromaDB semantic fallback for near-miss matching
- Implication graph: PyTorch+TensorFlow→Deep Learning
- Proficiency estimation from experience context

### Agent 3 — Semantic Matcher
- Chain-of-Thought reasoning (87% accuracy, MDPI 2025)
- Bias Shield: strips demographic signals, runs blind score
- SHAP attribution on every score (which features drove it)
- Gap analysis with upskilling paths
- 6 tailored interview questions per candidate
- HITL trigger evaluation

### Agent 4 — Entity Resolver
- Anchor-first: explicit links = 100% confidence
- Leet-speak decoder (ni3av→Nirav)
- GitHub API integration (real commit/language data)
- Generic Open Graph scraper for unknown platforms
- NetworkX identity graph for multi-hop resolution

### Agent 5 — Skill Intelligence (nightly)
- Clusters unknown skills by embedding similarity
- Auto-proposes canonical names for high-confidence clusters
- Queues low-confidence proposals for human review
- Self-evolving taxonomy that improves with every batch

---

## Evaluation Metrics

| Metric | Target | Method |
|---|---|---|
| Parse F1 (name/email) | > 0.95 | Ground truth comparison |
| Parse F1 (skills) | > 0.85 | Manual annotation on 50 resumes |
| Match accuracy | > 0.85 | NDCG@10 vs expert rankings |
| End-to-end latency | < 10s | `pytest tests/integration/test_latency.py` |
| Bias score | < 0.05 | `/api/v1/match/bias-audit` |

---

## Project Structure

```
talentai-x/
├── backend/
│   ├── app/
│   │   ├── agents/          # 4 agent implementations + orchestrator
│   │   ├── api/routes/      # FastAPI route handlers
│   │   ├── api/middleware/  # Auth, rate limiting
│   │   ├── core/            # Config, pipeline state contract
│   │   ├── db/              # SQLAlchemy models, migrations
│   │   └── services/        # Candidate, ChromaDB, taxonomy services
│   ├── tests/               # Unit + integration tests
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/pages/  # Dashboard, Upload, Match, etc.
│       ├── components/layouts/ # Sidebar navigation
│       └── api/client.ts      # Typed API client
├── data/
│   ├── generate_resumes.py    # Synthetic resume generator
│   └── synthetic_resumes/     # Sample data for testing
├── scripts/setup.sh           # One-command setup
├── docker-compose.yml
└── .env.example
```

---

## Team

| Member | Role | Owns |
|---|---|---|
| Darshil Modi | Team Lead / Orchestration | LangGraph pipeline, FastAPI, Docker |
| Kavan Modi | Frontend | React UI, all pages, API client |
| Harpal Prajapati | Parse Agent | PDF/DOCX extraction, layout detection |
| Het Chaudhari | Normalize + Match Agent | Taxonomy, skill graph, CoT matching |
| Rudra Patel | Bias + Explainability Agent | Bias detection, explanation |

---

## Research Citations

- UW 2025: LLMs show 85% racial bias in resume ranking (3M+ comparisons)
- MDPI 2025: CoT + embeddings achieves 87% matching accuracy zero-shot
- IJRTI 2025: SHAP/LIME explainability in ethical AI hiring systems
- Brookings 2025: Intersectional bias in AI resume screening
- arXiv 2025: Parallelized task decomposition for resume parsing
- Stanford 2025: Age + gender bias in AI resume tools

---

*Built with Gemini (Google) · LangGraph · FastAPI · React · ChromaDB*
