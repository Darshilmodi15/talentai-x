# TalentAI-X — Multi-Agent Talent Intelligence System

> A multi-agent talent intelligence system with explainable matching and human review.  
> Built for Tic Tech Toe '26 · Domain: Multi-Agent AI & Talent Intelligence

---

## Overview and current status

TalentAI-X parses resumes, normalizes skills, matches candidates to jobs, and resolves candidate identity across platforms using a LangGraph pipeline. It was built for Tic Tech Toe '26 (Multi-Agent AI & Talent Intelligence).

**Implemented**
- FastAPI service with a LangGraph orchestrator over four agents: parse, normalize, match, and entity resolution.
- Skill normalization using a synonym map, fuzzy matching (rapidfuzz) and ChromaDB embeddings.
- Gemini-assisted semantic matching with a per-decision, SHAP-style feature attribution (a custom attribution routine, not the `shap` library).
- Batch resume processing with Celery and Redis, PostgreSQL storage, and Supabase-authenticated routes with server-side role checks.
- Human-in-the-loop review triggers for borderline matches, and generated interview questions per candidate.

**Partially implemented**
- Bias Shield: demographic and prestige signals (institution, company names) are stripped to build a blind profile. The blind-versus-full score comparison is currently a placeholder that does not re-run the matcher, so the reported bias delta should not be treated as a measured result.

**Not implemented / not evaluated**
- Agent 5 (skill intelligence, nightly job) shown in the diagram below is a planned component and is not in this repository.
- No accuracy benchmark is included in this repository. Earlier claims about match accuracy and regulatory compliance had no supporting evaluation here and have been removed.

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
# For real account login, also set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.
# The browser must never receive SUPABASE_SERVICE_ROLE_KEY or SUPABASE_JWT_SECRET.
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

### Authentication and demo access

The frontend supports Supabase Auth for individual and recruiter accounts. Add the
Supabase project URL and anon key to `.env`; the Vite configuration loads the root
environment file for local development and Docker Compose passes these values to
the frontend container.

There are no public demo buttons. To create private team demo accounts, temporarily
add these values to the root `.env`:

```bash
DEMO_INDIVIDUAL_EMAIL=...
DEMO_INDIVIDUAL_PASSWORD=...
DEMO_RECRUITER_EMAIL=...
DEMO_RECRUITER_PASSWORD=...
SUPABASE_SERVICE_ROLE_KEY=...
```

Then run `python scripts/create_demo_users.py`. The script creates or updates
confirmed Supabase users and assigns their server-side roles. Remove the service
role key and demo passwords after running; never add them to Vercel or any
`VITE_*` variable.

### Deployment environment variables

| Environment | Required variables |
|---|---|
| Local root `.env` | `GEMINI_API_KEY`, database/Redis values, `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` |
| Render backend | `GEMINI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `SUPABASE_URL`, `SUPABASE_JWT_SECRET` (legacy HS256 projects), `SUPABASE_JWKS_URL` (current signing-key projects), `SUPABASE_JWT_AUDIENCE`, `SECRET_KEY`, and allowed-origin values |
| Vercel frontend | `VITE_API_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` only |

`SUPABASE_JWT_SECRET` and `SUPABASE_SERVICE_ROLE_KEY` are backend/setup secrets.
They must not be exposed to the browser.

### Gemini model selection

The backend prefers `GEMINI_MODEL` when that model supports `generateContent` for
the configured API key. It checks the key's available models and falls back to
`GEMINI_MODEL_FALLBACKS` when necessary. The current embedding default is
`gemini-embedding-001`; update the values in `.env` if your Google project exposes
a different model set.

---

## API Reference

Protected endpoints require the signed-in user's Supabase access token. The
backend validates the JWT and loads the user's role from `user_profiles`;
frontend role metadata is never used for authorization.

### Resume Parsing
```bash
# Upload single resume
curl -X POST http://localhost:8000/api/v1/parse \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -F "file=@resume.pdf"
# Returns: { "job_id": "uuid", "status": "queued" }

# Poll status
curl http://localhost:8000/api/v1/jobs/{job_id} \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Get agent trace
curl http://localhost:8000/api/v1/jobs/{job_id}/trace \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Batch upload
curl -X POST http://localhost:8000/api/v1/parse/batch \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -F "files=@resume1.pdf" -F "files=@resume2.docx"
```

### Candidate Intelligence
```bash
# List candidates
curl http://localhost:8000/api/v1/candidates \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Full profile
curl http://localhost:8000/api/v1/candidates/{id} \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Normalized skill profile
curl http://localhost:8000/api/v1/candidates/{id}/skills \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

### Matching
```bash
# Match one candidate to JD
curl -X POST http://localhost:8000/api/v1/match \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "uuid",
    "job_description": "Senior ML Engineer, Python, PyTorch required..."
  }'

# Batch rank
curl -X POST http://localhost:8000/api/v1/match/batch \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_ids": ["uuid1", "uuid2"],
    "job_description": "...",
    "top_n": 10
  }'

# Bias audit (NYC Local Law 144 compliance)
curl -X POST "http://localhost:8000/api/v1/match/bias-audit?job_description=..." \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
```

### Skill Taxonomy
```bash
# Browse
curl "http://localhost:8000/api/v1/skills/taxonomy?category=technical" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Search
curl "http://localhost:8000/api/v1/skills/taxonomy/search?q=pytorch" \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"

# Emerging skills (Agent 5 output)
curl http://localhost:8000/api/v1/skills/taxonomy/emerging \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"
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
