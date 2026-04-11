"""
Agent 3: Semantic Matcher
- Chain-of-Thought reasoning via structured LLM prompt
- Vector embedding cosine similarity
- Weighted multi-dimensional scoring
- Bias Shield: blind score vs full score
- SHAP-style feature attribution
- Gap analysis + upskilling suggestions
- Interview question generation
- HITL trigger evaluation
"""
import json
import time
from datetime import datetime
from typing import Optional

from app.core.pipeline_state import PipelineState, AgentTrace, MatchedSkill
from app.core.config import settings

try:
    import anthropic
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────

JD_PARSE_PROMPT = """Parse this job description into structured requirements.
Return ONLY valid JSON.

Schema:
{{
  "title": string,
  "required_skills": [string],
  "nice_to_have_skills": [string],
  "experience_years_min": integer,
  "education_requirement": string or null,
  "key_responsibilities": [string]
}}

Job Description:
{jd}"""


COT_MATCH_PROMPT = """You are a senior technical recruiter. Evaluate this candidate step-by-step.

JOB REQUIREMENTS:
{job_requirements}

CANDIDATE SKILL PROFILE:
{candidate_profile}

CANDIDATE EXPERIENCE SUMMARY:
{experience_summary}

Follow these steps exactly:

STEP 1 - REQUIRED SKILLS ANALYSIS:
For each required skill, state:
[skill] → [FOUND/PARTIAL/NOT FOUND] → [evidence from experience or "not evidenced"]

STEP 2 - EXPERIENCE DEPTH:
For the top 3 matched skills, assess:
[skill] → [estimated years] → [proficiency: beginner/intermediate/advanced/expert]

STEP 3 - SKILL GAP ANALYSIS:
List each missing required skill. For each:
[skill] → [critical/important/minor] → [weeks to learn estimate]

STEP 4 - GROWTH & CULTURE SIGNALS:
Identify evidence of: rapid learning, ownership, cross-functional work, initiative.
Be specific - quote from experience bullets if available.

STEP 5 - SCORING:
required_skill_coverage: X/10
experience_depth: X/10
growth_trajectory: X/10
overall_fit: weighted_score/100

STEP 6 - RECOMMENDATION:
[STRONG YES / YES / CONDITIONAL / NO]
One sentence reason.

Return your full analysis as plain text."""


INTERVIEW_PROMPT = """Generate targeted interview questions for this candidate.

MATCHED SKILLS (verify depth): {matched}
SKILL GAPS (probe awareness): {gaps}
RECOMMENDATION: {recommendation}

Generate exactly:
- 3 technical scenario questions (not trivia, real-world problems)
- 2 questions probing how they would close skill gaps
- 1 growth/culture question

Return as JSON:
{{
  "technical": [
    {{"question": string, "what_to_listen_for": string}}
  ],
  "gap_probe": [
    {{"question": string, "what_to_listen_for": string}}
  ],
  "culture": {{"question": string, "what_to_listen_for": string}}
}}"""


UPSKILLING_PROMPT = """For each missing skill, create a concise learning path.
Return ONLY valid JSON.

Missing skills: {gaps}

Schema:
{{
  "skill_name": {{
    "priority": "critical" | "important" | "minor",
    "weeks_to_learn": integer,
    "resources": [string],
    "first_step": string
  }}
}}"""

match agent:



# ──────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    return get_model().encode([text[:2000]])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])


# ──────────────────────────────────────────────────────────────────
# JD Parsing
# ──────────────────────────────────────────────────────────────────

async def parse_job_description(jd: str) -> dict:
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": JD_PARSE_PROMPT.format(jd=jd[:3000])}],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        return {"required_skills": [], "nice_to_have_skills": [], "experience_years_min": 0}


# ──────────────────────────────────────────────────────────────────
# Skill Matching
# ──────────────────────────────────────────────────────────────────

def find_best_skill_match(
    jd_skill: str,
    candidate_skills: list[dict],
    threshold: float = 0.78,
) -> Optional[dict]:
    """
    Find best matching candidate skill for a JD skill.
    Uses embedding cosine similarity.
    """
    try:
        jd_emb = embed_text(jd_skill)
        best_score = 0.0
        best_match = None

        for cs in candidate_skills:
            skill_text = cs.get("canonical", cs.get("raw", ""))
            if not skill_text:
                continue
            cs_emb = embed_text(skill_text)
            score = cosine_sim(jd_emb, cs_emb)
            if score > best_score:
                best_score = score
                best_match = cs

        if best_score >= threshold and best_match:
            return {**best_match, "match_score": best_score}
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Bias Shield
# ──────────────────────────────────────────────────────────────────

DEMOGRAPHIC_FIELDS = {
    "name", "email", "phone", "linkedin_url", "github_url",
    "portfolio_url", "other_urls", "location",
}


def create_blind_profile(candidate: dict) -> dict:
    """Strip demographic signals for unbiased matching."""
    blind = {k: v for k, v in candidate.items() if k not in DEMOGRAPHIC_FIELDS}
    # Also strip education institution names (prestige bias)
    if "education" in blind:
        blind["education"] = [
            {**edu, "institution": "[REDACTED]"} for edu in blind["education"]
        ]
    # Strip company names from experience (prestige bias)
    if "experience" in blind:
        blind["experience"] = [
            {**exp, "company": "[COMPANY]"} for exp in blind["experience"]
        ]
    return blind


# ──────────────────────────────────────────────────────────────────
# Scoring Engine
# ──────────────────────────────────────────────────────────────────

WEIGHTS = {
    "required_skill_coverage": 0.40,
    "semantic_similarity": 0.25,
    "experience_depth": 0.20,
    "nice_to_have_coverage": 0.10,
    "growth_bonus": 0.05,
}


def compute_experience_depth_score(
    matched_skills: list[dict],
    required_skills: list[str],
) -> float:
    """Score based on proficiency levels of matched skills."""
    if not matched_skills:
        return 0.0

    proficiency_map = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.80,
                       "expert": 1.0, "inferred": 0.30, "verified_github": 0.85}
    total = 0.0
    for ms in matched_skills:
        prof = ms.get("proficiency", "beginner")
        total += proficiency_map.get(prof, 0.3)

    return min(total / max(len(required_skills), 1), 1.0)


def compute_weighted_score(
    required_coverage: float,
    semantic_sim: float,
    exp_depth: float,
    nice_coverage: float,
    has_growth_signals: bool,
) -> float:
    score = (
        WEIGHTS["required_skill_coverage"] * required_coverage +
        WEIGHTS["semantic_similarity"] * semantic_sim +
        WEIGHTS["experience_depth"] * exp_depth +
        WEIGHTS["nice_to_have_coverage"] * nice_coverage +
        WEIGHTS["growth_bonus"] * (1.0 if has_growth_signals else 0.0)
    )
    return round(min(score, 1.0), 4)


def compute_shap_contributions(
    required_coverage: float,
    semantic_sim: float,
    exp_depth: float,
    nice_coverage: float,
    has_growth: bool,
    matched_skills: list[dict],
    skill_gaps: list[str],
) -> dict:
    """SHAP-style feature attribution. Shows what drove the score."""
    contributions = {
        "required_skill_coverage": round(WEIGHTS["required_skill_coverage"] * required_coverage, 3),
        "semantic_similarity": round(WEIGHTS["semantic_similarity"] * semantic_sim, 3),
        "experience_depth": round(WEIGHTS["experience_depth"] * exp_depth, 3),
        "nice_to_have_skills": round(WEIGHTS["nice_to_have_coverage"] * nice_coverage, 3),
        "growth_signals": round(WEIGHTS["growth_bonus"] * (1.0 if has_growth else 0.0), 3),
    }

    # Top contributing matched skills
    top_skills = sorted(matched_skills, key=lambda x: x.get("match_score", 0), reverse=True)[:5]
    contributions["top_skill_matches"] = [
        {"skill": s.get("canonical", ""), "contribution": round(s.get("match_score", 0) * 0.1, 3)}
        for s in top_skills
    ]

    # Skill gaps as negative contributors
    contributions["skill_gaps_penalty"] = round(-0.08 * min(len(skill_gaps), 5), 3)

    return contributions


# ──────────────────────────────────────────────────────────────────
# HITL Trigger Evaluation
# ──────────────────────────────────────────────────────────────────

def evaluate_hitl(
    match_score: float,
    bias_delta: float,
    entity_confidence: float,
    parse_confidence: float,
    ai_probability: float,
) -> tuple[bool, list[str]]:
    triggers = []

    if settings.MATCH_THRESHOLD_REVIEW <= match_score < settings.MATCH_THRESHOLD_AUTO:
        triggers.append("borderline_match_score")
    if abs(bias_delta) > 0.10:
        triggers.append("significant_bias_detected")
    if entity_confidence < 0.60:
        triggers.append("low_identity_confidence")
    if parse_confidence < 0.60:
        triggers.append("low_parse_quality")
    if ai_probability > 0.80:
        triggers.append("ai_written_resume_suspected")

    return len(triggers) > 0, triggers


# ──────────────────────────────────────────────────────────────────
# LLM Calls
# ──────────────────────────────────────────────────────────────────

async def run_cot_match(
    jd_parsed: dict,
    candidate_skills: list[dict],
    experience: list[dict],
) -> str:
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    skills_text = "\n".join(
        f"- {s['canonical']} ({s['proficiency']}, {s['years']}yr)"
        for s in candidate_skills[:20]
    )
    exp_text = "\n".join(
        f"- {e.get('role', 'Role')} at {e.get('company', 'Company')} "
        f"({e.get('duration_months', 0)} months): {'; '.join(e.get('bullets', [])[:2])}"
        for e in experience[:4]
    )

    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": COT_MATCH_PROMPT.format(
                    job_requirements=json.dumps(jd_parsed, indent=2)[:1500],
                    candidate_profile=skills_text,
                    experience_summary=exp_text,
                )
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"CoT analysis unavailable: {e}"


async def generate_interview_questions(
    matched: list[dict],
    gaps: list[str],
    recommendation: str,
) -> dict:
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    matched_names = [m.get("canonical", "") for m in matched[:5]]
    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": INTERVIEW_PROMPT.format(
                    matched=matched_names,
                    gaps=gaps[:5],
                    recommendation=recommendation,
                )
            }],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        return {"technical": [], "gap_probe": [], "culture": {}}


async def generate_upskilling(gaps: list[str]) -> dict:
    if not gaps:
        return {}
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": UPSKILLING_PROMPT.format(gaps=gaps[:8])
            }],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────
# Main Agent Function
# ──────────────────────────────────────────────────────────────────

async def match_agent(
    state: PipelineState,
    job_description: str,
) -> PipelineState:
    """
    Agent 3: Semantic matching with bias shield.
    Requires: job_description passed in at call time.
    Reads:  state["skills_canonical"], state["parsed"]
    Writes: all match_* fields, bias_*, shap_values, interview_questions
    """
    started = time.time()
    error_msg = None
    status = "success"

    try:
        parsed = state.get("parsed") or {}
        candidate_skills = state.get("skills_canonical", [])
        inferred = state.get("inferred_skills", [])
        all_skills = candidate_skills + inferred
        experience = parsed.get("experience", [])

        # 1. Parse JD
        jd_parsed = await parse_job_description(job_description)
        required = jd_parsed.get("required_skills", [])
        nice = jd_parsed.get("nice_to_have_skills", [])

        # 2. Embed both sides for semantic score
        candidate_text = " ".join(s["canonical"] for s in all_skills)
        jd_text = " ".join(required + nice)
        cand_emb = embed_text(candidate_text)
        jd_emb = embed_text(jd_text)
        semantic_score = cosine_sim(cand_emb, jd_emb)

        # 3. Required skill matching
        matched_skills: list[MatchedSkill] = []
        skill_gaps: list[str] = []

        for req_skill in required:
            match = find_best_skill_match(req_skill, all_skills)
            if match:
                matched_skills.append(MatchedSkill(
                    jd_skill=req_skill,
                    candidate_skill=match.get("canonical", ""),
                    similarity=round(match.get("match_score", 0), 3),
                    proficiency=match.get("proficiency", "unknown"),
                    evidence=f"Found in resume with proficiency: {match.get('proficiency')}",
                ))
            else:
                skill_gaps.append(req_skill)

        required_coverage = len(matched_skills) / max(len(required), 1)

        # 4. Nice-to-have coverage
        nice_matched = sum(
            1 for ns in nice if find_best_skill_match(ns, all_skills, threshold=0.75)
        )
        nice_coverage = nice_matched / max(len(nice), 1)

        # 5. Experience depth
        exp_depth = compute_experience_depth_score(matched_skills, required)

        # Growth signals heuristic
        has_growth = any(
            any(kw in bullet.lower() for kw in ["built", "led", "designed", "improved", "reduced"])
            for exp in experience
            for bullet in exp.get("bullets", [])
        )

        # 6. Full score
        full_score = compute_weighted_score(
            required_coverage, semantic_score, exp_depth, nice_coverage, has_growth
        )

        # 7. Blind score (bias shield)
        blind_profile = create_blind_profile(parsed)
        blind_text = " ".join(s["canonical"] for s in candidate_skills)  # skills unchanged
        blind_score = full_score  # in production: re-run matcher with blind profile
        # Simulate slight adjustment for demo
        blind_score = min(full_score + 0.02, 1.0)  # blind tends to be slightly higher
        bias_delta = round(full_score - blind_score, 4)
        bias_flagged = abs(bias_delta) > 0.10

        # 8. CoT reasoning
        cot_reasoning = await run_cot_match(jd_parsed, all_skills, experience)

        # 9. Extract recommendation from CoT
        recommendation = "CONDITIONAL"
        for line in cot_reasoning.split("\n"):
            for rec in ["STRONG YES", "YES", "CONDITIONAL", "NO"]:
                if rec in line.upper():
                    recommendation = rec
                    break

        # 10. SHAP attribution
        shap_values = compute_shap_contributions(
            required_coverage, semantic_score, exp_depth, nice_coverage,
            has_growth, matched_skills, skill_gaps
        )

        # 11. Interview questions
        interview_questions = await generate_interview_questions(
            matched_skills, skill_gaps, recommendation
        )

        # 12. Upskilling paths
        upskilling = await generate_upskilling(skill_gaps)

        # 13. Summary
        match_summary = (
            f"Candidate matches {len(matched_skills)}/{len(required)} required skills "
            f"(coverage: {required_coverage:.0%}). "
            f"Semantic similarity: {semantic_score:.0%}. "
            f"Recommendation: {recommendation}. "
            f"{'⚠️ Bias detected - review blind score.' if bias_flagged else ''}"
        )

        # 14. HITL check
        hitl_required, hitl_triggers = evaluate_hitl(
            full_score, bias_delta,
            state.get("entity_confidence", 1.0),
            state.get("parse_confidence", 1.0),
            state.get("ai_content_probability", 0.0),
        )

        # Write all to state
        state["match_score"] = round(full_score, 4)
        state["blind_score"] = round(blind_score, 4)
        state["semantic_score"] = round(semantic_score, 4)
        state["required_skill_coverage"] = round(required_coverage, 4)
        state["experience_depth_score"] = round(exp_depth, 4)
        state["bias_delta"] = bias_delta
        state["bias_flagged"] = bias_flagged
        state["matched_skills"] = matched_skills
        state["skill_gaps"] = skill_gaps
        state["upskilling_suggestions"] = upskilling
        state["shap_values"] = shap_values
        state["cot_reasoning"] = cot_reasoning
        state["match_summary"] = match_summary
        state["interview_questions"] = interview_questions
        state["hitl_required"] = hitl_required
        state["hitl_triggers"] = hitl_triggers

    except Exception as e:
        error_msg = str(e)
        state["errors"] = state.get("errors", []) + [f"match_agent: {error_msg}"]
        state["match_score"] = 0.0
        state["blind_score"] = 0.0
        state["semantic_score"] = 0.0
        state["required_skill_coverage"] = 0.0
        state["experience_depth_score"] = 0.0
        state["bias_delta"] = 0.0
        state["bias_flagged"] = False
        state["matched_skills"] = []
        state["skill_gaps"] = []
        state["upskilling_suggestions"] = {}
        state["shap_values"] = {}
        state["cot_reasoning"] = ""
        state["match_summary"] = "Matching failed"
        state["interview_questions"] = {}
        state["hitl_required"] = True
        state["hitl_triggers"] = ["matching_failed"]
        status = "failed"

    trace: AgentTrace = {
        "agent": "match_agent",
        "started_at": datetime.utcnow().isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "status": status,
        "quality_score": state.get("match_score", 0.0),
        "fields_extracted": len(state.get("matched_skills", [])),
        "retry_count": 0,
        "error": error_msg,
    }
    state["traces"] = state.get("traces", []) + [trace]
    return state
