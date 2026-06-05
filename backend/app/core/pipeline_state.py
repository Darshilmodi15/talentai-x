"""
PipelineState is the single typed dict that flows through the LangGraph pipeline.
This is the contract every agent must read from and write to.
Every teammate must know this before writing any agent code.
"""
from typing import TypedDict, Optional
import uuid


class AgentTrace(TypedDict):
    agent: str
    started_at: str
    duration_ms: int
    status: str          # "success" | "retry" | "degraded" | "failed"
    quality_score: float
    fields_extracted: int
    retry_count: int
    error: Optional[str]


class SkillEntry(TypedDict):
    raw: Optional[str]
    canonical: str
    category: str
    proficiency: str     # "beginner" | "intermediate" | "advanced" | "expert" | "inferred"
    years: float
    inferred: bool
    source: str          # "resume" | "github" | "linkedin" | "inferred"


class ExperienceEntry(TypedDict):
    company: Optional[str]
    role: Optional[str]
    start: Optional[str]
    end: Optional[str]
    duration_months: int
    bullets: list[str]
    skills_mentioned: list[str]


class PlatformProfile(TypedDict):
    platform: str
    url: str
    handle: str
    display_name: Optional[str]
    bio: Optional[str]
    skills: list[str]
    location: Optional[str]
    email: Optional[str]
    links_found: list[str]
    confidence: float
    method: str          # "explicit_link" | "email_match" | "context_anchor" | "name_fuzzy"


class MatchedSkill(TypedDict):
    jd_skill: str
    candidate_skill: str
    similarity: float
    proficiency: str
    evidence: str


class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    job_id: str                    # UUID of the ParseJob row
    raw_file: bytes
    file_name: str
    file_type: str                 # "pdf" | "docx" | "txt"
    layout_type: str               # "single_column" | "two_column" | "table_heavy" | "image_based"

    # ── After Parser Agent ─────────────────────────────────
    parsed: Optional[dict]         # full structured extraction result
    parse_confidence: float
    resume_language: str
    ai_content_probability: float

    # ── After Normalizer Agent ─────────────────────────────
    skills_canonical: list[SkillEntry]
    experience_months_total: int
    inferred_skills: list[SkillEntry]
    emerging_skills_found: list[str]   # skills not in taxonomy

    # ── After Entity Resolver Agent ────────────────────────
    platform_profiles: dict[str, PlatformProfile]
    entity_confidence: float
    enriched_skills: list[SkillEntry]

    # ── Candidate ID (after DB write) ──────────────────────
    candidate_id: Optional[str]

    # ── After Matcher Agent ────────────────────────────────
    match_score: float
    blind_score: float
    semantic_score: float
    required_skill_coverage: float
    experience_depth_score: float
    bias_delta: float
    bias_flagged: bool
    matched_skills: list[MatchedSkill]
    skill_gaps: list[str]
    upskilling_suggestions: dict
    shap_values: dict
    cot_reasoning: str
    match_summary: str
    interview_questions: dict
    hitl_required: bool
    hitl_triggers: list[str]

    # ── Observability ──────────────────────────────────────
    traces: list[AgentTrace]
    errors: list[str]
    overall_status: str  # "completed" | "partial" | "failed"
