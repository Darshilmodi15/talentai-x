import uuid
from datetime import datetime
from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime,
    ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.database import Base
import enum


class ProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class MatchStatus(str, enum.Enum):
    AUTO_MERGED = "auto_approved"
    PENDING_REVIEW = "pending_review"
    UNRESOLVED = "unresolved"


# ──────────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────────
class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ──────────────────────────────────────────────────────────────────
# Parse Jobs
# ──────────────────────────────────────────────────────────────────
class ParseJob(Base):
    __tablename__ = "parse_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus), default=ProcessingStatus.QUEUED
    )
    file_name: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(20))
    file_path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    traces: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # relationship
    candidate: Mapped["Candidate | None"] = relationship("Candidate", back_populates="parse_job")


# ──────────────────────────────────────────────────────────────────
# Candidates
# ──────────────────────────────────────────────────────────────────
class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parse_job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("parse_jobs.id"), nullable=False)

    # Basic Info
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Parsed Data (full JSON)
    parsed_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Normalized Skills
    skills_canonical: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Entity Resolution
    github_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    linkedin_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_platforms: Mapped[dict] = mapped_column(JSON, default=dict)

    # Enrichment
    enriched_skills: Mapped[list] = mapped_column(JSON, default=list)
    ai_content_probability: Mapped[float] = mapped_column(Float, default=0.0)
    parse_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Metadata
    resume_language: Mapped[str] = mapped_column(String(10), default="en")
    experience_months_total: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    parse_job: Mapped["ParseJob"] = relationship("ParseJob", back_populates="candidate")
    experiences: Mapped[list["Experience"]] = relationship("Experience", back_populates="candidate", cascade="all, delete-orphan")
    educations: Mapped[list["Education"]] = relationship("Education", back_populates="candidate", cascade="all, delete-orphan")
    match_results: Mapped[list["MatchResult"]] = relationship("MatchResult", back_populates="candidate")


class Experience(Base):
    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    company: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[str | None] = mapped_column(String(200))
    start_date: Mapped[str | None] = mapped_column(String(50))
    end_date: Mapped[str | None] = mapped_column(String(50))
    duration_months: Mapped[int] = mapped_column(Integer, default=0)
    bullets: Mapped[list] = mapped_column(JSON, default=list)
    skills_mentioned: Mapped[list] = mapped_column(JSON, default=list)

    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="experiences")


class Education(Base):
    __tablename__ = "educations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    institution: Mapped[str | None] = mapped_column(String(200))
    degree: Mapped[str | None] = mapped_column(String(200))
    field: Mapped[str | None] = mapped_column(String(200))
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpa: Mapped[float | None] = mapped_column(Float, nullable=True)

    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="educations")


# ──────────────────────────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────────────────────────
class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    company: Mapped[str | None] = mapped_column(String(200), nullable=True)
    required_skills: Mapped[list] = mapped_column(JSON, default=list)
    nice_to_have_skills: Mapped[list] = mapped_column(JSON, default=list)
    experience_years_min: Mapped[int] = mapped_column(Integer, default=0)
    parsed_requirements: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match_results: Mapped[list["MatchResult"]] = relationship("MatchResult", back_populates="job")


# ──────────────────────────────────────────────────────────────────
# Match Results
# ──────────────────────────────────────────────────────────────────
class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"))

    # Scores
    match_score: Mapped[float] = mapped_column(Float)
    blind_score: Mapped[float] = mapped_column(Float)
    semantic_score: Mapped[float] = mapped_column(Float)
    required_skill_coverage: Mapped[float] = mapped_column(Float)
    experience_depth_score: Mapped[float] = mapped_column(Float)

    # Bias
    bias_delta: Mapped[float] = mapped_column(Float, default=0.0)
    bias_flagged: Mapped[bool] = mapped_column(Boolean, default=False)

    # Details
    matched_skills: Mapped[list] = mapped_column(JSON, default=list)
    skill_gaps: Mapped[list] = mapped_column(JSON, default=list)
    upskilling_suggestions: Mapped[dict] = mapped_column(JSON, default=dict)
    shap_values: Mapped[dict] = mapped_column(JSON, default=dict)
    cot_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    interview_questions: Mapped[dict] = mapped_column(JSON, default=dict)

    # HITL
    status: Mapped[MatchStatus] = mapped_column(SAEnum(MatchStatus), default=MatchStatus.AUTO_MERGED)
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    candidate: Mapped["Candidate"] = relationship("Candidate", back_populates="match_results")
    job: Mapped["Job"] = relationship("Job", back_populates="match_results")


# ──────────────────────────────────────────────────────────────────
# Skill Taxonomy
# ──────────────────────────────────────────────────────────────────
class SkillTaxonomy(Base):
    __tablename__ = "skill_taxonomy"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100))       # technical, soft, domain
    parent: Mapped[str | None] = mapped_column(String(200), nullable=True)
    synonyms: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(50), default="manual")  # manual, auto_discovered
    chroma_embedded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EmergingSkill(Base):
    __tablename__ = "emerging_skills"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_skill: Mapped[str] = mapped_column(String(200))
    count: Mapped[int] = mapped_column(Integer, default=1)
    contexts: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, approved, rejected
    proposed_canonical: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────
# HITL Review Queue
# ──────────────────────────────────────────────────────────────────
class HITLReviewItem(Base):
    __tablename__ = "hitl_queue"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("match_results.id"))
    trigger_reason: Mapped[str] = mapped_column(String(100))
    priority: Mapped[str] = mapped_column(String(20), default="normal")
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────
# Webhooks
# ──────────────────────────────────────────────────────────────────
class Webhook(Base):
    __tablename__ = "webhooks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    callback_url: Mapped[str] = mapped_column(String(500))
    secret: Mapped[str] = mapped_column(String(100))
    events: Mapped[list] = mapped_column(JSON, default=list)  # ["job.completed", "match.done"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
