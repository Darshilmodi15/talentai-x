"""
Agent 1: Resume Parser
- Detects layout (single-col, two-col, table, image-based)
- Routes to correct text extractor
- Runs 4 specialized LLM prompts in PARALLEL
- Detects AI-written content
- Detects language
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Optional
import io

from app.core.pipeline_state import PipelineState, AgentTrace
from app.core.config import settings

# These imports are conditional — they work once installed
try:
    import pdfplumber
    import fitz  # PyMuPDF fallback
    import docx
    from langdetect import detect as detect_language
    import anthropic
    import numpy as np
except ImportError:
    pass  # handled at runtime with clear error messages


# ──────────────────────────────────────────────────────────────────
# PROMPTS — tightly scoped per extraction task
# ──────────────────────────────────────────────────────────────────

PROMPT_BASIC_INFO = """Extract ONLY basic personal information from this resume text.
Return ONLY valid JSON, no explanation, no markdown.

Schema:
{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "location": string or null,
  "summary": string or null,
  "linkedin_url": string or null,
  "github_url": string or null,
  "portfolio_url": string or null,
  "other_urls": [string]
}

If a field is not present, use null. Never invent data.

Resume text:
{text}"""


PROMPT_EXPERIENCE = """Extract ONLY work experience from this resume text.
Return ONLY valid JSON, no explanation, no markdown.

Schema:
{
  "experience": [
    {
      "company": string or null,
      "role": string or null,
      "start": string or null,
      "end": string or null,
      "duration_months": integer (estimate if not stated),
      "bullets": [string],
      "skills_mentioned": [string]
    }
  ],
  "experience_months_total": integer
}

Resume text:
{text}"""


PROMPT_EDUCATION = """Extract ONLY education from this resume text.
Return ONLY valid JSON, no explanation, no markdown.

Schema:
{
  "education": [
    {
      "institution": string or null,
      "degree": string or null,
      "field": string or null,
      "year": integer or null,
      "gpa": float or null
    }
  ]
}

Resume text:
{text}"""


PROMPT_SKILLS_CERTS = """Extract ONLY skills, certifications, and projects from this resume text.
Return ONLY valid JSON, no explanation, no markdown.

Schema:
{
  "skills": [string],
  "certifications": [
    {"name": string, "issuer": string or null, "year": integer or null}
  ],
  "projects": [
    {"name": string, "description": string, "tech": [string], "url": string or null}
  ],
  "publications": [string]
}

Resume text:
{text}"""


# ──────────────────────────────────────────────────────────────────
# Layout Detection
# ──────────────────────────────────────────────────────────────────

def detect_pdf_layout(file_bytes: bytes) -> str:
    """Classify PDF layout to route to correct extractor."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return "image_based"

            page = pdf.pages[0]
            words = page.extract_words()

            if not words:
                return "image_based"

            # Check for two-column layout by x-coordinate bimodal distribution
            x_coords = [w["x0"] for w in words]
            page_mid = page.width / 2

            left_count = sum(1 for x in x_coords if x < page_mid * 0.6)
            right_count = sum(1 for x in x_coords if x > page_mid * 1.4)

            if left_count > 30 and right_count > 30:
                return "two_column"

            # Check for table-heavy
            tables = page.extract_tables()
            if tables and len(tables) > 2:
                return "table_heavy"

            return "single_column"
    except Exception:
        return "single_column"


# ──────────────────────────────────────────────────────────────────
# Text Extractors per Layout
# ──────────────────────────────────────────────────────────────────

def extract_single_column(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )


def extract_two_column(file_bytes: bytes) -> str:
    """Split page into left/right streams and extract each."""
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            mid = page.width / 2
            left = page.crop((0, 0, mid, page.height)).extract_text() or ""
            right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
            texts.append(left + "\n" + right)
    return "\n\n".join(texts)


def extract_table_heavy(file_bytes: bytes) -> str:
    """Extract tables first, then remaining text."""
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    table_text += " | ".join(str(cell or "") for cell in row) + "\n"
            body_text = page.extract_text() or ""
            texts.append(table_text + "\n" + body_text)
    return "\n\n".join(texts)


def extract_with_ocr(file_bytes: bytes) -> str:
    """For scanned/image PDFs — uses PyMuPDF + Tesseract."""
    try:
        import pytesseract
        from PIL import Image

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        return f"OCR failed: {e}"


def extract_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)
    return "\n".join(parts)


def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


EXTRACTORS = {
    "single_column": extract_single_column,
    "two_column": extract_two_column,
    "table_heavy": extract_table_heavy,
    "image_based": extract_with_ocr,
}


def get_raw_text(file_bytes: bytes, file_type: str, layout: str) -> str:
    if file_type == "pdf":
        extractor = EXTRACTORS.get(layout, extract_single_column)
        return extractor(file_bytes)
    elif file_type == "docx":
        return extract_docx(file_bytes)
    else:
        return extract_txt(file_bytes)


# ──────────────────────────────────────────────────────────────────
# AI Content Detection
# ──────────────────────────────────────────────────────────────────

def detect_ai_content(text: str) -> float:
    """
    Heuristic AI-content detector.
    AI text has unnaturally uniform sentence lengths (low burstiness).
    Returns probability 0.0–1.0.
    """
    try:
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if len(sentences) < 5:
            return 0.0

        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        std = variance ** 0.5

        burstiness = std / mean if mean > 0 else 1.0
        # Low burstiness = suspicious
        ai_prob = max(0.0, min(1.0, 1.0 - (burstiness / 0.5)))
        return round(ai_prob, 3)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────
# LLM Calls — run in parallel
# ──────────────────────────────────────────────────────────────────

async def call_claude(prompt: str, max_tokens: int = 1500) -> dict:
    """Single async Claude call. Returns parsed JSON or empty dict."""
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}


async def extract_all_parallel(raw_text: str) -> tuple[dict, dict, dict, dict]:
    """Run all 4 extraction prompts concurrently."""
    # Truncate to avoid context window issues
    text_chunk = raw_text[:6000]

    basic, experience, education, skills = await asyncio.gather(
        call_claude(PROMPT_BASIC_INFO.format(text=text_chunk), max_tokens=800),
        call_claude(PROMPT_EXPERIENCE.format(text=text_chunk), max_tokens=1500),
        call_claude(PROMPT_EDUCATION.format(text=text_chunk), max_tokens=600),
        call_claude(PROMPT_SKILLS_CERTS.format(text=text_chunk), max_tokens=1000),
    )
    return basic, experience, education, skills


def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    """Merge the 4 parallel extraction results into one clean dict."""
    return {
        "name": basic.get("name"),
        "email": basic.get("email"),
        "phone": basic.get("phone"),
        "location": basic.get("location"),
        "summary": basic.get("summary"),
        "linkedin_url": basic.get("linkedin_url"),
        "github_url": basic.get("github_url"),
        "portfolio_url": basic.get("portfolio_url"),
        "other_urls": basic.get("other_urls", []),
        "experience": experience.get("experience", []),
        "experience_months_total": experience.get("experience_months_total", 0),
        "education": education.get("education", []),
        "skills": skills.get("skills", []),
        "certifications": skills.get("certifications", []),
        "projects": skills.get("projects", []),
        "publications": skills.get("publications", []),
    }


def compute_parse_confidence(parsed: dict) -> float:
    """Score how complete the parse is. 0.0–1.0"""
    key_fields = ["name", "email", "skills", "experience", "education"]
    filled = sum(1 for f in key_fields if parsed.get(f))
    return round(filled / len(key_fields), 2)


# ──────────────────────────────────────────────────────────────────
# Main Agent Function
# ──────────────────────────────────────────────────────────────────

async def parse_agent(state: PipelineState) -> PipelineState:
    """
    Agent 1: Parse resume from raw bytes.
    Writes: layout_type, parsed, parse_confidence, resume_language,
            ai_content_probability, traces
    """
    started = time.time()
    retry_count = 0
    error_msg = None

    try:
        # Step 1: Detect layout for PDFs
        layout = "single_column"
        if state["file_type"] == "pdf":
            layout = detect_pdf_layout(state["raw_file"])

        state["layout_type"] = layout

        # Step 2: Extract raw text
        raw_text = get_raw_text(state["raw_file"], state["file_type"], layout)

        if not raw_text.strip():
            raise ValueError("No text could be extracted from the file")

        # Step 3: Detect language
        try:
            lang = detect_language(raw_text[:500])
        except Exception:
            lang = "en"
        state["resume_language"] = lang

        # Step 4: Detect AI content
        state["ai_content_probability"] = detect_ai_content(raw_text)

        # Step 5: Parallel LLM extraction (with one retry)
        basic, experience, education, skills = await extract_all_parallel(raw_text)

        # Retry if basic info is completely empty
        if not basic and retry_count < 1:
            retry_count += 1
            basic, experience, education, skills = await extract_all_parallel(raw_text)

        # Step 6: Merge
        parsed = merge_extractions(basic, experience, education, skills)
        state["parsed"] = parsed
        state["parse_confidence"] = compute_parse_confidence(parsed)
        state["experience_months_total"] = parsed.get("experience_months_total", 0)
        status = "success"

    except Exception as e:
        error_msg = str(e)
        state["errors"] = state.get("errors", []) + [f"parse_agent: {error_msg}"]
        state["parsed"] = {}
        state["parse_confidence"] = 0.0
        state["layout_type"] = "unknown"
        state["resume_language"] = "en"
        state["ai_content_probability"] = 0.0
        state["experience_months_total"] = 0
        status = "failed"

    # Record trace
    trace: AgentTrace = {
        "agent": "parse_agent",
        "started_at": datetime.utcnow().isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "status": status,
        "quality_score": state.get("parse_confidence", 0.0),
        "fields_extracted": len([v for v in (state.get("parsed") or {}).values() if v]),
        "retry_count": retry_count,
        "error": error_msg,
    }

    state["traces"] = state.get("traces", []) + [trace]
    return state
