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
import random
from datetime import datetime, timezone
from typing import Optional, Any
import io
import logging

from app.core.pipeline_state import PipelineState, AgentTrace
from app.core.config import settings

import google.generativeai as genai

logger = logging.getLogger(__name__)

logger.warning("PARSE_AGENT_VERSION_2026_06_06_v3_QUOTA_FIX")

# Patterns that indicate Gemini quota/rate-limit errors
QUOTA_ERROR_PATTERNS = [
    "429",
    "too many requests",
    "resource exhausted",
    "resourceexhausted",
    "quota exceeded",
    "rate limit",
    "ratelimit",
    "generaterequestsperday",
]
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
        import pdfplumber
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
    import pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )


def extract_two_column(file_bytes: bytes) -> str:
    """Split page into left/right streams and extract each."""
    texts = []
    import pdfplumber
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
    import pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    table_text += " | ".join((cell or "") for cell in row) + "\n"
            body_text = page.extract_text() or ""
            texts.append(table_text + "\n" + body_text)
    return "\n\n".join(texts)


def extract_with_ocr(file_bytes: bytes) -> str:
    """For scanned/image PDFs — uses PyMuPDF + Tesseract."""
    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF fallback

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            text = pytesseract.image_to_string(img)
            texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        return f"OCR failed: {e}"


def extract_docx(file_bytes: bytes) -> str:
    import docx
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

class GeminiCallError(Exception):
    def __init__(self, message, raw_response, cleaned_response, traceback_str, exception_type):
        super().__init__(message)
        self.raw_response = raw_response
        self.cleaned_response = cleaned_response
        self.traceback_str = traceback_str
        self.exception_type = exception_type


class GeminiQuotaError(GeminiCallError):
    """Raised specifically for 429 / quota / rate-limit errors from Gemini.
    These should NEVER be silently swallowed."""
    pass


def _is_quota_error(exception: Exception) -> bool:
    """Check if an exception is a Gemini quota/rate-limit error."""
    error_str = str(exception).lower()
    return any(pattern in error_str for pattern in QUOTA_ERROR_PATTERNS)

async def call_gemini(prompt: str, max_tokens: int = 1500, _retries: int = 3) -> dict:
    """Single async Gemini call with exponential backoff for 429 errors.
    Returns parsed JSON or raises GeminiQuotaError / GeminiCallError."""
    import traceback
    content = ""
    cleaned_response = ""
    last_exception = None

    for attempt in range(_retries):
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(settings.GEMINI_MODEL)

            logger.info(f"Gemini call attempt {attempt + 1}/{_retries}, prompt length={len(prompt)}")

            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                ),
            )
            content = response.text.strip()
            logger.info(f"Gemini response received, length={len(content)}")
            logger.info(f"Raw Gemini response (first 500): {content[:500]}")

            # Remove markdown wrappers
            cleaned_response = content
            import re
            # Try standard markdown fence first
            fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if fence_match:
                cleaned_response = fence_match.group(1).strip()
            else:
                # Try to find JSON object boundaries directly
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    cleaned_response = json_match.group(0).strip()
                else:
                    # Last resort: strip backticks
                    cleaned_response = content.strip("`").strip()
                    if cleaned_response.startswith("json"):
                        cleaned_response = cleaned_response[4:].strip()

            logger.info(f"Cleaned response (first 500): {cleaned_response[:500]}")

            try:
                parsed = json.loads(cleaned_response)
            except Exception as e:
                logger.exception("JSON parse failed")
                raise ValueError(
                    f"JSON Parse Failed.\n"
                    f"Exception: {type(e).__name__}: {e}\n"
                    f"Raw response length: {len(content)}\n"
                    f"Cleaned response (first 500 chars): {cleaned_response[:500]}"
                )

            if not isinstance(parsed, dict):
                raise ValueError("Gemini did not return a JSON object")

            return parsed

        except Exception as e:
            last_exception = e
            logger.error(f"CALL_GEMINI_FAILED (attempt {attempt + 1}/{_retries}): {type(e).__name__}: {e}")

            # If it's a quota/rate-limit error, retry with exponential backoff
            if _is_quota_error(e):
                if attempt < _retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Gemini quota/rate-limit error. Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    # All retries exhausted for quota error — raise specific exception
                    logger.error(f"GEMINI QUOTA ERROR — all {_retries} retries exhausted")
                    raise GeminiQuotaError(
                        f"GEMINI_QUOTA_EXHAUSTED after {_retries} retries\n"
                        f"TYPE={type(e).__name__}\nERROR={str(e)}\n"
                        f"TRACEBACK={traceback.format_exc()}",
                        raw_response=content,
                        cleaned_response=cleaned_response,
                        traceback_str=traceback.format_exc(),
                        exception_type=type(e).__name__,
                    )
            else:
                # Non-quota error — don't retry, raise immediately
                raise GeminiCallError(
                    f"CALL_GEMINI_FAILED\nTYPE={type(e).__name__}\nERROR={str(e)}\n"
                    f"TRACEBACK={traceback.format_exc()}",
                    raw_response=content,
                    cleaned_response=cleaned_response,
                    traceback_str=traceback.format_exc(),
                    exception_type=type(e).__name__,
                )

    # Should never reach here, but just in case
    raise GeminiCallError(
        f"CALL_GEMINI_FAILED after {_retries} attempts",
        raw_response=content,
        cleaned_response=cleaned_response,
        traceback_str="",
        exception_type=type(last_exception).__name__ if last_exception else "Unknown",
    )


async def extract_all_parallel(raw_text: str) -> tuple[dict, dict, dict, dict]:
    """Run all 4 extraction prompts concurrently.
    Uses return_exceptions=True so one failed prompt doesn't kill the others.

    IMPORTANT: If ALL 4 calls fail, or ANY fail with a quota error,
    this function raises an exception instead of returning empty dicts.
    """
    # Truncate to avoid context window issues
    text_chunk = raw_text[:6000]
    logger.info(f"Starting parallel Gemini extraction, text chunk length={len(text_chunk)}")

    results = await asyncio.gather(
        call_gemini(PROMPT_BASIC_INFO.replace("{text}", text_chunk), max_tokens=800),
        call_gemini(PROMPT_EXPERIENCE.replace("{text}", text_chunk), max_tokens=1500),
        call_gemini(PROMPT_EDUCATION.replace("{text}", text_chunk), max_tokens=600),
        call_gemini(PROMPT_SKILLS_CERTS.replace("{text}", text_chunk), max_tokens=1000),
        return_exceptions=True,
    )

    labels = ["basic_info", "experience", "education", "skills_certs"]
    processed = []
    failures = []
    quota_errors = []

    for label, result in zip(labels, results):
        if isinstance(result, GeminiQuotaError):
            logger.error(
                f"QUOTA ERROR in '{label}': {type(result).__name__}: {result}"
            )
            quota_errors.append((label, result))
            failures.append((label, result))
            processed.append({})
        elif isinstance(result, Exception):
            logger.error(
                f"Gemini extraction '{label}' failed: "
                f"{type(result).__name__}: {result}"
            )
            failures.append((label, result))
            processed.append({})
        else:
            assert isinstance(result, dict)
            logger.info(f"Gemini extraction '{label}' succeeded: {list(result.keys())}")
            processed.append(result)

    # If ANY call hit a quota error, raise immediately — don't return partial empty data
    if quota_errors:
        label, err = quota_errors[0]
        raise GeminiQuotaError(
            f"Gemini quota/rate-limit error in '{label}' extraction. "
            f"{len(quota_errors)}/{len(labels)} calls hit quota limits. "
            f"Original error: {err}",
            raw_response=getattr(err, 'raw_response', ''),
            cleaned_response=getattr(err, 'cleaned_response', ''),
            traceback_str=getattr(err, 'traceback_str', ''),
            exception_type=getattr(err, 'exception_type', 'GeminiQuotaError'),
        )

    # If ALL 4 calls failed (non-quota), raise so parse_agent marks as failed
    if len(failures) == len(labels):
        label, err = failures[0]
        raise GeminiCallError(
            f"All {len(labels)} Gemini extraction calls failed. "
            f"First error ({label}): {type(err).__name__}: {err}",
            raw_response=getattr(err, 'raw_response', ''),
            cleaned_response=getattr(err, 'cleaned_response', ''),
            traceback_str=getattr(err, 'traceback_str', ''),
            exception_type=getattr(err, 'exception_type', type(err).__name__),
        )

    logger.info(f"Extraction results: {len(labels) - len(failures)}/{len(labels)} succeeded")
    return processed[0], processed[1], processed[2], processed[3]


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
    traceback_str = None
    exception_type = None
    raw_gemini_response = None
    cleaned_gemini_response = None

    try:
        logger.info(f"=== PARSE_AGENT START === job_id={state.get('job_id')} file={state.get('file_name')}")

        # Step 1: Detect layout for PDFs
        layout = "single_column"
        if state["file_type"] == "pdf":
            layout = detect_pdf_layout(state["raw_file"])
        logger.info(f"Layout detected: {layout}")

        state["layout_type"] = layout

        # Step 2: Extract raw text
        raw_text = get_raw_text(state["raw_file"], state["file_type"], layout)
        logger.info(f"PDF text extraction: {len(raw_text)} chars")
        logger.info(f"Extracted text (first 500 chars): {raw_text[:500]}")

        if not raw_text.strip():
            raise ValueError("No text could be extracted from the file")

        # Step 3: Detect language
        try:
            from langdetect import detect as detect_language
            lang = detect_language(raw_text[:500])
        except Exception:
            lang = "en"
        state["resume_language"] = lang
        logger.info(f"Language detected: {lang}")

        # Step 4: Detect AI content
        state["ai_content_probability"] = detect_ai_content(raw_text)

        # Step 5: Parallel LLM extraction (with one retry)
        basic, experience, education, skills = await extract_all_parallel(raw_text)

        # Retry if basic info is completely empty (non-quota failures only)
        if not basic and retry_count < 1:
            retry_count += 1
            logger.warning("Basic info empty, retrying extraction...")
            basic, experience, education, skills = await extract_all_parallel(raw_text)

        # Step 6: Merge
        parsed = merge_extractions(basic, experience, education, skills)
        logger.info(
            f"Merged parse result: name={parsed.get('name')}, "
            f"email={parsed.get('email')}, "
            f"phone={parsed.get('phone')}, "
            f"skills_count={len(parsed.get('skills', []))}, "
            f"experience_count={len(parsed.get('experience', []))}, "
            f"education_count={len(parsed.get('education', []))}"
        )

        state["parsed"] = parsed
        confidence = compute_parse_confidence(parsed)
        state["parse_confidence"] = confidence
        state["experience_months_total"] = parsed.get("experience_months_total", 0)

        # Validate: if confidence is 0, the parse is effectively empty
        if confidence == 0.0:
            logger.error(
                "Parse confidence is 0.0 — all key fields are empty. "
                "Marking as failed despite no exception."
            )
            status = "failed"
            error_msg = (
                "Parse completed but extracted zero data. "
                "All key fields (name, email, skills, experience, education) are empty. "
                "This likely indicates a Gemini API issue or incompatible resume format."
            )
            state["errors"] = state.get("errors", []) + [f"parse_agent: {error_msg}"]
        else:
            status = "success"
            logger.info(f"Parse succeeded with confidence={confidence}")

    except GeminiQuotaError as e:
        import traceback as tb
        error_msg = "Gemini quota exceeded"
        logger.error(f"=== PARSE_AGENT QUOTA ERROR === {str(e)}")
        state["errors"] = state.get("errors", []) + [error_msg]
        state["parsed"] = {}
        state["parse_confidence"] = 0.0
        state["layout_type"] = state.get("layout_type", "unknown")
        state["resume_language"] = state.get("resume_language", "en")
        state["ai_content_probability"] = state.get("ai_content_probability", 0.0)
        state["experience_months_total"] = 0
        status = "failed"

        # Do NOT log these to the state traces to avoid leaking to frontend
        traceback_str = None
        exception_type = None
        raw_gemini_response = None
        cleaned_gemini_response = None

    except Exception as e:
        import traceback as tb
        error_msg = str(e)
        logger.error(f"=== PARSE_AGENT ERROR === {type(e).__name__}: {error_msg}")
        state["errors"] = state.get("errors", []) + [f"parse_agent: {error_msg}"]
        state["parsed"] = {}
        state["parse_confidence"] = 0.0
        state["layout_type"] = state.get("layout_type", "unknown")
        state["resume_language"] = state.get("resume_language", "en")
        state["ai_content_probability"] = state.get("ai_content_probability", 0.0)
        state["experience_months_total"] = 0
        status = "failed"

        traceback_str = getattr(e, "traceback_str", tb.format_exc())
        exception_type = getattr(e, "exception_type", type(e).__name__)
        raw_gemini_response = getattr(e, "raw_response", None)
        cleaned_gemini_response = getattr(e, "cleaned_response", None)

    # Record trace
    fields_extracted = len([v for v in (state.get("parsed") or {}).values() if v])
    trace: AgentTrace = {
        "agent": "parse_agent",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.time() - started) * 1000),
        "status": status,
        "quality_score": state.get("parse_confidence", 0.0),
        "fields_extracted": fields_extracted,
        "retry_count": retry_count,
        "error": error_msg,
    }

    if status == "failed":
        trace["exception_type"] = exception_type
        trace["traceback"] = traceback_str
        trace["raw_gemini_response"] = raw_gemini_response
        trace["cleaned_gemini_response"] = cleaned_gemini_response

    logger.info(
        f"=== PARSE_AGENT END === status={status} fields_extracted={fields_extracted} "
        f"confidence={state.get('parse_confidence', 0.0)} duration_ms={trace['duration_ms']}"
    )

    state["traces"] = state.get("traces", []) + [trace]
    return state
