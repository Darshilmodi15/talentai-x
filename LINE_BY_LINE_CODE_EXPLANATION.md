# Parse Agent - Line-by-Line Code Knowledge

## 📄 File Header & Imports (Lines 1-28)

```python
"""
Agent 1: Resume Parser
- Detects layout (single-col, two-col, table, image-based)
- Routes to correct text extractor
- Runs 4 specialized LLM prompts in PARALLEL
- Detects AI-written content
- Detects language
"""
```
**What it does**: Module docstring explains the agent's purpose. Used for API documentation and IDE tooltips.

```python
import asyncio          # Enables async/await syntax and concurrent task execution
import json             # Parse JSON strings ↔ convert Python dicts to/from JSON
import time             # Measure execution time for performance tracking
from datetime import datetime  # Get ISO8601 timestamps for tracing
from typing import Optional   # Type hint for nullable values (Optional[str] = str | None)
import io               # Create in-memory file-like objects from bytes
```
**Why these imports?**
- `asyncio`: Allows parallel Claude API calls (4 at the same time)
- `json`: Claude returns JSON strings; need to parse them to Python dicts
- `time`: Measure how long parsing takes (stored in traces)
- `datetime`: ISO8601 timestamps for debugging logs
- `io`: Store PDF bytes in memory without hitting disk
- `typing`: Type hints help IDEs catch bugs early

```python
from app.core.pipeline_state import PipelineState, AgentTrace  # Import state container
from app.core.config import settings  # Import API keys, model names, settings
```
**What it does**:
- `PipelineState`: Dict-like object that flows through all agents; contains resume data
- `AgentTrace`: Type for execution metadata (duration, status, errors)

```python
try:
    import pdfplumber     # Extract text from PDFs with layout awareness
    import fitz           # PyMuPDF - fallback for OCR on scanned PDFs
    import docx           # Extract text from Word (.docx) files
    from langdetect import detect as detect_language  # Detect resume language
    import anthropic      # Claude API client for LLM calls
    import numpy as np    # Numerical operations (optional, may not be used)
except ImportError:
    pass                  # Don't crash server if packages missing; handle at runtime
```
**Why try/except?**
- These packages are optional dependencies
- If user doesn't install them, server starts anyway
- Errors caught at runtime when packages actually used
- Better UX than crashing on startup

---

## 🎯 Prompt Definitions (Lines 35-105)

```python
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
```

**Key points**:
- `{text}` is a placeholder → replaced with actual resume text at runtime
- `ONLY valid JSON` → prevents Claude from adding markdown/explanation
- `or null` → tells Claude missing data = null, not empty string or made-up values
- Lowercase keys + camelCase values → standardized format for backend

**Why separate prompts?** (BASIC_INFO, EXPERIENCE, EDUCATION, SKILLS_CERTS)
- Focused prompts → more accurate results
- Can run in parallel → faster execution
- Each has different max_tokens (experience=1500, basic=800)
- Easier to debug if one extraction fails

---

## 📐 Layout Detection Function (Lines 111-146)

```python
def detect_pdf_layout(file_bytes: bytes) -> str:
    """Classify PDF layout to route to correct extractor."""
```
**Input**: `file_bytes` = raw PDF bytes (e.g., from `open(file, 'rb').read()`)  
**Output**: `str` = layout type (`"single_column"` | `"two_column"` | `"table_heavy"` | `"image_based"`)

```python
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
```
**Line breakdown**:
- `io.BytesIO(file_bytes)` = convert raw bytes to file-like object (pdfplumber needs this)
- `with ... as pdf` = context manager (auto-closes file after block)
- `pdfplumber.open()` = parse PDF structure without loading entire file into memory

```python
            if not pdf.pages:
                return "image_based"
```
**Logic**: If PDF has 0 pages (corrupted or image-only format) → `"image_based"` (try OCR)

```python
            page = pdf.pages[0]  # Get first page only
            words = page.extract_words()  # Extract all word objects from page
```
**What `extract_words()` returns**:
```python
[
    {"x0": 50.0, "top": 100, "text": "John", ...},
    {"x0": 120.0, "top": 100, "text": "Doe", ...},
    {"x0": 300.0, "top": 100, "text": "Engineer", ...},  # Notice x0 > 300!
]
```
Pdfplumber data includes position on page (x0 = left edge in points)

```python
            if not words:
                return "image_based"
```
**Test**: If no words extracted → PDF is scanned/image → use OCR

```python
            x_coords = [w["x0"] for w in words]  # Extract all x-positions: [50, 120, 300, 315, ...]
            page_mid = page.width / 2  # Calculate page midpoint (e.g., width=612 → mid=306)
```
**Why extract x-coordinates?** Two-column layouts have words on left (x < 200) AND right (x > 400):
```
[Single column]          [Two columns]
All words x: 50-300      Left column: 50-200 | Right column: 350-500
```

```python
            left_count = sum(1 for x in x_coords if x < page_mid * 0.6)
            right_count = sum(1 for x in x_coords if x > page_mid * 1.4)
```
**Math explanation**:
- `page_mid * 0.6` = 60% → left boundary (e.g., 306 * 0.6 = 183.6)
- `page_mid * 1.4` = 140% → right boundary (e.g., 306 * 1.4 = 428.4)
- Count words in each region
- **Example**: If left_count=50, right_count=45 → bimodal distribution → two-column layout

```python
            if left_count > 30 and right_count > 30:
                return "two_column"
```
**Threshold**: Requires 30+ words on each side (rejects noise)

```python
            tables = page.extract_tables()  # Find table structures
            if tables and len(tables) > 2:  # More than 2 tables on page?
                return "table_heavy"
```
**Logic**: Resumes with 3+ tables (skills matrix, experience table, education table) → "table_heavy"

```python
            return "single_column"  # Default layout
    except Exception:
        return "single_column"  # If anything breaks, assume single-column (safest)
```

---

## 🗂️ Text Extraction Functions (Lines 152-232)

### extract_single_column
```python
def extract_single_column(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )
```
**Logic**:
1. Open PDF from bytes
2. Loop through ALL pages
3. `page.extract_text() or ""` = extract text OR use empty string if None
4. `"\n".join([...])` = join all pages with newline separator

**Example**:
```
Page 1: "John Doe\nSoftware Engineer"
Page 2: "Experience:\n• Tech Corp"
Result: "John Doe\nSoftware Engineer\nExperience:\n• Tech Corp"
```

### extract_two_column
```python
def extract_two_column(file_bytes: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            mid = page.width / 2  # Find page center
            # Crop LEFT half (0, 0, mid, height)
            left = page.crop((0, 0, mid, page.height)).extract_text() or ""
            # Crop RIGHT half (mid, 0, width, height)
            right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
            texts.append(left + "\n" + right)  # Combine left + right
    return "\n\n".join(texts)  # Join pages with double newline
```

**Why crop?** Prevents mixing up left/right columns:
```
Original layout:        After cropping:
NAME    | SKILLS       NAME
EXP     | CERT         EXP
        | AWARDS       SKILLS
                       CERT
                       AWARDS
```

### extract_table_heavy
```python
def extract_table_heavy(file_bytes: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()  # Extract ALL tables on page
            table_text = ""
            for table in tables:            # For each table
                for row in table:           # For each row
                    # Convert cells to strings, join with " | "
                    table_text += " | ".join(str(cell or "") for cell in row) + "\n"
            body_text = page.extract_text() or ""  # Extract regular text
            texts.append(table_text + "\n" + body_text)  # Tables THEN text
    return "\n\n".join(texts)
```

**Example**:
```
Table:
| Skill   | Years |
| Python  | 5     |

Converts to:
Skill | Years
Python | 5
```

### extract_with_ocr
```python
def extract_with_ocr(file_bytes: bytes) -> str:
    try:
        import pytesseract  # Google's OCR library
        from PIL import Image  # Convert images to PIL format

        doc = fitz.open(stream=file_bytes, filetype="pdf")  # Use PyMuPDF
        texts = []
        for page in doc:
            mat = fitz.Matrix(2, 2)  # 2x zoom = 200% enlargement
            pix = page.get_pixmap(matrix=mat)  # Render page to image at 2x zoom
            # Convert pixel data to RGB Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)  # OCR the image
            texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        return f"OCR failed: {e}"
```

**Why 2x zoom?** OCR works better on larger images; 2x zoom improves accuracy

### extract_docx
```python
def extract_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))  # Parse .docx
    parts = []
    for para in doc.paragraphs:  # Loop through paragraphs
        if para.text.strip():    # Skip empty paragraphs
            parts.append(para.text)
    for table in doc.tables:     # Loop through tables
        for row in table.rows:
            # Join cells in row with " | "
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():  # Skip empty rows
                parts.append(row_text)
    return "\n".join(parts)
```

### extract_txt
```python
def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")
```
**Explanation**:
- `bytes.decode("utf-8")` = convert bytes to text using UTF-8 encoding
- `errors="replace"` = if invalid UTF-8 found, replace with `?` instead of crashing

### The EXTRACTORS Dictionary
```python
EXTRACTORS = {
    "single_column": extract_single_column,
    "two_column": extract_two_column,
    "table_heavy": extract_table_heavy,
    "image_based": extract_with_ocr,
}
```
**Pattern**: Maps layout type → function. Used for routing (lookup table).

### get_raw_text Dispatcher
```python
def get_raw_text(file_bytes: bytes, file_type: str, layout: str) -> str:
    if file_type == "pdf":
        extractor = EXTRACTORS.get(layout, extract_single_column)  # Lookup function; default to single_column
        return extractor(file_bytes)  # Call the function
    elif file_type == "docx":
        return extract_docx(file_bytes)
    else:
        return extract_txt(file_bytes)
```
**Flow**:
```
get_raw_text(bytes, "pdf", "two_column")
 ├─ Lookup EXTRACTORS["two_column"] → extract_two_column function
 ├─ Call extract_two_column(bytes)
 └─ Return text result
```

---

## 🤖 AI Content Detection (Lines 240-271)

```python
def detect_ai_content(text: str) -> float:
    """
    Heuristic AI-content detector.
    AI text has unnaturally uniform sentence lengths (low burstiness).
    Returns probability 0.0–1.0.
    """
```
**Core insight**: AI models generate uniform sentence lengths; humans vary naturally.

```python
    try:
        import re  # Regular expressions
        # Split on sentence endings, keep only sentences > 10 words
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
```
**Example**:
```
Text: "I built systems. I led teams. My achievements include Java coding and Docker."
Regex split on . ! ? → ["I built systems", "I led teams", "My achievements include Java coding and Docker"]
Filter (>10 words) → ["My achievements include Java coding and Docker"]  #only this is >10 words
```

```python
        if len(sentences) < 5:
            return 0.0  # Too few sentences; can't analyze → assume human
```

```python
        lengths = [len(s.split()) for s in sentences]  # Word count per sentence
        # Example: [10, 12, 8, 14, 11] → word counts
```

```python
        mean = sum(lengths) / len(lengths)  # Average: (10+12+8+14+11)/5 = 11
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)  # Spread²
        # Variance = ((10-11)² + (12-11)² + ... ) / 5 = (1 + 1 + 9 + 9 + 0) / 5 = 4
        std = variance ** 0.5  # Standard deviation = sqrt(4) = 2
```

**Statistics explanation**:
- **mean** = average sentence length
- **variance** = average squared deviation
- **std** = square root of variance (on same scale as original)

```python
        burstiness = std / mean if mean > 0 else 1.0
        # Burstiness = 2 / 11 = 0.18 (LOW burstiness = uniform = AI)
        # If std=5, mean=10 → 0.5 (good variation = human)
```

```python
        ai_prob = max(0.0, min(1.0, 1.0 - (burstiness / 0.5)))
        # 1.0 - (0.18 / 0.5) = 1.0 - 0.36 = 0.64
        # max/min clamp result to 0.0–1.0 range
```

**Conversion**:
- burstiness=0 → ai_prob = 1.0 - 0 = 1.0 (100% AI)
- burstiness=0.5 → ai_prob = 1.0 - 1.0 = 0.0 (human)
- burstiness=0.25 → ai_prob = 1.0 - 0.5 = 0.5 (50% likely AI)

```python
        return round(ai_prob, 3)  # Round to 3 decimals: 0.642 → 0.642
    except Exception:
        return 0.0  # On any error, assume safe (no AI)
```

---

## 🔄 Parallel Claude Calls (Lines 278-336)

```python
async def call_claude(prompt: str, max_tokens: int = 1500) -> dict:
    """Single async Claude call. Returns parsed JSON or empty dict."""
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
```
**Why AsyncAnthropic?** Async client allows `await` = non-blocking (can run in parallel)

```python
    try:
        response = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,        # "claude-3.5-sonnet" etc
            max_tokens=max_tokens,                 # Token limit for response
            messages=[{"role": "user", "content": prompt}],  # Single-turn message
        )
```
**What happens**:
1. `await` = pause execution until Claude responds
2. Multiple `call_claude()` calls can run simultaneously
3. Request sent to Claude API with prompt + resume text

```python
        content = response.content[0].text.strip()  # Extract text response
```
**Why `[0]`?** Claude can return multiple content types; `.text` is the first one

```python
        if content.startswith("```"):  # Claude often wraps JSON in markdown
            content = content.split("```")[1]  # Extract middle part
            if content.startswith("json"):  # Remove "```json" prefix
                content = content[4:]
```

**Example**:
```
Claude returns:
```json
{"name": "John"}
```

After stripping:
{"name": "John"}
```

```python
        return json.loads(content)  # Parse JSON string → Python dict
    except json.JSONDecodeError:
        return {}  # If JSON invalid, return empty dict (safe fail)
    except Exception:
        return {}  # On any error, return empty dict
```

### Extract All Parallel
```python
async def extract_all_parallel(raw_text: str) -> tuple[dict, dict, dict, dict]:
    """Run all 4 extraction prompts concurrently."""
    text_chunk = raw_text[:6000]  # Truncate to 6000 chars (avoid token limit)
```
**Why truncate?** Claude has token limit (~10k); 6000 chars ≈ 1500 tokens

```python
    basic, experience, education, skills = await asyncio.gather(
        call_claude(PROMPT_BASIC_INFO.format(text=text_chunk), max_tokens=800),
        call_claude(PROMPT_EXPERIENCE.format(text=text_chunk), max_tokens=1500),
        call_claude(PROMPT_EDUCATION.format(text=text_chunk), max_tokens=600),
        call_claude(PROMPT_SKILLS_CERTS.format(text=text_chunk), max_tokens=1000),
    )
```
**What `asyncio.gather()` does**:
```
Without gather (SEQUENTIAL):
Call 1 [===] → 5 sec
Call 2      [====] → 6 sec
Call 3           [==] → 2 sec
Call 4              [===] → 5 sec
Total: ~18 seconds

With gather (PARALLEL):
Call 1 [===]
Call 2 [====]
Call 3 [==]
Call 4 [===]
Total: ~6 seconds (max of all)
```

```python
    return basic, experience, education, skills  # Return 4-tuple
```

---

## 🔀 Merge Extractions Function (Lines 339-369)

```python
def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    """Merge the 4 parallel extraction results into one clean dict."""
    return {
        "name": basic.get("name"),  # Get from basic, default to None if missing
        "email": basic.get("email"),
        ...
        "experience": experience.get("experience", []),  # Default to empty list
        "experience_months_total": experience.get("experience_months_total", 0),
        "education": education.get("education", []),
        ...
    }
```
**Pattern**: `.get(key, default)` = safe access (returns default if key missing)
- `basic.get("name")` = return name OR None
- `experience.get("experience", [])` = return experience list OR empty list

---

## 📊 Confidence Scoring (Lines 372-379)

```python
def compute_parse_confidence(parsed: dict) -> float:
    """Score how complete the parse is. 0.0–1.0"""
    key_fields = ["name", "email", "skills", "experience", "education"]
    filled = sum(1 for f in key_fields if parsed.get(f))
    # Generator expression: count non-empty fields
    # Example: name="John", email="john@", skills=[], exp=[..], edu=[]
    #         → 3 fields filled (name, email, experience)
    return round(filled / len(key_fields), 2)
    # 3 / 5 = 0.60
```

---

## 🚀 Main Agent Function (Lines 385-455)

```python
async def parse_agent(state: PipelineState) -> PipelineState:
    """
    Agent 1: Parse resume from raw bytes.
    Writes: layout_type, parsed, parse_confidence, resume_language,
            ai_content_probability, traces
    """
    started = time.time()  # Timestamp for duration calculation
    retry_count = 0        # Track retry attempts
    error_msg = None       # Store error message if occurs
```

### Step 1: Layout Detection
```python
    try:
        layout = "single_column"  # Default layout
        if state["file_type"] == "pdf":
            layout = detect_pdf_layout(state["raw_file"])
        state["layout_type"] = layout  # Store detected layout
```

### Step 2: Text Extraction
```python
        raw_text = get_raw_text(state["raw_file"], state["file_type"], layout)
        
        if not raw_text.strip():
            raise ValueError("No text could be extracted from the file")
```
**Validation**: Fail if text empty (corrupted/unreadable file)

### Step 3: Language Detection
```python
        try:
            lang = detect_language(raw_text[:500])  # Detect from first 500 chars
        except Exception:
            lang = "en"  # Default to English if detection fails
        state["resume_language"] = lang
```

### Step 4: AI Content Detection
```python
        state["ai_content_probability"] = detect_ai_content(raw_text)
```
Stores probability 0.0–1.0 in state

### Step 5: Parallel LLM Extraction
```python
        basic, experience, education, skills = await extract_all_parallel(raw_text)
        
        # Retry if basic info is completely empty
        if not basic and retry_count < 1:
            retry_count += 1
            basic, experience, education, skills = await extract_all_parallel(raw_text)
```
**Retry logic**: If basic info empty, try one more time (handles transient API failures)

### Step 6: Merge & Score
```python
        parsed = merge_extractions(basic, experience, education, skills)
        state["parsed"] = parsed
        state["parse_confidence"] = compute_parse_confidence(parsed)
        state["experience_months_total"] = parsed.get("experience_months_total", 0)
        status = "success"
```

### Error Handling
```python
    except Exception as e:
        error_msg = str(e)
        state["errors"] = state.get("errors", []) + [f"parse_agent: {error_msg}"]
        # Append error to errors list (other agents may have added errors)
        state["parsed"] = {}
        state["parse_confidence"] = 0.0
        state["layout_type"] = "unknown"
        state["resume_language"] = "en"
        state["ai_content_probability"] = 0.0
        state["experience_months_total"] = 0
        status = "failed"
```
**Defensive defaults**: Set all fields to safe values on error

### Tracing
```python
    trace: AgentTrace = {
        "agent": "parse_agent",
        "started_at": datetime.utcnow().isoformat(),  # ISO8601 timestamp
        "duration_ms": int((time.time() - started) * 1000),  # Elapsed time in ms
        "status": status,  # "success" or "failed"
        "quality_score": state.get("parse_confidence", 0.0),  # Extracted confidence
        "fields_extracted": len([v for v in (state.get("parsed") or {}).values() if v]),
        # Count non-empty values in parsed dict
        "retry_count": retry_count,  # How many retries needed
        "error": error_msg,  # Error message if failed
    }

    state["traces"] = state.get("traces", []) + [trace]
    # Append trace to traces list (other agents may have added traces)
    return state
```

**Tracing purpose**: Record how parsing went for debugging/monitoring:
```json
{
  "agent": "parse_agent",
  "started_at": "2026-04-11T10:30:45.123Z",
  "duration_ms": 2847,
  "status": "success",
  "quality_score": 0.85,
  "fields_extracted": 18,
  "retry_count": 0,
  "error": null
}
```

---

## 📈 Data Flow Visualization

```
Input: state = {
    "raw_file": <bytes>,
    "file_type": "pdf",
    "file_name": "resume.pdf",
    "job_id": "abc123"
}

Step 1: detect_pdf_layout()
  └─ analyze x-coordinates → "two_column"

Step 2: get_raw_text()
  └─ extract_two_column()
    └─ split page left/right → raw_text = "John Doe\n...\nSkills: Python, ..."

Step 3: detect_language()
  └─ "en"

Step 4: detect_ai_content()
  └─ 0.05 (5% likely AI)

Step 5: extract_all_parallel()
  ├─ call_claude(PROMPT_BASIC_INFO) → {name: "John Doe", email: "john@..."}
  ├─ call_claude(PROMPT_EXPERIENCE) → {experience: [...], experience_months_total: 48}
  ├─ call_claude(PROMPT_EDUCATION) → {education: [...]}
  └─ call_claude(PROMPT_SKILLS_CERTS) → {skills: [...], certifications: [...]}
      (All run in parallel with asyncio.gather())

Step 6: merge_extractions()
  └─ combined_dict = {...all extracted fields...}

Step 7: compute_parse_confidence()
  └─ 0.85 (85% confidence)

Output: state = {
    "raw_file": <bytes>,
    "file_type": "pdf",
    "file_name": "resume.pdf",
    "job_id": "abc123",
    "layout_type": "two_column",
    "parsed": {...full extracted resume...},
    "parse_confidence": 0.85,
    "resume_language": "en",
    "ai_content_probability": 0.05,
    "experience_months_total": 48,
    "traces": [{...trace info...}]
}
```

---

## 🔑 Key Python Concepts Used

| Concept | Example | Why |
|---------|---------|-----|
| **async/await** | `await call_claude()` | Run 4 Claude calls simultaneously |
| **Generator expressions** | `sum(1 for f in fields if ...)` | Memory-efficient counting |
| **.get() with default** | `dict.get("key", default)` | Safe dict access |
| **Type hints** | `def func(x: str) -> int:` | IDE autocomplete + error detection |
| **Context managers** | `with open() as f:` | Auto-close files/connections |
| **try/except** | `try: ... except: ...` | Graceful error handling |
| **List comprehensions** | `[x for x in list if ...]` | Concise filtering |
| **Dictionary unpacking** | `{**dict1, **dict2}` | Merge multiple dicts |
| **f-strings** | `f"Error: {error_msg}"` | String interpolation |
| **Tuples** | `(a, b, c) = function()` | Return multiple values |

---

## 🎯 Execution Flow Summary

```
parse_agent(state) called
    ↓
IF PDF → detect_pdf_layout()
    ↓
get_raw_text() → extract based on layout or file type
    ↓
detect_language() from first 500 chars
    ↓
detect_ai_content() → calculate burstiness
    ↓
extract_all_parallel() →  4 Claude calls (PARALLEL, not sequential)
    ├─ basic info
    ├─ experience
    ├─ education
    └─ skills
    ↓
IF all failed → retry once
    ↓
merge_extractions() → combine 4 results into 1 dict
    ↓
compute_parse_confidence() → quality score 0-1
    ↓
Record trace (duration, status, fields extracted, errors)
    ↓
RETURN updated state
```

