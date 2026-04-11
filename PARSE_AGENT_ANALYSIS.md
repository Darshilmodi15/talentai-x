# Parse Agent Analysis & Agent Flow Roadmap

## 📋 Parse Agent Overview

The **Parse Agent** is **Agent 1** in the TalentAI-X multi-agent pipeline. It transforms raw resume file bytes into structured, standardized JSON data about a candidate's professional profile.

---

## 🔍 What Does Parse Agent Do?

### **Core Responsibilities**

| Function | Purpose |
|----------|---------|
| **Layout Detection** | Classifies PDF structure (single-column, two-column, table-heavy, image-based) |
| **Text Extraction** | Routes to appropriate extractor based on layout |
| **Parallel LLM Extraction** | Runs 4 Claude prompts simultaneously to extract different information types |
| **Language Detection** | Identifies resume language using `langdetect` library |
| **AI Content Detection** | Flags AI-written content using burstiness heuristic |
| **Confidence Scoring** | Computes parse quality (0.0–1.0 scale) |
| **Error Handling** | Graceful failures with retry logic |

---

## 🏗️ Detailed Architecture

### **Step 1: Layout Detection**
```python
detect_pdf_layout(file_bytes) → str
```
**Input**: Raw PDF bytes  
**Output**: Layout type → `"single_column" | "two_column" | "table_heavy" | "image_based"`

**Logic**:
- Checks if PDF has text layers (if not → "image_based")
- Analyzes word x-coordinates for bimodal distribution (if found → "two_column")
- Counts tables to detect table-heavy documents
- Default: "single_column"

---

### **Step 2: Text Extraction**
Conditional extraction based on file type:
```
PDF → detect layout → select extractor
├─ single_column() - Standard linear extraction
├─ two_column()    - Splits page mid-point (left/right streams)
├─ table_heavy()   - Extracts tables THEN body text
└─ extract_with_ocr() - Tesseract for scanned/image PDFs

DOCX → extract_docx() - Parse paragraphs + tables

TXT → extract_txt() - UTF-8 decoding
```

---

### **Step 3: Parallel LLM Extraction** ⚡
Runs **4 specialized Claude prompts concurrently** for different data types:

```python
async def extract_all_parallel(raw_text):
    # All 4 calls happen at the SAME TIME, not sequentially
    return await asyncio.gather(
        call_claude(PROMPT_BASIC_INFO),      # Name, email, phone, URLs
        call_claude(PROMPT_EXPERIENCE),      # Work experience + duration
        call_claude(PROMPT_EDUCATION),       # Schools, degrees, GPA
        call_claude(PROMPT_SKILLS_CERTS),    # Skills, certifications, projects
    )
```

**Each Prompt Extracts**:

1. **BASIC INFO**
   - Name, email, phone, location
   - Summary/headline
   - LinkedIn, GitHub, portfolio URLs

2. **EXPERIENCE**
   - Company, role, dates
   - Duration in months
   - Job responsibilities (bullets)
   - Skills mentioned in context

3. **EDUCATION**
   - Institution, degree, field
   - Graduation year, GPA

4. **SKILLS & CERTIFICATIONS**
   - Skill list
   - Certifications with issuer/year
   - Personal projects with tech stack
   - Publications/research

---

### **Step 4: Language Detection**
```python
detect_language(raw_text[:500]) → str
# Returns: "en", "fr", "de", "zh", etc.
# Fallback: "en" if detection fails
```

---

### **Step 5: AI Content Detection**
```python
detect_ai_content(text) → float (0.0–1.0)
```

**Heuristic Algorithm**:
- AI-written text has **unnaturally uniform sentence lengths** (low "burstiness")
- Human writing naturally varies sentence length (high burstiness)
- **Low burstiness** = high AI probability

**Calculation**:
1. Extract sentences (length > 10 words)
2. Compute standard deviation of sentence lengths
3. Burstiness = std_dev / mean_length
4. AI_probability = 1.0 - (burstiness / 0.5) [clamped 0–1]

**Why it matters**: Flags potentially fake/generated resumes

---

### **Step 6: Data Merging & Confidence Scoring**

```python
merge_extractions(basic, experience, education, skills) → dict
```

Creates unified resume document with all 4 extractions merged.

```python
compute_parse_confidence(parsed_dict) → float
```

Scores based on key field presence:
- **Key fields**: name, email, skills, experience, education
- **Score** = (fields_present / 5) → 0.0 to 1.0
- **Example**: Has name + email + skills + experience (4/5) = **0.80 confidence**

---

## 📤 Parse Agent Output

```python
state dictionary updated with:
{
    "layout_type": str,                    # "single_column" | "two_column" | ...
    "resume_language": str,                # "en", "fr", etc.
    "ai_content_probability": float,       # 0.0–1.0
    "parse_confidence": float,             # 0.0–1.0 (data quality)
    "experience_months_total": int,        # Total career months
    "parsed": {
        "name": str | null,
        "email": str | null,
        "phone": str | null,
        "location": str | null,
        "summary": str | null,
        "linkedin_url": str | null,
        "github_url": str | null,
        "portfolio_url": str | null,
        "other_urls": [str],
        
        "experience": [
            {
                "company": str,
                "role": str,
                "start": str,  # e.g., "2022-01"
                "end": str,
                "duration_months": int,
                "bullets": [str],
                "skills_mentioned": [str]
            }
        ],
        "experience_months_total": int,
        
        "education": [
            {
                "institution": str,
                "degree": str,
                "field": str,
                "year": int,
                "gpa": float
            }
        ],
        
        "skills": [str],
        "certifications": [
            {
                "name": str,
                "issuer": str | null,
                "year": int | null
            }
        ],
        "projects": [
            {
                "name": str,
                "description": str,
                "tech": [str],
                "url": str | null
            }
        ],
        "publications": [str]
    },
    
    "traces": [
        {
            "agent": "parse_agent",
            "started_at": ISO8601,
            "duration_ms": int,
            "status": "success" | "failed",
            "quality_score": float,
            "fields_extracted": int,
            "retry_count": int,
            "error": str | null
        }
    ]
}
```

---

## 🔗 Multi-Agent Pipeline Roadmap

```
┌─────────────────────────────────────────────────────────────┐
│                   USER UPLOADS RESUME                       │
│            (PDF, DOCX, or TXT format)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │     PARSE AGENT (Agent 1)    │ ← YOU ARE HERE
        │  Data Extraction & Layout    │
        │  - Extract text from PDF     │
        │  - Parallel LLM prompts      │
        │  - Language detection        │
        │  - AI content detection      │
        └──────────────────┬───────────┘
                           │
         OUTPUT: Structured resume JSON
         (parsed skills, experience, education)
                           │
                           ▼
        ┌──────────────────────────────┐
        │  NORMALIZE AGENT (Agent 2)   │ ← NEXT STEP
        │  Skill Standardization       │
        │  - Synonym mapping (JS→JS)   │
        │  - Fuzzy matching            │
        │  - Semantic embedding search │
        │  - Skill categorization      │
        │  - Proficiency estimation    │
        └──────────────────┬───────────┘
                           │
         OUTPUT: Normalized skills with tags
                           │
                           ▼
        ┌──────────────────────────────┐
        │    MATCH AGENT (Agent 3)     │
        │  Candidate-Job Matching      │
        │  - Parse job description     │
        │  - Chain-of-Thought scoring  │
        │  - Semantic similarity       │
        │  - Gap analysis              │
        │  - Bias detection (Bias Shield)|
        │  - SHAP explanations         │
        │  - Interview questions       │
        └──────────────────┬───────────┘
                           │
         OUTPUT: Match score, gaps, explanations
                           │
                           ▼
        ┌──────────────────────────────┐
        │ ENTITY RESOLVE AGENT (Agent 4)│
        │  Profile Verification        │
        │  - GitHub verification       │
        │  - LinkedIn verification     │
        │  - Stack Overflow lookup     │
        │  - Portfolio validation      │
        └──────────────────┬───────────┘
                           │
         OUTPUT: Verified candidate entity
                           │
                           ▼
        ┌──────────────────────────────┐
        │  DATABASE & SEARCH INDEX     │
        │  Store & index candidate     │
        │  Make searchable             │
        └──────────────────────────────┘
```

---

## 🔄 Data Flow: Parse → Normalize

**Parse Agent Output** (raw extracted skills):
```json
{
  "skills": ["Python", "python", "PYTHON", "js", "JavaScript", "React.js", "django"],
  "experience": [
    {
      "bullets": ["Built REST APIs using Python Django"],
      "skills_mentioned": ["Python", "REST API", "Django"]
    }
  ]
}
```

**↓ Passed to Normalize Agent ↓**

**Normalize Agent Input** receives this data and:
1. **Normalizes duplicates**: "Python" + "python" + "PYTHON" → ✅ "Python"
2. **Resolves abbreviations**: "js" → ✅ "JavaScript"
3. **Maps synonyms**: "React.js" → ✅ "React"
4. **Extracts skill duration**: From experience → "Python: 3 years"
5. **Categorizes**: "Python" → technical, "teamwork" → soft
6. **Estimates proficiency**: intermediate/advanced based on context
7. **Adds taxonomy tags**: Links to skill hierarchy

**Normalize Agent Output** (clean skills):
```json
{
  "skills": [
    {
      "name": "Python",
      "category": "technical",
      "proficiency": "intermediate",
      "estimated_years": 3,
      "confidence": 0.95
    },
    {
      "name": "JavaScript",
      "category": "technical",
      "proficiency": "intermediate",
      "estimated_years": 2,
      "confidence": 0.85
    },
    ...
  ]
}
```

---

## 🎯 Why This Agent Design?

| Aspect | Benefit |
|--------|---------|
| **Parallel Extraction** | 4 concurrent Claude calls = faster than sequential |
| **Layout Detection** | Handles messy, non-standard resume PDFs |
| **Multiple Extractors** | Different extraction for different document structures |
| **Language Detection** | Supports international resumes, enables multilingual support |
| **AI Detection** | Flags potentially fraudulent/generated resumes |
| **Confidence Scoring** | Tells downstream agents how trustworthy the data is |
| **Separation of Concerns** | Parse agent only extracts; doesn't normalize or match |

---

## ⚙️ Configuration & Dependencies

```python
# From app/core/config.py
settings.ANTHROPIC_MODEL        # Claude model (3.5-sonnet, etc.)
settings.ANTHROPIC_API_KEY      # API authentication
settings.RATE_LIMIT_PER_MINUTE  # API rate limiting

# External libraries used:
- pdfplumber              # PDF text extraction
- fitz (PyMuPDF)          # PDF fallback + OCR support
- python-docx             # .docx parsing
- langdetect              # Language detection
- anthropic               # Claude API async client
- asyncio                 # Parallel execution
```

---

## 🚀 How to Use Parse Agent

### **Direct Call** (in backend code):
```python
from app.agents.parse_agent import parse_agent
from app.core.pipeline_state import PipelineState

# Initialize state
state = PipelineState({
    "raw_file": file_bytes,      # PDF/DOCX/TXT bytes
    "file_type": "pdf",          # Detected file type
    "file_name": "resume.pdf",
    "job_id": "job-uuid"
})

# Run agent
state = await parse_agent(state)

# Access results
print(state["parsed"])              # Extracted data
print(state["parse_confidence"])    # Quality score 0.0–1.0
print(state["ai_content_probability"])  # Fraud detection
print(state["traces"])              # Execution metadata
```

### **Via API**:
```bash
curl -X POST http://localhost:8000/api/v1/parse \
  -H "X-API-Key: your-api-key" \
  -F "file=@resume.pdf"

# Returns: { "job_id": "...", "status": "processing" }

# Poll for results
curl http://localhost:8000/api/v1/jobs/{job_id} \
  -H "X-API-Key: your-api-key"
```

---

## 📊 Example Output

```json
{
  "parse_confidence": 0.85,
  "layout_type": "two_column",
  "resume_language": "en",
  "ai_content_probability": 0.05,
  "experience_months_total": 48,
  
  "parsed": {
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "phone": "+1-555-0123",
    "location": "San Francisco, CA",
    "linkedin_url": "linkedin.com/in/alice-johnson",
    "github_url": "github.com/alice-jp",
    
    "experience": [
      {
        "company": "TechCorp",
        "role": "Senior Software Engineer",
        "start": "2021-06",
        "end": "2024-01",
        "duration_months": 31,
        "bullets": [
          "Led team of 5 engineers on ML pipeline development",
          "Designed and deployed PyTorch model to production",
          "Improved inference latency by 40% through optimization"
        ],
        "skills_mentioned": ["Machine Learning", "PyTorch", "Leadership"]
      }
    ],
    
    "skills": [
      "Python", "PyTorch", "TensorFlow", "Docker", 
      "AWS", "Kubernetes", "SQL", "Leadership"
    ],
    
    "education": [
      {
        "institution": "Stanford University",
        "degree": "MS",
        "field": "Computer Science",
        "year": 2021,
        "gpa": 3.8
      }
    ],
    
    "certifications": [
      {
        "name": "AWS Certified Solutions Architect",
        "issuer": "Amazon",
        "year": 2023
      }
    ]
  },
  
  "traces": [
    {
      "agent": "parse_agent",
      "status": "success",
      "duration_ms": 2847,
      "quality_score": 0.85,
      "fields_extracted": 18,
      "retry_count": 0
    }
  ]
}
```

---

## ✅ Summary

| Question | Answer |
|----------|--------|
| **What is Parse Agent?** | Agent 1 that extracts resume data into structured JSON |
| **What makes it special?** | Parallel LLM extraction, layout detection, AI content detection |
| **What does it output?** | Structured resume with skills, experience, education, metadata |
| **Who uses its output?** | Normalize Agent (Agent 2) receives the parsed data |
| **Why separate from Normalize?** | Separation of concerns: extraction vs. standardization |
| **Is it the first step?** | ✅ Yes, it's always Agent 1 in the pipeline |
| **What's next?** | **Normalize Agent** standardizes skills using synonym mapping + embeddings |

