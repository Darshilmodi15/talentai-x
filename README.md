# TalentAI-X: Multi-Agent AI Talent Intelligence System

TalentAI-X is a multi-agent AI system that doesn’t just read resumes — it verifies, analyzes, and explains real candidate capabilities using evidence from multiple sources.

PROBLEM

Current hiring systems (ATS):
rely on keyword matching
fail to understand context
cannot verify real skills
miss qualified candidates
allow resume exaggeration

 Result:
 Poor hiring decisions
 Wasted recruiter time
 
 SOLUTION
TalentAI-X builds an intelligent, explainable hiring system that:
 Parses resumes
 Normalizes skills
 Matches candidates semantically
 Verifies claims using real data
 Provides explainable insights
 
 CORE SYSTEM (MULTI-AGENT ARCHITECTURE)
 1. Resume Parsing Agent
Extracts structured data from PDF/DOCX
Handles complex layouts
 2. Skill Normalization Agent
Maps skills to taxonomy
Handles synonyms (React.js → React)
Infers higher-level skills
 3. Entity Resolution Agent 
Matches candidate across:
Resume
GitHub
LinkedIn
 Even with different usernames
 4. Claim Verification Agent 
Compares resume claims vs GitHub evidence
Detects exaggeration
 5. Semantic Matching Agent
Uses embeddings (not keywords)
Matches candidate to job description

KEY INNOVATIONS (THIS WINS HACKATHON)
 1. Evidence-Based Hiring
 Not just “what candidate says”
 But “what candidate has actually done”
 2. Cross-Platform Intelligence
Resume + GitHub + LinkedIn
 3. Chain-of-Thought Reasoning
Step-by-step explanation
No black-box scoring
 4. Skill Gap Analysis
Shows missing skills
Suggests improvement
 5. Growth Trajectory (Advanced Thinking)
Evaluates learning trend

 SYSTEM FLOW
Resume Upload
   ↓
Parsing Agent
   ↓
Skill Normalization
   ↓
Entity Resolution
   ↓
GitHub Data Fetch
   ↓
Claim Verification
   ↓
Semantic Matching
   ↓
Final Report

TECH STACK
Backend:
Python + FastAPI
AI:
LLMs (Gemini / Claude)
Sentence Transformers
Data:
PostgreSQL
ChromaDB (vector DB)
Parsing:
pdfplumber
python-docx
Deployment:
Localhost + optional Vercel
 
 DATASET
500+ resumes (real + AI-generated)
5000+ skill taxonomy
100+ job descriptions
Multiple industries

 OUTPUT
Candidate Report:
Match Score (%)
Skill Match + Missing Skills
Verified Projects
Claim Strength
AI Reasoning
 
DESIGN PRINCIPLES
 Explainable AI
 Modular architecture
 Privacy-first input
 Model-agnostic system

 IMPACT
For recruiters:
faster hiring
better decisions
reduced bias
For candidates:
fair evaluation
clear feedback
skill improvement insights

# TalentAI-X transforms hiring from keyword-based filtering to evidence-driven intelligence.
