import axios, { AxiosResponse, AxiosError } from 'axios'
import toast from 'react-hot-toast'

const API_BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '/_/backend')
const API_KEY = import.meta.env.VITE_API_KEY || 'dev_key_change_in_production'

export const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  timeout: 120_000,
  headers: {
    'X-API-Key': API_KEY,
  },
})

const normalizeApiError = (error: AxiosError<any>) => {
  if (error.code === 'ECONNABORTED') return 'Request timed out. The AI service may be busy; please try again.'
  if (!error.response) return 'Unable to connect to server. Make sure the backend is running.'

  const data = error.response.data
  if (typeof data?.message === 'string') return data.message
  if (typeof data?.detail === 'string') return data.detail
  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((d: any) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((x: string) => x !== 'body').join('.') : ''
        return `${field ? `${field}: ` : ''}${d.msg || 'Invalid value'}`
      })
      .join('; ')
  }
  if (typeof data === 'string') return data
  if (error.response.status >= 500) return 'Server error. Please try again; if it repeats, check backend logs.'
  return `Request failed (${error.response.status})`
}

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError<any>) => Promise.reject(new Error(normalizeApiError(error)))
)

// ── Mock Data ──────────────────────────────────────────────────

const MOCK_CANDIDATES = [
  {
    id: "cand-1",
    name: "Alexander Wright",
    email: "alexander.wright@devmail.net",
    phone: "+1 (555) 234-5678",
    location: "San Francisco, CA",
    parse_confidence: 0.94,
    ai_content_probability: 0.12,
    experience_months_total: 72,
    experience_months: 72,
    skills_count: 14,
    resume_language: "en",
    summary: "Senior Software Engineer with 6+ years of experience specializing in building highly scalable web applications. Strong expertise in React, TypeScript, Python, and cloud infrastructure.",
    platform_profiles: {
      github: {
        handle: "alexwright-dev",
        url: "https://github.com/Darshilmodi15"
      }
    },
    experience: [
      {
        role: "Senior Frontend Engineer",
        company: "ScaleApp Technologies",
        start: "2022-03",
        end: "Present"
      },
      {
        role: "Software Engineer",
        company: "WebFlow Solutions",
        start: "2020-01",
        end: "2022-02"
      }
    ],
    education: [
      { degree: "B.S.", field: "Computer Science", institution: "Stanford University", year: "2020" }
    ],
    certifications: [
      { name: "AWS Certified Solutions Architect", year: "2022" }
    ]
  },
  {
    id: "cand-2",
    name: "Sarah Chen",
    email: "sarah.chen@techintel.io",
    phone: "+1 (555) 876-5432",
    location: "Seattle, WA",
    parse_confidence: 0.97,
    ai_content_probability: 0.05,
    experience_months_total: 48,
    experience_months: 48,
    skills_count: 18,
    resume_language: "en",
    summary: "AI Research Scientist with a strong background in Machine Learning, Natural Language Processing, and LLM orchestration. Experienced in developing custom agents using PyTorch and LangChain.",
    platform_profiles: {
      github: {
        handle: "schen-ai",
        url: "https://github.com/Darshilmodi15"
      }
    },
    experience: [
      {
        role: "AI Developer",
        company: "NeuralLabs Systems",
        start: "2022-08",
        end: "Present"
      },
      {
        role: "Machine Learning Intern",
        company: "Cognitive Computing Corp",
        start: "2021-06",
        end: "2022-05"
      }
    ],
    education: [
      { degree: "M.S.", field: "Artificial Intelligence", institution: "University of Washington", year: "2022" }
    ],
    certifications: [
      { name: "TensorFlow Developer Certificate", year: "2021" }
    ]
  },
  {
    id: "cand-3",
    name: "Marcus Thompson",
    email: "m.thompson@cloudscale.org",
    phone: "+1 (555) 345-6789",
    location: "Austin, TX",
    parse_confidence: 0.89,
    ai_content_probability: 0.22,
    experience_months_total: 96,
    experience_months: 96,
    skills_count: 11,
    resume_language: "en",
    summary: "DevOps & Cloud Architect with 8 years of experience building and automating high-availability infrastructure. Expert in AWS, Kubernetes, Terraform, and CI/CD pipelines.",
    platform_profiles: {
      github: {
        handle: "mthompson-cloud",
        url: "https://github.com/Darshilmodi15"
      }
    },
    experience: [
      {
        role: "Lead DevOps Architect",
        company: "CloudScale Systems",
        start: "2021-02",
        end: "Present"
      },
      {
        role: "Systems Administrator",
        company: "Global Infrastructure Group",
        start: "2018-05",
        end: "2021-01"
      }
    ],
    education: [
      { degree: "B.S.", field: "Computer Engineering", institution: "University of Texas at Austin", year: "2018" }
    ],
    certifications: [
      { name: "Certified Kubernetes Administrator (CKA)", year: "2020" }
    ]
  }
]

const MOCK_SKILLS: Record<string, any> = {
  "cand-1": {
    total_skills: 14,
    by_category: { Frontend: 6, Backend: 4, DevOps: 4 },
    skills: [
      { canonical: "React", proficiency: "expert", source: "resume", years: 5, inferred: false },
      { canonical: "TypeScript", proficiency: "expert", source: "resume", years: 4, inferred: false },
      { canonical: "Node.js", proficiency: "advanced", source: "resume", years: 4, inferred: false },
      { canonical: "Python", proficiency: "advanced", source: "resume", years: 3, inferred: false },
      { canonical: "Docker", proficiency: "intermediate", source: "resume", years: 2, inferred: true },
      { canonical: "TailwindCSS", proficiency: "expert", source: "resume", years: 4, inferred: false },
      { canonical: "PostgreSQL", proficiency: "intermediate", source: "resume", years: 3, inferred: false },
      { canonical: "GraphQL", proficiency: "advanced", source: "resume", years: 2, inferred: false }
    ]
  },
  "cand-2": {
    total_skills: 18,
    by_category: { "AI/ML": 8, Backend: 6, Math: 4 },
    skills: [
      { canonical: "PyTorch", proficiency: "expert", source: "resume", years: 4, inferred: false },
      { canonical: "Python", proficiency: "expert", source: "resume", years: 5, inferred: false },
      { canonical: "LangChain", proficiency: "advanced", source: "resume", years: 2, inferred: false },
      { canonical: "Transformers", proficiency: "expert", source: "resume", years: 3, inferred: false },
      { canonical: "FastAPI", proficiency: "advanced", source: "resume", years: 3, inferred: false },
      { canonical: "Pandas", proficiency: "expert", source: "resume", years: 4, inferred: false },
      { canonical: "Scikit-Learn", proficiency: "advanced", source: "resume", years: 4, inferred: false }
    ]
  },
  "cand-3": {
    total_skills: 11,
    by_category: { DevOps: 6, Cloud: 5 },
    skills: [
      { canonical: "Kubernetes", proficiency: "expert", source: "resume", years: 5, inferred: false },
      { canonical: "Docker", proficiency: "expert", source: "resume", years: 6, inferred: false },
      { canonical: "AWS", proficiency: "expert", source: "resume", years: 6, inferred: false },
      { canonical: "Terraform", proficiency: "advanced", source: "resume", years: 4, inferred: false },
      { canonical: "GitHub Actions", proficiency: "advanced", source: "resume", years: 4, inferred: false },
      { canonical: "Linux", proficiency: "expert", source: "resume", years: 8, inferred: false },
      { canonical: "Prometheus", proficiency: "intermediate", source: "resume", years: 3, inferred: true }
    ]
  }
}

// ── Parse ──────────────────────────────────────────────────────

export const parseResume = async (file: File): Promise<any> => {
  throw new Error("Project is under review: Resume uploading is temporarily disabled.")
}

export const parseBatch = async (files: File[]): Promise<any> => {
  throw new Error("Project is under review: Batch resume uploading is temporarily disabled.")
}

export const getJobStatus = async (jobId: string): Promise<any> => {
  return { status: 'failed', error_message: 'Project is under review' }
}

export const getJobTrace = async (jobId: string): Promise<any> => {
  return []
}

// ── Candidates ─────────────────────────────────────────────────

export const listCandidates = async (page = 1, search = ''): Promise<any> => {
  toast.success("Loaded demo profiles (API offline / under review)", {
    id: "demo-candidates-toast",
    duration: 4000
  })

  let filtered = [...MOCK_CANDIDATES]
  if (search) {
    const q = search.toLowerCase()
    filtered = filtered.filter(c => c.name.toLowerCase().includes(q) || c.email.toLowerCase().includes(q))
  }

  return {
    candidates: filtered,
    total: filtered.length
  }
}

export const getCandidate = async (id: string): Promise<any> => {
  const cand = MOCK_CANDIDATES.find(c => c.id === id) || MOCK_CANDIDATES[0]
  return cand
}

export const getCandidateSkills = async (id: string): Promise<any> => {
  return MOCK_SKILLS[id] || MOCK_SKILLS["cand-1"]
}

export const getCandidateMatches = async (id: string): Promise<any> => {
  return []
}

// ── Match ──────────────────────────────────────────────────────

export const matchCandidate = async (candidateId: string, jobDescription: string): Promise<any> => {
  throw new Error("Project is under review: Candidate matching is temporarily disabled.")
}

export const batchMatch = async (candidateIds: string[], jobDescription: string, topN = 10): Promise<any> => {
  throw new Error("Project is under review: Batch matching is temporarily disabled.")
}

export const runBiasAudit = async (jobDescription: string): Promise<any> => {
  throw new Error("Project is under review: Bias audit is temporarily disabled.")
}

// ── Taxonomy ───────────────────────────────────────────────────

export const browseTaxonomy = async (category?: string, page = 1): Promise<any> => {
  toast.success("Loaded demo skill taxonomy (API offline / under review)", {
    id: "demo-taxonomy-toast",
    duration: 4000
  })

  const skills = [
    { canonical_name: "React", category: "technical", frequency: 142 },
    { canonical_name: "Python", category: "technical", frequency: 118 },
    { canonical_name: "TypeScript", category: "technical", frequency: 95 },
    { canonical_name: "Docker", category: "technical", frequency: 84 },
    { canonical_name: "Kubernetes", category: "technical", frequency: 53 },
    { canonical_name: "PostgreSQL", category: "technical", frequency: 72 },
    { canonical_name: "PyTorch", category: "technical", frequency: 41 }
  ]

  return {
    skills,
    total: skills.length
  }
}

export const searchTaxonomy = async (q: string): Promise<any> => {
  return {
    results: [],
    total: 0
  }
}

export const getEmergingSkills = async (): Promise<any> => {
  return {
    count: 3,
    emerging_skills: [
      { id: "1", raw_skill: "LangChain", occurrences: 12, first_seen: "2026-01-01", status: "pending" },
      { id: "2", raw_skill: "Vector Databases", occurrences: 8, first_seen: "2026-02-01", status: "pending" },
      { id: "3", raw_skill: "Next.js 14", occurrences: 5, first_seen: "2026-03-01", status: "pending" }
    ]
  }
}

// ── Admin ──────────────────────────────────────────────────────

export const getSystemStats = async (): Promise<any> => {
  return {
    total_candidates_parsed: 3,
    total_parse_jobs: 14,
    total_matches_run: 84,
    hitl_queue_pending: 1,
    bias_flags_raised: 2
  }
}

export const getHITLQueue = async (): Promise<any> => {
  return {
    pending_count: 1,
    items: [
      {
        id: "hitl-1",
        priority: "high",
        expires_at: new Date(Date.now() + 86400000 * 3).toISOString(),
        match_result_id: "match-12345",
        trigger_reason: "High gender bias flag detected on Blind Scoring audit"
      }
    ]
  }
}

export const resolveHITLItem = async (itemId: string, decision: string, notes = ''): Promise<any> => {
  throw new Error("Project is under review: Admin decisions are temporarily disabled.")
}
