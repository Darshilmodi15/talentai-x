import axios, { AxiosResponse, AxiosError, InternalAxiosRequestConfig } from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''
const API_KEY = import.meta.env.VITE_API_KEY || ''

/**
 * Retrieve the current Supabase or custom JWT access token.
 */
export const getAuthToken = (): string | null => {
  if (typeof window === 'undefined') return null

  // 1. Check explicit local token storage
  const explicit = localStorage.getItem('talentai_auth_token')
  if (explicit) return explicit

  // 2. Discover Supabase auth token stored in localStorage (sb-<project-ref>-auth-token)
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key && key.startsWith('sb-') && key.endsWith('-auth-token')) {
      try {
        const parsed = JSON.parse(localStorage.getItem(key) || '{}')
        if (parsed?.access_token) return parsed.access_token
      } catch {
        // ignore JSON parse errors
      }
    }
  }

  return null
}

export const setAuthToken = (token: string | null): void => {
  if (typeof window === 'undefined') return
  if (token) {
    localStorage.setItem('talentai_auth_token', token)
  } else {
    localStorage.removeItem('talentai_auth_token')
  }
}

/**
 * Check whether the current session is running in local UI demo mode.
 */
export const isDemoMode = (): boolean => {
  if (typeof window === 'undefined') return false
  return localStorage.getItem('talentai_demo_mode') === 'true'
}

export const setDemoMode = (enabled: boolean): void => {
  if (typeof window === 'undefined') return
  if (enabled) {
    localStorage.setItem('talentai_demo_mode', 'true')
  } else {
    localStorage.removeItem('talentai_demo_mode')
  }
}

export const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  timeout: 120_000,
})

// Attach Bearer token or API key to outgoing requests
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getAuthToken()
  if (token) {
    config.headers.set('Authorization', `Bearer ${token}`)
  } else if (API_KEY) {
    config.headers.set('X-API-Key', API_KEY)
  }
  return config
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
  if (error.response.status === 401) return 'Unauthorized. Please sign in or provide valid credentials.'
  if (error.response.status === 403) return 'Forbidden. Your account does not have permission for this action.'
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
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/parse', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const parseBatch = async (files: File[]): Promise<any> => {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  const { data } = await api.post('/parse/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getJobStatus = async (jobId: string): Promise<any> => {
  const { data } = await api.get(`/jobs/${jobId}`)
  return data
}

export const getJobTrace = async (jobId: string): Promise<any> => {
  const { data } = await api.get(`/jobs/${jobId}/trace`)
  return data
}

// ── Candidates ─────────────────────────────────────────────────

export const listCandidates = async (page = 1, search = ''): Promise<any> => {
  if (isDemoMode()) {
    const filtered = search
      ? MOCK_CANDIDATES.filter(
          (c) =>
            c.name.toLowerCase().includes(search.toLowerCase()) ||
            c.email.toLowerCase().includes(search.toLowerCase())
        )
      : MOCK_CANDIDATES
    return {
      total: filtered.length,
      page,
      page_size: 20,
      candidates: filtered,
    }
  }
  const { data } = await api.get('/candidates', { params: { page, search: search || undefined } })
  return data
}

export const getCandidate = async (id: string): Promise<any> => {
  if (isDemoMode()) {
    return MOCK_CANDIDATES.find((c) => c.id === id) || MOCK_CANDIDATES[0]
  }
  const { data } = await api.get(`/candidates/${id}`)
  return data
}

export const getCandidateSkills = async (id: string): Promise<any> => {
  if (isDemoMode()) {
    return MOCK_SKILLS[id] || MOCK_SKILLS['cand-1']
  }
  const { data } = await api.get(`/candidates/${id}/skills`)
  return data
}

export const getCandidateMatches = async (id: string): Promise<any> => {
  if (isDemoMode()) {
    return {
      candidate_id: id,
      total_matches: 0,
      matches: [],
    }
  }
  const { data } = await api.get(`/candidates/${id}/matches`)
  return data
}

// ── Match ──────────────────────────────────────────────────────

export const matchCandidate = async (candidateId: string, jobDescription: string): Promise<any> => {
  if (isDemoMode()) {
    return {
      match_score: 0.88,
      blind_score: 0.91,
      bias_delta: -0.03,
      bias_flagged: false,
      required_skill_coverage: 0.85,
      skill_gaps: ['Docker', 'Kubernetes'],
      upskilling_suggestions: { Docker: 'Complete Docker Essentials', Kubernetes: 'CKA certification track' },
      interview_questions: { technical: ['Explain React fiber architecture and reconciliation.'] },
      cot_reasoning: 'Demo evaluation: candidate demonstrates strong foundation with minor skill gap in container orchestration.',
      summary: 'Well-rounded senior frontend candidate.',
      shap_values: { skills: 0.55, experience: 0.3, education: 0.15 },
      hitl_required: false,
    }
  }
  const { data } = await api.post('/match', {
    candidate_id: candidateId,
    job_description: jobDescription,
  })
  return data
}

export const batchMatch = async (candidateIds: string[], jobDescription: string, topN = 10): Promise<any> => {
  if (isDemoMode()) {
    return {
      total_candidates: candidateIds.length,
      failed_candidates: 0,
      failures: [],
      job_description_preview: jobDescription.slice(0, 100) + '...',
      ranking_method: 'blind_score',
      shortlist: MOCK_CANDIDATES.map((c) => ({
        candidate_id: c.id,
        name: c.name,
        match_score: 0.88,
        blind_score: 0.9,
        bias_flagged: false,
        required_skill_coverage: 0.85,
        skill_gaps: ['Kubernetes'],
        recommendation: 'Strong candidate for interview',
      })).slice(0, topN),
    }
  }
  const { data } = await api.post('/match/batch', {
    candidate_ids: candidateIds,
    job_description: jobDescription,
    top_n: topN,
  })
  return data
}

export const runBiasAudit = async (jobDescription: string): Promise<any> => {
  const { data } = await api.post('/match/bias-audit', null, {
    params: { job_description: jobDescription },
  })
  return data
}

// ── Taxonomy ───────────────────────────────────────────────────

export const browseTaxonomy = async (category?: string, page = 1): Promise<any> => {
  const { data } = await api.get('/skills/taxonomy', {
    params: { category, page },
  })
  return data
}

export const searchTaxonomy = async (q: string): Promise<any> => {
  const { data } = await api.get('/skills/taxonomy/search', { params: { q } })
  return data
}

export const getEmergingSkills = async (): Promise<any> => {
  const { data } = await api.get('/skills/taxonomy/emerging')
  return data
}

// ── Admin ──────────────────────────────────────────────────────

export const getSystemStats = async (): Promise<any> => {
  const { data } = await api.get('/admin/stats')
  return data
}

export const getHITLQueue = async (): Promise<any> => {
  const { data } = await api.get('/admin/hitl-queue')
  return data
}

export const resolveHITLItem = async (itemId: string, decision: string, notes = ''): Promise<any> => {
  const { data } = await api.post(`/admin/hitl-queue/${itemId}/resolve`, null, {
    params: { decision, notes },
  })
  return data
}
