import axios, { AxiosResponse, AxiosError } from 'axios'

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

// ── Parse ──────────────────────────────────────────────────────

export const parseResume = async (file: File) => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/parse', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const parseBatch = async (files: File[]) => {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  const { data } = await api.post('/parse/batch', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getJobStatus = async (jobId: string) => {
  const { data } = await api.get(`/jobs/${jobId}`)
  return data
}

export const getJobTrace = async (jobId: string) => {
  const { data } = await api.get(`/jobs/${jobId}/trace`)
  return data
}

// ── Candidates ─────────────────────────────────────────────────

export const listCandidates = async (page = 1, search = '') => {
  const { data } = await api.get('/candidates', { params: { page, search } })
  return data
}

export const getCandidate = async (id: string) => {
  const { data } = await api.get(`/candidates/${id}`)
  return data
}

export const getCandidateSkills = async (id: string) => {
  const { data } = await api.get(`/candidates/${id}/skills`)
  return data
}

export const getCandidateMatches = async (id: string) => {
  const { data } = await api.get(`/candidates/${id}/matches`)
  return data
}

// ── Match ──────────────────────────────────────────────────────

export const matchCandidate = async (candidateId: string, jobDescription: string) => {
  const { data } = await api.post('/match', {
    candidate_id: candidateId,
    job_description: jobDescription,
    save_result: true,
  })
  return data
}

export const batchMatch = async (candidateIds: string[], jobDescription: string, topN = 10) => {
  const { data } = await api.post('/match/batch', {
    candidate_ids: candidateIds,
    job_description: jobDescription,
    top_n: topN,
  })
  return data
}

export const runBiasAudit = async (jobDescription: string) => {
  const { data } = await api.post('/match/bias-audit', null, {
    params: { job_description: jobDescription },
  })
  return data
}

// ── Taxonomy ───────────────────────────────────────────────────

export const browseTaxonomy = async (category?: string, page = 1) => {
  const { data } = await api.get('/skills/taxonomy', { params: { category, page } })
  return data
}

export const searchTaxonomy = async (q: string) => {
  const { data } = await api.get('/skills/taxonomy/search', { params: { q } })
  return data
}

export const getEmergingSkills = async () => {
  const { data } = await api.get('/skills/taxonomy/emerging')
  return data
}

// ── Admin ──────────────────────────────────────────────────────

export const getSystemStats = async () => {
  const { data } = await api.get('/admin/stats')
  return data
}

export const getHITLQueue = async () => {
  const { data } = await api.get('/admin/hitl-queue')
  return data
}

export const resolveHITLItem = async (itemId: string, decision: string, notes = '') => {
  const { data } = await api.post(`/admin/hitl-queue/${itemId}/resolve`, null, {
    params: { decision, notes },
  })
  return data
}
