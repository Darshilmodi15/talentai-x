import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { BarChart3, Shield, AlertTriangle, CheckCircle, Eye } from 'lucide-react'
import toast from 'react-hot-toast'
import { listCandidates, batchMatch, runBiasAudit } from '../../api/client'

export default function BatchMatchPage() {
  const navigate = useNavigate()
  const [jobDescription, setJobDescription] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [result, setResult] = useState<any>(null)
  const [auditResult, setAuditResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [auditing, setAuditing] = useState(false)

  const { data: candidatesData } = useQuery({
    queryKey: ['candidates-all'],
    queryFn: () => listCandidates(1, ''),
  })

  const toggleCandidate = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  const handleBatchMatch = async () => {
    if (!jobDescription.trim()) return toast.error('Enter a job description')
    if (selectedIds.length === 0) return toast.error('Select at least one candidate')
    setLoading(true)
    setResult(null)
    try {
      const data = await batchMatch(selectedIds, jobDescription, 20)
      setResult(data)
    } catch (err: any) {
      toast.error(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleBiasAudit = async () => {
    if (!jobDescription.trim()) return toast.error('Enter a job description first')
    setAuditing(true)
    try {
      const data = await runBiasAudit(jobDescription)
      setAuditResult(data)
    } catch (err: any) {
      toast.error(err.message)
    } finally {
      setAuditing(false)
    }
  }

  const scoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-amber-600'
    return 'text-red-600'
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Batch Match</h1>
        <p className="text-gray-500 mt-1">Rank multiple candidates against a single job description</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: JD + candidate selection */}
        <div className="lg:col-span-2 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Job Description</label>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the full job description..."
              rows={10}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 resize-none"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Candidates ({selectedIds.length} selected)
              </label>
              <button
                onClick={() => setSelectedIds(
                  selectedIds.length === candidatesData?.candidates?.length
                    ? []
                    : candidatesData?.candidates?.map((c: any) => c.id) || []
                )}
                className="text-xs text-indigo-600 hover:text-indigo-800"
              >
                {selectedIds.length === candidatesData?.candidates?.length ? 'Deselect all' : 'Select all'}
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg max-h-56 overflow-y-auto">
              {candidatesData?.candidates?.map((c: any) => (
                <label
                  key={c.id}
                  className="flex items-center gap-3 px-3 py-2 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-0"
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(c.id)}
                    onChange={() => toggleCandidate(c.id)}
                    className="rounded text-indigo-600"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900 truncate">{c.name || 'Unknown'}</p>
                    <p className="text-xs text-gray-400">{c.skills_count} skills</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleBatchMatch}
            disabled={loading}
            className="w-full bg-indigo-600 text-white rounded-lg py-2.5 text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />Running...</> : <><BarChart3 className="w-4 h-4" />Rank candidates</>}
          </button>

          <button
            onClick={handleBiasAudit}
            disabled={auditing}
            className="w-full border border-indigo-300 text-indigo-600 rounded-lg py-2.5 text-sm font-medium hover:bg-indigo-50 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {auditing ? <><div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />Auditing...</> : <><Shield className="w-4 h-4" />Run bias audit</>}
          </button>
        </div>

        {/* Right: results */}
        <div className="lg:col-span-3 space-y-4">
          {/* Bias audit result */}
          {auditResult && (
            <div className={`border rounded-xl p-5 ${auditResult.status === 'FAIR' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                {auditResult.status === 'FAIR'
                  ? <CheckCircle className="w-5 h-5 text-green-600" />
                  : <AlertTriangle className="w-5 h-5 text-red-600" />}
                <span className={`font-semibold text-sm ${auditResult.status === 'FAIR' ? 'text-green-800' : 'text-red-800'}`}>
                  Bias Audit: {auditResult.status}
                </span>
                <span className="text-xs text-gray-500 ml-auto">Score variance: {auditResult.bias_score}</span>
              </div>
              <p className="text-sm text-gray-600">{auditResult.interpretation}</p>
              <p className="text-xs text-gray-400 mt-2">Compliant with: {auditResult.compliant_with?.join(', ')}</p>
            </div>
          )}

          {/* Ranked shortlist */}
          {result && (
            <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100 bg-gray-50 flex items-center justify-between">
                <h2 className="font-semibold text-gray-900 text-sm">
                  Ranked shortlist — {result.total_candidates} candidates
                </h2>
                <span className="text-xs text-gray-500">Ranked by blind score</span>
              </div>
              <div className="divide-y divide-gray-100">
                {result.shortlist?.map((r: any, i: number) => (
                  <div key={r.candidate_id} className="px-5 py-3 flex items-center gap-4">
                    <span className="text-lg font-bold text-gray-300 w-7">#{i + 1}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium text-gray-900 truncate">{r.name}</p>
                        {r.bias_flagged && (
                          <span className="px-1.5 py-0.5 bg-red-100 text-red-600 text-xs rounded-full">bias flag</span>
                        )}
                      </div>
                      <div className="flex items-center gap-4 mt-1">
                        <span className={`text-xs font-semibold ${scoreColor(r.blind_score)}`}>
                          {Math.round(r.blind_score * 100)}% blind
                        </span>
                        <span className="text-xs text-gray-400">
                          {Math.round(r.required_skill_coverage * 100)}% skill coverage
                        </span>
                        {r.skill_gaps?.length > 0 && (
                          <span className="text-xs text-amber-600">{r.skill_gaps.length} gaps</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        r.recommendation === 'STRONG YES' ? 'bg-green-100 text-green-700' :
                        r.recommendation === 'YES' ? 'bg-blue-100 text-blue-700' :
                        r.recommendation === 'NO' ? 'bg-red-100 text-red-700' :
                        'bg-amber-100 text-amber-700'
                      }`}>
                        {r.recommendation}
                      </span>
                      <button
                        onClick={() => navigate(`/candidates/${r.candidate_id}`)}
                        className="p-1 hover:bg-gray-100 rounded"
                      >
                        <Eye className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && !auditResult && (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-12 text-center">
              <BarChart3 className="w-10 h-10 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">Ranked shortlist will appear here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
