import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Zap, Shield, AlertTriangle, CheckCircle, ChevronDown, ChevronUp,
  BookOpen, MessageSquare, BarChart3
} from 'lucide-react'
import toast from 'react-hot-toast'
import { listCandidates, matchCandidate } from '../../api/client'

export default function MatchPage() {
  const [selectedCandidate, setSelectedCandidate] = useState('')
  const [jobDescription, setJobDescription] = useState('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [showReasoning, setShowReasoning] = useState(false)
  const [showInterviews, setShowInterviews] = useState(false)

  const { data: candidatesData } = useQuery({
    queryKey: ['candidates'],
    queryFn: () => listCandidates(),
  })

  const handleMatch = async () => {
    if (!selectedCandidate) return toast.error('Select a candidate')
    if (!jobDescription.trim()) return toast.error('Enter a job description')
    setLoading(true)
    setResult(null)
    try {
      const data = await matchCandidate(selectedCandidate, jobDescription)
      setResult(data)
    } catch (err: any) {
      toast.error(err.message)
    } finally {
      setLoading(false)
    }
  }

  const scoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.6) return 'text-amber-600'
    return 'text-red-600'
  }

  const scoreBar = (score: number) => {
    const pct = Math.round(score * 100)
    const color = score >= 0.8 ? 'bg-green-500' : score >= 0.6 ? 'bg-amber-500' : 'bg-red-500'
    return (
      <div className="flex items-center gap-3">
        <div className="flex-1 bg-gray-100 rounded-full h-2">
          <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
        </div>
        <span className={`text-sm font-bold ${scoreColor(score)}`}>{pct}%</span>
      </div>
    )
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Match Candidate</h1>
        <p className="text-gray-500 mt-1">
          Chain-of-Thought semantic matching with bias shield
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input panel */}
        <div className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Candidate</label>
            <select
              value={selectedCandidate}
              onChange={(e) => setSelectedCandidate(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            >
              <option value="">Select a parsed candidate...</option>
              {candidatesData?.candidates?.map((c: any) => (
                <option key={c.id} value={c.id}>
                  {c.name || 'Unknown'} · {c.skills_count} skills · {c.email || 'no email'}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Job Description</label>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the full job description here..."
              rows={14}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
            />
          </div>

          <button
            onClick={handleMatch}
            disabled={loading}
            className="w-full bg-indigo-600 text-white rounded-lg py-3 font-medium text-sm hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running pipeline...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Run Match
              </>
            )}
          </button>
        </div>

        {/* Result panel */}
        <div className="space-y-4">
          {!result && !loading && (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-8 text-center">
              <Zap className="w-10 h-10 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-400 text-sm">Match result will appear here</p>
            </div>
          )}

          {result && (
            <>
              {/* Scores */}
              <div className="bg-white border border-gray-200 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-semibold text-gray-900">Match scores</h2>
                  <span className={`text-2xl font-bold ${scoreColor(result.match_score)}`}>
                    {Math.round(result.match_score * 100)}%
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Overall match</span>
                    </div>
                    {scoreBar(result.match_score)}
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Blind score (bias-protected)</div>
                    {scoreBar(result.blind_score)}
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Semantic similarity</div>
                    {scoreBar(result.semantic_score)}
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Required skill coverage</div>
                    {scoreBar(result.required_skill_coverage)}
                  </div>
                </div>
              </div>

              {/* Bias flag */}
              {result.bias_flagged && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Bias detected</p>
                    <p className="text-xs text-red-600 mt-1">
                      Bias delta: {(result.bias_delta * 100).toFixed(1)}pp between blind and full score.
                      Human review recommended.
                    </p>
                  </div>
                </div>
              )}
              {!result.bias_flagged && (
                <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-center gap-3">
                  <Shield className="w-5 h-5 text-green-500" />
                  <p className="text-sm text-green-700">
                    Bias delta {(Math.abs(result.bias_delta) * 100).toFixed(1)}pp — within acceptable range
                  </p>
                </div>
              )}

              {/* Matched skills */}
              <div className="bg-white border border-gray-200 rounded-xl p-5">
                <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  Matched skills ({result.matched_skills?.length || 0})
                </h3>
                <div className="flex flex-wrap gap-2">
                  {result.matched_skills?.slice(0, 15).map((s: any, i: number) => (
                    <span
                      key={i}
                      className="px-2 py-1 bg-green-50 text-green-700 text-xs rounded-full border border-green-200"
                      title={`Similarity: ${Math.round((s.similarity || 0) * 100)}%`}
                    >
                      {s.candidate_skill} ({s.proficiency})
                    </span>
                  ))}
                </div>
              </div>

              {/* Skill gaps */}
              {result.skill_gaps?.length > 0 && (
                <div className="bg-white border border-gray-200 rounded-xl p-5">
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    Skill gaps ({result.skill_gaps.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {result.skill_gaps.map((s: string, i: number) => (
                      <span
                        key={i}
                        className="px-2 py-1 bg-amber-50 text-amber-700 text-xs rounded-full border border-amber-200"
                      >
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* SHAP */}
              <div className="bg-white border border-gray-200 rounded-xl p-5">
                <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-indigo-500" />
                  Score explanation (SHAP)
                </h3>
                <div className="space-y-2">
                  {Object.entries(result.shap_values || {})
                    .filter(([k]) => !['top_skill_matches'].includes(k))
                    .map(([key, val]: [string, any]) => {
                      const v = typeof val === 'number' ? val : 0
                      return (
                        <div key={key} className="flex items-center justify-between text-xs">
                          <span className="text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
                          <span className={`font-mono font-medium ${v >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {v >= 0 ? '+' : ''}{v.toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* CoT Reasoning toggle */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <button
                  onClick={() => setShowReasoning(!showReasoning)}
                  className="w-full flex items-center justify-between p-4 text-sm font-semibold text-gray-900 hover:bg-gray-50"
                >
                  <span className="flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-indigo-500" />
                    Full CoT reasoning
                  </span>
                  {showReasoning ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {showReasoning && (
                  <div className="px-4 pb-4">
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap bg-gray-50 rounded-lg p-3 max-h-64 overflow-y-auto">
                      {result.cot_reasoning}
                    </pre>
                  </div>
                )}
              </div>

              {/* Interview questions toggle */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <button
                  onClick={() => setShowInterviews(!showInterviews)}
                  className="w-full flex items-center justify-between p-4 text-sm font-semibold text-gray-900 hover:bg-gray-50"
                >
                  <span className="flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-indigo-500" />
                    Interview questions ({
                      (result.interview_questions?.technical?.length || 0) +
                      (result.interview_questions?.gap_probe?.length || 0) +
                      (result.interview_questions?.culture ? 1 : 0)
                    })
                  </span>
                  {showInterviews ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {showInterviews && result.interview_questions && (
                  <div className="px-4 pb-4 space-y-3">
                    {result.interview_questions.technical?.map((q: any, i: number) => (
                      <div key={i} className="bg-indigo-50 rounded-lg p-3">
                        <p className="text-xs font-semibold text-indigo-700 mb-1">Technical {i + 1}</p>
                        <p className="text-sm text-gray-800">{q.question}</p>
                        {q.what_to_listen_for && (
                          <p className="text-xs text-gray-500 mt-1 italic">
                            Listen for: {q.what_to_listen_for}
                          </p>
                        )}
                      </div>
                    ))}
                    {result.interview_questions.gap_probe?.map((q: any, i: number) => (
                      <div key={i} className="bg-amber-50 rounded-lg p-3">
                        <p className="text-xs font-semibold text-amber-700 mb-1">Gap probe {i + 1}</p>
                        <p className="text-sm text-gray-800">{q.question}</p>
                      </div>
                    ))}
                    {result.interview_questions.culture && (
                      <div className="bg-green-50 rounded-lg p-3">
                        <p className="text-xs font-semibold text-green-700 mb-1">Culture & growth</p>
                        <p className="text-sm text-gray-800">{result.interview_questions.culture.question}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Summary */}
              <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-4">
                <p className="text-sm font-semibold text-indigo-800 mb-1">Recommendation: {result.recommendation}</p>
                <p className="text-sm text-indigo-700">{result.summary}</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
