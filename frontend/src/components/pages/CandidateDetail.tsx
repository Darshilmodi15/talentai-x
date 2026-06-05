import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeft, User, Mail, Phone, MapPin, Globe, Github, Linkedin,
  Star, Clock, Shield, AlertTriangle, Zap, ExternalLink
} from 'lucide-react'
import { getCandidate, getCandidateSkills } from '../../api/client'

const PROF_COLORS: Record<string, string> = {
  expert:        'bg-purple-100 text-purple-700 border-purple-200',
  advanced:      'bg-blue-100 text-blue-700 border-blue-200',
  intermediate:  'bg-green-100 text-green-700 border-green-200',
  beginner:      'bg-gray-100 text-gray-600 border-gray-200',
  inferred:      'bg-amber-50 text-amber-600 border-amber-200',
  verified_github: 'bg-indigo-100 text-indigo-700 border-indigo-200',
}

export default function CandidateDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: candidate, isLoading } = useQuery({
    queryKey: ['candidate', id],
    queryFn: () => getCandidate(id!),
    enabled: !!id,
  })

  const { data: skillsData } = useQuery({
    queryKey: ['candidate-skills', id],
    queryFn: () => getCandidateSkills(id!),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-32 bg-gray-100 rounded-xl animate-pulse" />
        ))}
      </div>
    )
  }

  if (!candidate) {
    return (
      <div className="text-center py-16">
        <p className="text-gray-400">Candidate not found</p>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <button
          onClick={() => navigate('/candidates')}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-gray-500" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-900">{candidate.name || 'Unknown Candidate'}</h1>
          <p className="text-gray-500 text-sm mt-0.5">
            Parse confidence: {Math.round((candidate.parse_confidence || 0) * 100)}%
            {candidate.ai_content_probability > 0.7 && (
              <span className="ml-3 text-amber-600">⚠ High AI-content probability</span>
            )}
          </p>
        </div>
        <button
          onClick={() => navigate(`/match?candidate=${id}`)}
          className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700"
        >
          <Zap className="w-4 h-4" />
          Match to job
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: contact + meta */}
        <div className="space-y-4">
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <div className="flex items-center justify-center mb-4">
              <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center">
                <User className="w-8 h-8 text-indigo-600" />
              </div>
            </div>
            <div className="space-y-2.5">
              {candidate.email && (
                <div className="flex items-center gap-2 text-sm">
                  <Mail className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="text-gray-700 truncate">{candidate.email}</span>
                </div>
              )}
              {candidate.phone && (
                <div className="flex items-center gap-2 text-sm">
                  <Phone className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="text-gray-700">{candidate.phone}</span>
                </div>
              )}
              {candidate.location && (
                <div className="flex items-center gap-2 text-sm">
                  <MapPin className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="text-gray-700">{candidate.location}</span>
                </div>
              )}
              {candidate.platform_profiles?.github && (
                <div className="flex items-center gap-2 text-sm">
                  <Github className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <a
                    href={candidate.platform_profiles.github.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-600 hover:underline flex items-center gap-1"
                  >
                    {candidate.platform_profiles.github.handle}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                  <span className="text-xs text-green-600 bg-green-50 px-1.5 py-0.5 rounded-full">Verified</span>
                </div>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 flex items-center gap-1"><Clock className="w-3.5 h-3.5" /> Experience</span>
              <span className="font-medium text-gray-900">
                {Math.round((candidate.experience_months_total || 0) / 12)} years
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 flex items-center gap-1"><Star className="w-3.5 h-3.5" /> Total skills</span>
              <span className="font-medium text-gray-900">{skillsData?.total_skills ?? '—'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 flex items-center gap-1"><Globe className="w-3.5 h-3.5" /> Language</span>
              <span className="font-medium text-gray-900">{candidate.resume_language?.toUpperCase() || 'EN'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500 flex items-center gap-1"><Shield className="w-3.5 h-3.5" /> AI content</span>
              <span className={`font-medium ${candidate.ai_content_probability > 0.7 ? 'text-amber-600' : 'text-green-600'}`}>
                {Math.round((candidate.ai_content_probability || 0) * 100)}%
              </span>
            </div>
          </div>

          {/* Summary */}
          {candidate.summary && (
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-2">Summary</h3>
              <p className="text-sm text-gray-600 leading-relaxed">{candidate.summary}</p>
            </div>
          )}
        </div>

        {/* Right: skills, experience, education */}
        <div className="lg:col-span-2 space-y-4">
          {/* Skills by proficiency */}
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-gray-900">Skills</h2>
              <div className="flex gap-3 text-xs text-gray-500">
                {skillsData?.by_category && Object.entries(skillsData.by_category).map(([cat, count]: [string, any]) => (
                  <span key={cat}>{count} {cat}</span>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {skillsData?.skills?.map((s: any, i: number) => (
                <span
                  key={i}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium border ${PROF_COLORS[s.proficiency] || PROF_COLORS.beginner}`}
                  title={`Source: ${s.source} · ${s.years}yr`}
                >
                  {s.canonical}
                  {s.inferred && ' ✦'}
                </span>
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-3">✦ = inferred from skill graph</p>
          </div>

          {/* Experience */}
          {candidate.experience?.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="font-semibold text-gray-900 mb-4">Experience</h2>
              <div className="space-y-4">
                {candidate.experience.map((exp: any, i: number) => (
                  <div key={i} className="border-l-2 border-indigo-200 pl-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="text-sm font-semibold text-gray-900">{exp.role || 'Role not specified'}</p>
                        <p className="text-sm text-gray-600">{exp.company || 'Company not specified'}</p>
                      </div>
                      <span className="text-xs text-gray-400 whitespace-nowrap ml-2">
                        {exp.start} — {exp.end || 'Present'}
                      </span>
                    </div>
                    {exp.bullets?.length > 0 && (
                      <ul className="mt-2 space-y-1">
                        {exp.bullets.slice(0, 3).map((b: string, j: number) => (
                          <li key={j} className="text-xs text-gray-500 flex gap-1.5">
                            <span className="mt-1 w-1 h-1 rounded-full bg-gray-400 flex-shrink-0" />
                            {b}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Education */}
          {candidate.education?.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="font-semibold text-gray-900 mb-3">Education</h2>
              <div className="space-y-3">
                {candidate.education.map((edu: any, i: number) => (
                  <div key={i} className="flex items-start justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {edu.degree} {edu.field ? `in ${edu.field}` : ''}
                      </p>
                      <p className="text-sm text-gray-500">{edu.institution}</p>
                    </div>
                    <span className="text-xs text-gray-400">{edu.year}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Certifications */}
          {candidate.certifications?.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="font-semibold text-gray-900 mb-3">Certifications</h2>
              <div className="flex flex-wrap gap-2">
                {candidate.certifications.map((cert: any, i: number) => (
                  <span key={i} className="px-3 py-1.5 bg-indigo-50 text-indigo-700 text-xs rounded-lg border border-indigo-200">
                    {cert.name} {cert.year ? `(${cert.year})` : ''}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
