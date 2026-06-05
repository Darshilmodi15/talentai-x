import { useQuery } from '@tanstack/react-query'
import { Users, Zap, AlertTriangle, Shield, Upload, ArrowRight } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { getSystemStats } from '../../api/client'

function StatCard({ label, value, icon: Icon, color }: {
  label: string; value: string | number; icon: any; color: string
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-gray-500 font-medium">{label}</p>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${color}`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
      </div>
      <p className="text-3xl font-bold text-gray-900">{value}</p>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: getSystemStats,
    refetchInterval: 30_000,
  })

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">Multi-agent talent intelligence pipeline overview</p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Candidates parsed" value={isLoading ? '...' : stats?.total_candidates_parsed ?? 0}
          icon={Users} color="bg-indigo-500"
        />
        <StatCard
          label="Matches run" value={isLoading ? '...' : stats?.total_matches_run ?? 0}
          icon={Zap} color="bg-green-500"
        />
        <StatCard
          label="HITL queue" value={isLoading ? '...' : stats?.hitl_queue_pending ?? 0}
          icon={AlertTriangle} color="bg-amber-500"
        />
        <StatCard
          label="Bias flags raised" value={isLoading ? '...' : stats?.bias_flags_raised ?? 0}
          icon={Shield} color="bg-red-500"
        />
      </div>

      {/* Quick actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <button
          onClick={() => navigate('/upload')}
          className="bg-indigo-600 text-white rounded-xl p-6 text-left hover:bg-indigo-700 transition-colors"
        >
          <Upload className="w-6 h-6 mb-3" />
          <p className="font-semibold text-lg">Upload Resume</p>
          <p className="text-indigo-200 text-sm mt-1">Parse PDF, DOCX, or TXT</p>
          <ArrowRight className="w-4 h-4 mt-3" />
        </button>

        <button
          onClick={() => navigate('/match')}
          className="bg-white border border-gray-200 rounded-xl p-6 text-left hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
        >
          <Zap className="w-6 h-6 mb-3 text-indigo-600" />
          <p className="font-semibold text-lg text-gray-900">Match Candidate</p>
          <p className="text-gray-500 text-sm mt-1">Semantic matching with CoT</p>
          <ArrowRight className="w-4 h-4 mt-3 text-gray-400" />
        </button>

        <button
          onClick={() => navigate('/candidates')}
          className="bg-white border border-gray-200 rounded-xl p-6 text-left hover:border-indigo-300 hover:bg-indigo-50 transition-colors"
        >
          <Users className="w-6 h-6 mb-3 text-indigo-600" />
          <p className="font-semibold text-lg text-gray-900">View Candidates</p>
          <p className="text-gray-500 text-sm mt-1">Browse parsed profiles</p>
          <ArrowRight className="w-4 h-4 mt-3 text-gray-400" />
        </button>
      </div>

      {/* Pipeline info */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-base font-semibold text-gray-900 mb-4">Agent pipeline</h2>
        <div className="flex items-center gap-3 flex-wrap">
          {[
            { label: '1. Parser', desc: 'Layout-aware extraction', color: 'bg-purple-100 text-purple-700' },
            { label: '2. Normalizer', desc: 'Skill taxonomy mapping', color: 'bg-blue-100 text-blue-700' },
            { label: '3. Resolver', desc: 'Identity graph matching', color: 'bg-green-100 text-green-700' },
            { label: '4. Matcher', desc: 'CoT semantic scoring', color: 'bg-amber-100 text-amber-700' },
            { label: '5. Bias Shield', desc: 'Blind score audit', color: 'bg-red-100 text-red-700' },
          ].map((agent, i) => (
            <div key={i} className={`px-4 py-2 rounded-lg ${agent.color}`}>
              <p className="text-xs font-semibold">{agent.label}</p>
              <p className="text-xs opacity-75">{agent.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
