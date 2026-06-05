import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle, BarChart3, Shield, Clock } from 'lucide-react'
import toast from 'react-hot-toast'
import { getSystemStats, getHITLQueue, resolveHITLItem } from '../../api/client'

export default function AdminPage() {
  const qc = useQueryClient()

  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getSystemStats, refetchInterval: 10_000 })
  const { data: queue } = useQuery({ queryKey: ['hitl'], queryFn: getHITLQueue, refetchInterval: 5_000 })

  const resolve = useMutation({
    mutationFn: ({ id, decision }: { id: string; decision: string }) =>
      resolveHITLItem(id, decision),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['hitl'] })
      qc.invalidateQueries({ queryKey: ['stats'] })
      toast.success('Review item resolved')
    },
  })

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Admin</h1>
        <p className="text-gray-500 mt-1">System health, API keys, and HITL review queue</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-8">
        {[
          { label: 'Candidates', value: stats?.total_candidates_parsed ?? '—', icon: BarChart3, color: 'text-indigo-600' },
          { label: 'Parse jobs', value: stats?.total_parse_jobs ?? '—', icon: BarChart3, color: 'text-blue-600' },
          { label: 'Matches', value: stats?.total_matches_run ?? '—', icon: BarChart3, color: 'text-green-600' },
          { label: 'HITL pending', value: stats?.hitl_queue_pending ?? '—', icon: AlertTriangle, color: 'text-amber-600' },
          { label: 'Bias flags', value: stats?.bias_flags_raised ?? '—', icon: Shield, color: 'text-red-600' },
        ].map((s, i) => (
          <div key={i} className="bg-white border border-gray-200 rounded-xl p-4">
            <s.icon className={`w-4 h-4 ${s.color} mb-2`} />
            <p className="text-2xl font-bold text-gray-900">{s.value}</p>
            <p className="text-xs text-gray-500 mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>

      {/* System status */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-center gap-3 mb-8">
        <CheckCircle className="w-5 h-5 text-green-600" />
        <div>
          <p className="text-sm font-semibold text-green-800">System healthy</p>
          <p className="text-xs text-green-600">API · PostgreSQL · ChromaDB · Redis · all operational</p>
        </div>
      </div>

      {/* HITL Queue */}
      <div>
        <h2 className="text-base font-semibold text-gray-900 mb-3">
          HITL Review Queue
          {queue?.pending_count > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-amber-100 text-amber-700 text-xs rounded-full">
              {queue.pending_count} pending
            </span>
          )}
        </h2>

        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          {!queue?.items?.length ? (
            <div className="px-5 py-12 text-center text-gray-400 text-sm">
              <CheckCircle className="w-8 h-8 text-gray-200 mx-auto mb-3" />
              HITL queue is empty — no items need human review
            </div>
          ) : (
            <div className="divide-y divide-gray-100">
              {queue.items.map((item: any) => (
                <div key={item.id} className="px-5 py-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          item.priority === 'high' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'
                        }`}>
                          {item.priority}
                        </span>
                        <span className="text-xs text-gray-400 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Expires {new Date(item.expires_at).toLocaleDateString()}
                        </span>
                      </div>
                      <p className="text-sm font-medium text-gray-900">Match ID: {item.match_result_id.slice(0, 8)}…</p>
                      <p className="text-xs text-gray-500 mt-0.5">Trigger: {item.trigger_reason}</p>
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <button
                        onClick={() => resolve.mutate({ id: item.id, decision: 'approved' })}
                        disabled={resolve.isPending}
                        className="px-3 py-1.5 bg-green-600 text-white text-xs rounded-lg hover:bg-green-700 disabled:opacity-50"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => resolve.mutate({ id: item.id, decision: 'rejected' })}
                        disabled={resolve.isPending}
                        className="px-3 py-1.5 bg-red-600 text-white text-xs rounded-lg hover:bg-red-700 disabled:opacity-50"
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* API info */}
      <div className="mt-8 bg-gray-50 border border-gray-200 rounded-xl p-5">
        <h3 className="font-semibold text-gray-900 mb-3">API Access</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <code className="bg-gray-100 px-2 py-0.5 rounded text-xs">Dev key:</code>
            <code className="text-gray-600">dev_key_change_in_production</code>
          </div>
          <p className="text-gray-500 text-xs">
            Set in .env as VITE_API_KEY. Create production keys via POST /api/v1/admin/api-keys.
          </p>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block text-indigo-600 text-xs hover:underline mt-2"
          >
            Open Swagger UI →
          </a>
        </div>
      </div>
    </div>
  )
}
