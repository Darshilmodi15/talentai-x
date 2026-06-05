import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, BookOpen, Tag, AlertCircle } from 'lucide-react'
import { browseTaxonomy, searchTaxonomy, getEmergingSkills } from '../../api/client'

const CATEGORY_COLORS: Record<string, string> = {
  technical: 'bg-blue-50 text-blue-700',
  soft: 'bg-green-50 text-green-700',
  domain: 'bg-purple-50 text-purple-700',
}

export default function TaxonomyPage() {
  const [searchQ, setSearchQ] = useState('')
  const [activeTab, setActiveTab] = useState<'browse' | 'emerging'>('browse')
  const [category, setCategory] = useState('')

  const { data: taxonomy, isLoading } = useQuery({
    queryKey: ['taxonomy', category],
    queryFn: () => browseTaxonomy(category || undefined),
    enabled: !searchQ && activeTab === 'browse',
  })

  const { data: searchResult } = useQuery({
    queryKey: ['taxonomy-search', searchQ],
    queryFn: () => searchTaxonomy(searchQ),
    enabled: searchQ.length >= 2,
  })

  const { data: emerging } = useQuery({
    queryKey: ['emerging-skills'],
    queryFn: getEmergingSkills,
    enabled: activeTab === 'emerging',
  })

  const displaySkills = searchQ.length >= 2
    ? searchResult?.results || []
    : taxonomy?.skills || []

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Skill Taxonomy</h1>
        <p className="text-gray-500 mt-1">{taxonomy?.total ?? 0} skills · Self-evolving via Agent 5</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-gray-100 p-1 rounded-lg w-fit">
        {(['browse', 'emerging'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab === 'browse' ? 'Browse' : `Emerging ${emerging?.count ? `(${emerging.count})` : ''}`}
          </button>
        ))}
      </div>

      {activeTab === 'browse' && (
        <>
          <div className="flex gap-3 mb-5">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search skills..."
                value={searchQ}
                onChange={(e) => setSearchQ(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">All categories</option>
              <option value="technical">Technical</option>
              <option value="soft">Soft skills</option>
              <option value="domain">Domain</option>
            </select>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="grid grid-cols-4 gap-0 divide-y divide-gray-100">
              {isLoading
                ? Array.from({ length: 20 }).map((_, i) => (
                    <div key={i} className="p-3"><div className="h-4 bg-gray-100 rounded animate-pulse" /></div>
                  ))
                : displaySkills.map((s: any, i: number) => (
                    <div key={i} className="p-3 hover:bg-gray-50">
                      <div className="flex items-start gap-2">
                        <Tag className="w-3.5 h-3.5 text-gray-400 flex-shrink-0 mt-0.5" />
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">{s.canonical_name}</p>
                          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                            <span className={`px-1.5 py-0.5 rounded text-xs ${CATEGORY_COLORS[s.category] || 'bg-gray-100 text-gray-600'}`}>
                              {s.category}
                            </span>
                            {s.parent && (
                              <span className="text-xs text-gray-400 truncate">↳ {s.parent}</span>
                            )}
                          </div>
                          {s.synonyms?.length > 0 && (
                            <p className="text-xs text-gray-400 mt-1 truncate">{s.synonyms.slice(0, 3).join(', ')}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
            </div>
          </div>
        </>
      )}

      {activeTab === 'emerging' && (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3 bg-amber-50 border-b border-amber-200">
            <p className="text-sm text-amber-700 flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              Skills found in resumes but not yet in taxonomy. Review and approve to add them.
            </p>
          </div>
          <div className="divide-y divide-gray-100">
            {emerging?.emerging_skills?.map((s: any) => (
              <div key={s.id} className="px-5 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{s.raw_skill}</p>
                  <p className="text-xs text-gray-400">Seen {s.occurrences}× · First seen {new Date(s.first_seen).toLocaleDateString()}</p>
                </div>
                <span className="px-2 py-1 bg-amber-50 text-amber-700 text-xs rounded-full border border-amber-200">
                  {s.status}
                </span>
              </div>
            ))}
            {!emerging?.emerging_skills?.length && (
              <div className="px-5 py-12 text-center text-gray-400 text-sm">
                No emerging skills pending review
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
