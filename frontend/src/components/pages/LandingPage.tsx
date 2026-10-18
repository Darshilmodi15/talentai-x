import { Link } from 'react-router-dom'
import { ArrowRight, FileText, ScanSearch, ShieldCheck, Sparkles } from 'lucide-react'

const features = [
  ['Resume intelligence', 'Turn PDF and DOCX resumes into structured, searchable profiles.', FileText],
  ['Explainable matching', 'Compare candidates to a role with transparent skill gaps and evidence.', ScanSearch],
  ['Bias-aware workflows', 'Keep human review in the loop when a decision needs attention.', ShieldCheck],
]

export default function LandingPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-gray-50 text-gray-900">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(79,70,229,0.12),_transparent_38%),radial-gradient(circle_at_bottom_left,_rgba(99,102,241,0.08),_transparent_35%)]" />
      <div className="relative mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <header className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 font-semibold tracking-tight text-gray-900">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-600 text-white"><Sparkles className="h-5 w-5" /></span>
            <span>TalentAI-X</span>
          </Link>
          <Link to="/login" className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:border-indigo-300 hover:bg-indigo-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">Sign in</Link>
        </header>

        <section className="grid items-center gap-16 py-24 lg:grid-cols-[1.05fr_0.95fr] lg:py-32">
          <div>
            <p className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
              <Sparkles className="h-3.5 w-3.5" /> Human-centered talent intelligence
            </p>
            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.05] tracking-tight text-gray-950 text-balance sm:text-6xl">
              Make every hiring decision clearer.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-gray-600">
              Upload a resume, get a useful overview, and move from raw documents to confident decisions without losing the human reviewer.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link to="/login" className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-3 text-sm font-semibold text-white hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
                Get started <ArrowRight className="h-4 w-4" />
              </Link>
              <Link to="/login?role=recruiter" className="rounded-lg border border-gray-300 bg-white px-5 py-3 text-sm font-semibold text-gray-700 hover:border-indigo-300 hover:bg-indigo-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
                HR / recruiter sign in
              </Link>
            </div>
          </div>

          <div className="rounded-3xl border border-gray-200 bg-white p-5 shadow-xl shadow-indigo-100/60">
            <div className="rounded-2xl bg-gray-50 p-5">
              <div className="flex items-center justify-between border-b border-gray-200 pb-4">
                <div><p className="text-sm font-medium text-gray-900">Candidate overview</p><p className="mt-1 text-xs text-gray-500">Generated from resume.pdf</p></div>
                <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-xs font-medium text-emerald-300">Ready for review</span>
              </div>
              <div className="grid gap-3 py-5 sm:grid-cols-3">
                <div className="rounded-xl bg-white p-3"><p className="text-xs text-gray-500">Skills found</p><p className="mt-2 text-2xl font-semibold text-gray-900">18</p></div>
                <div className="rounded-xl bg-white p-3"><p className="text-xs text-gray-500">Experience</p><p className="mt-2 text-2xl font-semibold text-gray-900">6.4y</p></div>
                <div className="rounded-xl bg-white p-3"><p className="text-xs text-gray-500">Confidence</p><p className="mt-2 text-2xl font-semibold text-gray-900">94%</p></div>
              </div>
              <div className="space-y-3">
                {['Python', 'FastAPI', 'PostgreSQL'].map((skill, index) => (
                  <div key={skill} className="flex items-center justify-between rounded-lg bg-white px-3 py-2.5 text-sm text-gray-800">
                  <span>{skill}</span><span className="text-xs text-indigo-600">{index === 0 ? 'Expert' : 'Advanced'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 border-t border-gray-200 py-10 md:grid-cols-3">
          {features.map(([title, description, Icon]) => (
            <div key={title as string} className="rounded-2xl border border-gray-200 bg-white p-5">
              <Icon className="h-5 w-5 text-indigo-600" />
              <h2 className="mt-4 text-sm font-semibold text-gray-900">{title as string}</h2>
              <p className="mt-2 text-sm leading-6 text-gray-600">{description as string}</p>
            </div>
          ))}
        </section>
      </div>
    </main>
  )
}
