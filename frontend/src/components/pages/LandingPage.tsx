import { Link } from 'react-router-dom'
import { ArrowRight, FileText, ScanSearch, ShieldCheck, Sparkles } from 'lucide-react'

const features = [
  ['Resume intelligence', 'Turn PDF and DOCX resumes into structured, searchable profiles.', FileText],
  ['Explainable matching', 'Compare candidates to a role with transparent skill gaps and evidence.', ScanSearch],
  ['Bias-aware workflows', 'Keep human review in the loop when a decision needs attention.', ShieldCheck],
]

export default function LandingPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-slate-950 text-white">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_rgba(99,102,241,0.28),_transparent_38%),radial-gradient(circle_at_bottom_left,_rgba(14,165,233,0.2),_transparent_35%)]" />
      <div className="relative mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <header className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 font-semibold tracking-tight">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-500"><Sparkles className="h-5 w-5" /></span>
            <span>TalentAI-X</span>
          </Link>
          <Link to="/login" className="rounded-lg border border-white/20 px-4 py-2 text-sm font-medium hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300">Sign in</Link>
        </header>

        <section className="grid items-center gap-16 py-24 lg:grid-cols-[1.05fr_0.95fr] lg:py-32">
          <div>
            <p className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-300/30 bg-indigo-300/10 px-3 py-1 text-xs font-medium text-indigo-200">
              <Sparkles className="h-3.5 w-3.5" /> Human-centered talent intelligence
            </p>
            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.05] tracking-tight text-balance sm:text-6xl">
              Make every hiring decision clearer.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
              Upload a resume, get a useful overview, and move from raw documents to confident decisions without losing the human reviewer.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Link to="/login" className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 px-5 py-3 text-sm font-semibold hover:bg-indigo-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-200">
                Get started <ArrowRight className="h-4 w-4" />
              </Link>
              <Link to="/login?role=recruiter" className="rounded-lg border border-white/20 px-5 py-3 text-sm font-semibold hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300">
                HR / recruiter sign in
              </Link>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/[0.07] p-5 shadow-2xl shadow-indigo-950/30 backdrop-blur">
            <div className="rounded-2xl bg-slate-900/90 p-5">
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div><p className="text-sm font-medium">Candidate overview</p><p className="mt-1 text-xs text-slate-400">Generated from resume.pdf</p></div>
                <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-xs font-medium text-emerald-300">Ready for review</span>
              </div>
              <div className="grid gap-3 py-5 sm:grid-cols-3">
                <div className="rounded-xl bg-white/5 p-3"><p className="text-xs text-slate-400">Skills found</p><p className="mt-2 text-2xl font-semibold">18</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-xs text-slate-400">Experience</p><p className="mt-2 text-2xl font-semibold">6.4y</p></div>
                <div className="rounded-xl bg-white/5 p-3"><p className="text-xs text-slate-400">Confidence</p><p className="mt-2 text-2xl font-semibold">94%</p></div>
              </div>
              <div className="space-y-3">
                {['Python', 'FastAPI', 'PostgreSQL'].map((skill, index) => (
                  <div key={skill} className="flex items-center justify-between rounded-lg bg-white/5 px-3 py-2.5 text-sm">
                    <span>{skill}</span><span className="text-xs text-indigo-300">{index === 0 ? 'Expert' : 'Advanced'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-4 border-t border-white/10 py-10 md:grid-cols-3">
          {features.map(([title, description, Icon]) => (
            <div key={title as string} className="rounded-2xl border border-white/10 bg-white/[0.04] p-5">
              <Icon className="h-5 w-5 text-indigo-300" />
              <h2 className="mt-4 text-sm font-semibold">{title as string}</h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">{description as string}</p>
            </div>
          ))}
        </section>
      </div>
    </main>
  )
}
