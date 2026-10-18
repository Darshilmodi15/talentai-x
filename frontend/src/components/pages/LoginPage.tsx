import { FormEvent, useEffect, useState } from 'react'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Loader2, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import { useAuth, UserRole } from '../../auth/AuthContext'

export default function LoginPage() {
  const { user, signIn, signUp } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [role, setRole] = useState<UserRole>(
    searchParams.get('role') === 'recruiter'
      ? 'recruiter'
      : 'individual'
  )
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (user) navigate('/dashboard', { replace: true })
  }, [navigate, user])

  if (user) return null

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    try {
      if (mode === 'signin') await signIn(email, password)
      else await signUp(name, email, password, role)
      toast.success(mode === 'signin' ? 'Welcome back' : 'Account created. Check your email if confirmation is enabled.')
      navigate((location.state as { from?: string } | null)?.from || '/dashboard', { replace: true })
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Unable to complete authentication')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-8">
      <div className="mx-auto flex max-w-6xl items-center justify-between">
        <Link to="/" className="inline-flex items-center gap-2 text-sm font-semibold text-slate-900"><ArrowLeft className="h-4 w-4" /> TalentAI-X</Link>
        <span className="text-xs text-slate-500">Secure workspace access</span>
      </div>
      <div className="mx-auto grid max-w-5xl gap-10 py-16 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
        <div className="hidden lg:block">
          <div className="mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-indigo-600 text-white"><Sparkles className="h-6 w-6" /></div>
          <h1 className="max-w-md text-4xl font-semibold tracking-tight text-slate-950">Your hiring workspace, ready when you are.</h1>
          <p className="mt-5 max-w-md leading-7 text-slate-600">Individuals get a private resume overview. Recruiters get bulk processing, matching, and review tools.</p>
        </div>
        <div className="mx-auto w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 shadow-xl shadow-slate-200/50 sm:p-8">
          <div className="mb-7">
            <h2 className="text-xl font-semibold text-slate-950">{mode === 'signin' ? 'Sign in' : 'Create your account'}</h2>
            <p className="mt-1 text-sm text-slate-500">{mode === 'signin' ? 'Continue to your workspace.' : 'Choose the workspace that fits your work.'}</p>
          </div>
          <form onSubmit={submit} className="space-y-4">
            {mode === 'signup' && <label className="block text-sm font-medium text-slate-700">Name<input required name="name" value={name} onChange={(event) => setName(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" autoComplete="name" /></label>}
            <label className="block text-sm font-medium text-slate-700">Email<input required type="email" name="email" value={email} onChange={(event) => setEmail(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" autoComplete="email" spellCheck={false} /></label>
            <label className="block text-sm font-medium text-slate-700">Password<input required minLength={6} type="password" name="password" value={password} onChange={(event) => setPassword(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500" autoComplete={mode === 'signin' ? 'current-password' : 'new-password'} /></label>
            {mode === 'signup' && <fieldset><legend className="text-sm font-medium text-slate-700">Workspace type</legend><div className="mt-2 grid grid-cols-2 gap-2">{(['individual', 'recruiter'] as UserRole[]).map((option) => <button type="button" key={option} onClick={() => setRole(option)} className={`rounded-lg border px-3 py-2.5 text-left text-sm capitalize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${role === option ? 'border-indigo-500 bg-indigo-50 text-indigo-700' : 'border-slate-200 text-slate-600'}`}>{option === 'recruiter' ? 'HR / Recruiter' : 'Individual'}</button>)}</div></fieldset>}
            <button disabled={busy} className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">{busy && <Loader2 className="h-4 w-4 animate-spin" />}{mode === 'signin' ? 'Sign in' : 'Create account'}</button>
          </form>
          <p className="mt-6 text-center text-sm text-slate-500">{mode === 'signin' ? 'New here?' : 'Already have an account?'} <button onClick={() => setMode(mode === 'signin' ? 'signup' : 'signin')} className="font-semibold text-indigo-600 hover:text-indigo-700">{mode === 'signin' ? 'Create an account' : 'Sign in'}</button></p>
        </div>
      </div>
    </main>
  )
}
