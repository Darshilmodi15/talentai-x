import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { Session, User } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'
import { setAccessToken } from '../api/client'

export type UserRole = 'individual' | 'recruiter'

type AppUser = {
  id: string
  email: string
  role: UserRole
  name: string
  isDemo?: boolean
}

type AuthContextValue = {
  user: AppUser | null
  session: Session | null
  loading: boolean
  signIn: (email: string, password: string) => Promise<void>
  signUp: (name: string, email: string, password: string, role: UserRole) => Promise<void>
  continueAsDemo: (role: UserRole) => void
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)
const DEMO_STORAGE_KEY = 'talentai-demo-user'

const toAppUser = (user: User, metadata: Record<string, unknown> = {}): AppUser => ({
  id: user.id,
  email: user.email ?? '',
  role: metadata.role === 'recruiter' ? 'recruiter' : 'individual',
  name: typeof metadata.name === 'string' && metadata.name ? metadata.name : user.email?.split('@')[0] ?? 'User',
})

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [user, setUser] = useState<AppUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!supabase) {
      const stored = localStorage.getItem(DEMO_STORAGE_KEY)
      if (stored) setUser(JSON.parse(stored))
      setLoading(false)
      return
    }

    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setUser(data.session?.user ? toAppUser(data.session.user, data.session.user.user_metadata) : null)
      setAccessToken(data.session?.access_token ?? null)
      setLoading(false)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession)
      setUser(nextSession?.user ? toAppUser(nextSession.user, nextSession.user.user_metadata) : null)
      setAccessToken(nextSession?.access_token ?? null)
    })

    return () => listener.subscription.unsubscribe()
  }, [])

  const value = useMemo<AuthContextValue>(() => ({
    user,
    session,
    loading,
    signIn: async (email, password) => {
      if (!supabase) throw new Error('Supabase Auth is not configured. Use demo access or add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.')
      const { error } = await supabase.auth.signInWithPassword({ email, password })
      if (error) throw error
    },
    signUp: async (name, email, password, role) => {
      if (!supabase) throw new Error('Supabase Auth is not configured. Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.')
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { name, role } },
      })
      if (error) throw error
    },
    continueAsDemo: (role) => {
      const demoUser: AppUser = {
        id: `demo-${role}`,
        email: role === 'recruiter' ? 'demo.hr@talentai.local' : 'demo.user@talentai.local',
        role,
        name: role === 'recruiter' ? 'Demo Recruiter' : 'Demo User',
        isDemo: true,
      }
      localStorage.setItem(DEMO_STORAGE_KEY, JSON.stringify(demoUser))
      setUser(demoUser)
    },
    signOut: async () => {
      localStorage.removeItem(DEMO_STORAGE_KEY)
      setAccessToken(null)
      if (supabase) {
        const { error } = await supabase.auth.signOut()
        if (error) throw error
      }
      setSession(null)
      setUser(null)
    },
  }), [loading, session, user])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
