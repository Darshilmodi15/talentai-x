import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'

import Layout from './components/layouts/Layout'
import Dashboard from './components/pages/Dashboard'
import UploadPage from './components/pages/UploadPage'
import CandidatesPage from './components/pages/CandidatesPage'
import CandidateDetail from './components/pages/CandidateDetail'
import MatchPage from './components/pages/MatchPage'
import TaxonomyPage from './components/pages/TaxonomyPage'
import AdminPage from './components/pages/AdminPage'
import BatchMatchPage from './components/pages/BatchMatchPage'
import LandingPage from './components/pages/LandingPage'
import LoginPage from './components/pages/LoginPage'
import RequireAuth from './auth/RequireAuth'
import { AuthProvider } from './auth/AuthContext'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
    },
  },
})

export default function App() {
  return (
    <AuthProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Toaster position="top-right" />
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<Layout />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/upload" element={<UploadPage />} />
                <Route path="/candidates" element={<CandidatesPage />} />
                <Route path="/candidates/:id" element={<CandidateDetail />} />
                <Route element={<RequireAuth role="recruiter" />}>
                  <Route path="/match" element={<MatchPage />} />
                  <Route path="/match/batch" element={<BatchMatchPage />} />
                  <Route path="/admin" element={<AdminPage />} />
                </Route>
                <Route path="/taxonomy" element={<TaxonomyPage />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </AuthProvider>
  )
}
