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
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Toaster position="top-right" />
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="upload" element={<UploadPage />} />
            <Route path="candidates" element={<CandidatesPage />} />
            <Route path="candidates/:id" element={<CandidateDetail />} />
            <Route path="match" element={<MatchPage />} />
            <Route path="match/batch" element={<BatchMatchPage />} />
            <Route path="taxonomy" element={<TaxonomyPage />} />
            <Route path="admin" element={<AdminPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
