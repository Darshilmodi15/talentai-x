import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Upload, Users, Zap, BookOpen,
  Settings, GitBranch, BarChart3, Shield
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { to: '/dashboard',   icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload',      icon: Upload,           label: 'Upload Resume' },
  { to: '/candidates',  icon: Users,            label: 'Candidates' },
  { to: '/match',       icon: Zap,              label: 'Match' },
  { to: '/match/batch', icon: BarChart3,        label: 'Batch Match' },
  { to: '/taxonomy',    icon: BookOpen,         label: 'Skill Taxonomy' },
  { to: '/admin',       icon: Settings,         label: 'Admin' },
]

export default function Layout() {
  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-sm font-bold text-gray-900">TalentAI-X</p>
              <p className="text-xs text-gray-500">Talent Intelligence</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                )
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-green-500" />
            <p className="text-xs text-gray-500">Bias-protected · GDPR ready</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
