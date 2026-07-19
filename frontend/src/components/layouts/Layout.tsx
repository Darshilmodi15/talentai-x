import { Outlet, NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Upload, Users, Zap, BookOpen,
  Settings, GitBranch, BarChart3, Shield, AlertTriangle, Github, ExternalLink
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
    <div className="flex flex-col h-screen overflow-hidden bg-gray-50">
      {/* Warning / Maintenance Banner */}
      <div className="bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-rose-500/10 border-b border-orange-500/20 px-4 py-3 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3 text-sm text-gray-800">
            <span className="flex h-3.5 w-3.5 items-center justify-center relative flex-shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
            <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
            <p className="font-medium text-gray-700">
              <span className="text-amber-800 font-semibold mr-1">Project Notice:</span>
              The API is temporarily disabled. I am actively working on resolving the backend connection to restore full functionality. In the meantime, you are welcome to explore my other projects!
            </p>
          </div>
          <a
            href="https://github.com/Darshilmodi15"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-1.5 bg-gray-900 hover:bg-black text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition-all duration-200"
          >
            <Github className="w-3.5 h-3.5" />
            <span>Explore My GitHub</span>
            <ExternalLink className="w-3 h-3 opacity-60" />
          </a>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
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
    </div>
  )
}
