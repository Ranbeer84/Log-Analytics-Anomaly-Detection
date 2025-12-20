import { NavLink } from 'react-router-dom'
import { LayoutDashboard, FileText, AlertTriangle, Settings } from 'lucide-react'

const Sidebar = () => {
  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
    { name: 'Logs', path: '/logs', icon: FileText },
    { name: 'Alerts', path: '/alerts', icon: AlertTriangle },
    { name: 'Settings', path: '/settings', icon: Settings },
  ]

  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-primary-50 text-primary-700'
                  : 'text-gray-700 hover:bg-gray-100'
              }`
            }
          >
            <item.icon className="w-5 h-5" />
            <span className="font-medium">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-gray-200">
        <div className="card bg-primary-50 border-primary-200">
          <h3 className="text-sm font-semibold text-primary-900 mb-1">Need Help?</h3>
          <p className="text-xs text-primary-700 mb-3">
            Check our documentation for guides and API references.
          </p>
          <button className="w-full btn btn-primary text-sm py-2">
            View Docs
          </button>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar