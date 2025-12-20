import { useState, useEffect } from 'react'
import { Bell, CheckCircle, X } from 'lucide-react'
import Navbar from '../common/Navbar'
import Sidebar from '../common/Sidebar'
import AlertDetail from './AlertDetail'
import AlertSettings from './AlertSettings'
import { alertsApi } from '../../services/api'
import { formatRelativeTime } from '../../utils/formatters'
import { ALERT_SEVERITY_COLORS } from '../../utils/constants'
import LoadingSpinner from '../common/LoadingSpinner'
import toast from 'react-hot-toast'
import clsx from 'clsx'

const AlertList = () => {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [filter, setFilter] = useState('all') // all, active, resolved

  useEffect(() => {
    fetchAlerts()
  }, [filter])

  const fetchAlerts = async () => {
    setLoading(true)
    try {
      const params = {}
      if (filter === 'active') params.status = 'active'
      if (filter === 'resolved') params.status = 'resolved'
      
      const response = await alertsApi.getAll(params)
      setAlerts(response.data.alerts || [])
    } catch (error) {
      toast.error('Failed to fetch alerts')
    } finally {
      setLoading(false)
    }
  }

  const handleAcknowledge = async (alertId) => {
    try {
      await alertsApi.acknowledge(alertId)
      toast.success('Alert acknowledged')
      fetchAlerts()
    } catch (error) {
      toast.error('Failed to acknowledge alert')
    }
  }

  const handleResolve = async (alertId) => {
    try {
      await alertsApi.resolve(alertId)
      toast.success('Alert resolved')
      fetchAlerts()
    } catch (error) {
      toast.error('Failed to resolve alert')
    }
  }

  const handleDelete = async (alertId) => {
    if (!window.confirm('Are you sure you want to delete this alert?')) return
    
    try {
      await alertsApi.delete(alertId)
      toast.success('Alert deleted')
      fetchAlerts()
    } catch (error) {
      toast.error('Failed to delete alert')
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        
        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Alerts</h2>
                <p className="text-gray-600 mt-1">
                  Manage and monitor system alerts and notifications
                </p>
              </div>
              <button
                onClick={() => setShowSettings(true)}
                className="btn btn-primary"
              >
                Alert Settings
              </button>
            </div>

            {/* Filters */}
            <div className="flex space-x-2 mb-6">
              {['all', 'active', 'resolved'].map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={clsx(
                    'px-4 py-2 rounded-lg font-medium transition-colors',
                    filter === f
                      ? 'bg-primary-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  )}
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>

            {/* Alerts List */}
            {loading ? (
              <LoadingSpinner text="Loading alerts..." />
            ) : alerts.length === 0 ? (
              <div className="text-center py-12">
                <Bell className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600">No alerts found</p>
              </div>
            ) : (
              <div className="space-y-4">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="card hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <div className={`w-1 h-8 rounded ${ALERT_SEVERITY_COLORS[alert.severity]}`} />
                          <div>
                            <h3 className="text-lg font-semibold text-gray-900">
                              {alert.title}
                            </h3>
                            <p className="text-sm text-gray-600">
                              {formatRelativeTime(alert.created_at)}
                            </p>
                          </div>
                        </div>
                        <p className="text-gray-700 ml-3">{alert.message}</p>
                        
                        {alert.acknowledged_at && (
                          <div className="flex items-center space-x-2 mt-2 ml-3 text-sm text-gray-600">
                            <CheckCircle className="w-4 h-4 text-green-600" />
                            <span>Acknowledged {formatRelativeTime(alert.acknowledged_at)}</span>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center space-x-2 ml-4">
                        {alert.status === 'active' && !alert.acknowledged_at && (
                          <button
                            onClick={() => handleAcknowledge(alert.id)}
                            className="btn btn-secondary"
                          >
                            Acknowledge
                          </button>
                        )}
                        {alert.status !== 'resolved' && (
                          <button
                            onClick={() => handleResolve(alert.id)}
                            className="btn btn-primary"
                          >
                            Resolve
                          </button>
                        )}
                        <button
                          onClick={() => setSelectedAlert(alert)}
                          className="btn btn-secondary"
                        >
                          Details
                        </button>
                        <button
                          onClick={() => handleDelete(alert.id)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                          <X className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </main>
      </div>

      {selectedAlert && (
        <AlertDetail
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onUpdate={fetchAlerts}
        />
      )}

      {showSettings && (
        <AlertSettings
          onClose={() => setShowSettings(false)}
          onSave={fetchAlerts}
        />
      )}
    </div>
  )
}

export default AlertList