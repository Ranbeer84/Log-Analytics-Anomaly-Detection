import { X, CheckCircle, XCircle } from 'lucide-react'
import { formatTimestamp } from '../../utils/formatters'
import { ALERT_SEVERITY_COLORS } from '../../utils/constants'
import { alertsApi } from '../../services/api'
import toast from 'react-hot-toast'

const AlertDetail = ({ alert, onClose, onUpdate }) => {
  const handleAcknowledge = async () => {
    try {
      await alertsApi.acknowledge(alert.id)
      toast.success('Alert acknowledged')
      onUpdate()
      onClose()
    } catch (error) {
      toast.error('Failed to acknowledge alert')
    }
  }

  const handleResolve = async () => {
    try {
      await alertsApi.resolve(alert.id)
      toast.success('Alert resolved')
      onUpdate()
      onClose()
    } catch (error) {
      toast.error('Failed to resolve alert')
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex items-center justify-between">
          <h3 className="text-xl font-bold text-gray-900">Alert Details</h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${ALERT_SEVERITY_COLORS[alert.severity]} text-white`}>
              {alert.severity}
            </div>
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              alert.status === 'active' ? 'bg-red-100 text-red-800' :
              alert.status === 'acknowledged' ? 'bg-yellow-100 text-yellow-800' :
              'bg-green-100 text-green-800'
            }`}>
              {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
            </div>
          </div>

          <div>
            <h4 className="text-2xl font-bold text-gray-900 mb-2">{alert.title}</h4>
            <p className="text-gray-700">{alert.message}</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-1">Created</label>
              <p className="text-sm text-gray-900">{formatTimestamp(alert.created_at)}</p>
            </div>
            {alert.acknowledged_at && (
              <div>
                <label className="text-sm font-medium text-gray-700 block mb-1">Acknowledged</label>
                <p className="text-sm text-gray-900">{formatTimestamp(alert.acknowledged_at)}</p>
              </div>
            )}
            {alert.resolved_at && (
              <div>
                <label className="text-sm font-medium text-gray-700 block mb-1">Resolved</label>
                <p className="text-sm text-gray-900">{formatTimestamp(alert.resolved_at)}</p>
              </div>
            )}
          </div>

          {alert.metadata && Object.keys(alert.metadata).length > 0 && (
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-2">Metadata</label>
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-xs text-gray-900 whitespace-pre-wrap font-mono">
                  {JSON.stringify(alert.metadata, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {alert.related_logs && alert.related_logs.length > 0 && (
            <div>
              <label className="text-sm font-medium text-gray-700 block mb-2">Related Logs</label>
              <div className="space-y-2">
                {alert.related_logs.map((log, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3">
                    <p className="text-sm text-gray-900">{log.message}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {formatTimestamp(log.timestamp)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="pt-4 border-t border-gray-200 flex space-x-3">
            {alert.status === 'active' && !alert.acknowledged_at && (
              <button
                onClick={handleAcknowledge}
                className="btn btn-secondary flex-1 flex items-center justify-center space-x-2"
              >
                <CheckCircle className="w-5 h-5" />
                <span>Acknowledge</span>
              </button>
            )}
            {alert.status !== 'resolved' && (
              <button
                onClick={handleResolve}
                className="btn btn-primary flex-1 flex items-center justify-center space-x-2"
              >
                <XCircle className="w-5 h-5" />
                <span>Resolve</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AlertDetail
