import { X, Copy, ExternalLink } from 'lucide-react'
import { formatTimestamp } from '../../utils/formatters'
import { LOG_LEVEL_COLORS } from '../../utils/constants'
import toast from 'react-hot-toast'
import clsx from 'clsx'

const LogDetail = ({ log, onClose }) => {
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }

  const copyFullLog = () => {
    const logText = JSON.stringify(log, null, 2)
    copyToClipboard(logText)
  }

  return (
    <div className="w-1/3 border-l border-gray-200 bg-white overflow-y-auto">
      <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex items-center justify-between z-10">
        <h3 className="text-lg font-semibold text-gray-900">Log Details</h3>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      <div className="p-6 space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center space-x-2 mb-2">
            <span className={clsx('badge', LOG_LEVEL_COLORS[log.level])}>
              {log.level}
            </span>
            {log.anomaly_score && log.anomaly_score > 0.7 && (
              <span className="badge bg-yellow-100 text-yellow-800">
                Anomaly: {(log.anomaly_score * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600">{formatTimestamp(log.timestamp)}</p>
        </div>

        {/* Message */}
        <div>
          <label className="text-sm font-medium text-gray-700 mb-2 block">Message</label>
          <div className="bg-gray-50 rounded-lg p-4 relative group">
            <p className="text-sm text-gray-900 whitespace-pre-wrap">{log.message}</p>
            <button
              onClick={() => copyToClipboard(log.message)}
              className="absolute top-2 right-2 p-1.5 bg-white rounded opacity-0 group-hover:opacity-100 transition-opacity shadow-sm"
              title="Copy message"
            >
              <Copy className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Metadata */}
        <div className="space-y-3">
          <label className="text-sm font-medium text-gray-700 block">Metadata</label>
          
          {log.service && (
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-sm text-gray-600">Service</span>
              <span className="text-sm font-medium text-gray-900">{log.service}</span>
            </div>
          )}
          
          {log.host && (
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-sm text-gray-600">Host</span>
              <span className="text-sm font-medium text-gray-900">{log.host}</span>
            </div>
          )}
          
          {log.user_id && (
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-sm text-gray-600">User ID</span>
              <span className="text-sm font-medium text-gray-900">{log.user_id}</span>
            </div>
          )}
          
          {log.request_id && (
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-sm text-gray-600">Request ID</span>
              <span className="text-sm font-medium text-gray-900 font-mono">{log.request_id}</span>
            </div>
          )}
          
          {log.response_time && (
            <div className="flex justify-between py-2 border-b border-gray-100">
              <span className="text-sm text-gray-600">Response Time</span>
              <span className="text-sm font-medium text-gray-900">{log.response_time}ms</span>
            </div>
          )}
        </div>

        {/* Stack Trace */}
        {log.stack_trace && (
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">Stack Trace</label>
            <div className="bg-gray-900 rounded-lg p-4 relative group">
              <pre className="text-xs text-gray-100 whitespace-pre-wrap font-mono overflow-x-auto">
                {log.stack_trace}
              </pre>
              <button
                onClick={() => copyToClipboard(log.stack_trace)}
                className="absolute top-2 right-2 p-1.5 bg-gray-800 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                title="Copy stack trace"
              >
                <Copy className="w-4 h-4 text-gray-300" />
              </button>
            </div>
          </div>
        )}

        {/* Additional Data */}
        {log.additional_data && Object.keys(log.additional_data).length > 0 && (
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">Additional Data</label>
            <div className="bg-gray-50 rounded-lg p-4 relative group">
              <pre className="text-xs text-gray-900 whitespace-pre-wrap font-mono overflow-x-auto">
                {JSON.stringify(log.additional_data, null, 2)}
              </pre>
              <button
                onClick={() => copyToClipboard(JSON.stringify(log.additional_data))}
                className="absolute top-2 right-2 p-1.5 bg-white rounded opacity-0 group-hover:opacity-100 transition-opacity shadow-sm"
                title="Copy data"
              >
                <Copy className="w-4 h-4 text-gray-600" />
              </button>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="pt-4 border-t border-gray-200 space-y-2">
          <button onClick={copyFullLog} className="w-full btn btn-secondary flex items-center justify-center space-x-2">
            <Copy className="w-4 h-4" />
            <span>Copy Full Log</span>
          </button>
          
          {log.id && (
            <button className="w-full btn btn-secondary flex items-center justify-center space-x-2">
              <ExternalLink className="w-4 h-4" />
              <span>View in Original System</span>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default LogDetail