import { useState, useEffect } from 'react'
import { Activity, Pause, Play } from 'lucide-react'
import { useWebSocket } from '../../hooks/useWebSocket'
import { formatTimestamp } from '../../utils/formatters'
import { LOG_LEVEL_COLORS } from '../../utils/constants'
import clsx from 'clsx'

const RealTimeLogs = () => {
  const [logs, setLogs] = useState([])
  const [isPaused, setIsPaused] = useState(false)
  const [maxLogs] = useState(50)
  const { eventData } = useWebSocket(['new_log'])

  useEffect(() => {
    if (!isPaused && eventData.new_log) {
      setLogs((prev) => {
        const newLogs = [eventData.new_log, ...prev]
        return newLogs.slice(0, maxLogs)
      })
    }
  }, [eventData.new_log, isPaused, maxLogs])

  const togglePause = () => {
    setIsPaused(!isPaused)
  }

  const clearLogs = () => {
    setLogs([])
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-900">Real-time Logs</h3>
          <span className="badge bg-primary-100 text-primary-800">{logs.length}</span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={togglePause}
            className="btn btn-secondary flex items-center space-x-2"
          >
            {isPaused ? (
              <>
                <Play className="w-4 h-4" />
                <span>Resume</span>
              </>
            ) : (
              <>
                <Pause className="w-4 h-4" />
                <span>Pause</span>
              </>
            )}
          </button>
          <button onClick={clearLogs} className="btn btn-secondary">
            Clear
          </button>
        </div>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto scrollbar-thin">
        {logs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Waiting for new logs...</p>
          </div>
        ) : (
          logs.map((log, index) => (
            <div
              key={`${log.timestamp}-${index}`}
              className={clsx(
                'p-3 rounded-lg border animate-fade-in',
                log.level === 'ERROR' || log.level === 'CRITICAL'
                  ? 'bg-red-50 border-red-200'
                  : 'bg-gray-50 border-gray-200'
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className={clsx('badge', LOG_LEVEL_COLORS[log.level])}>
                      {log.level}
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(log.timestamp, 'HH:mm:ss')}
                    </span>
                    {log.service && (
                      <span className="text-xs text-gray-500">
                        {log.service}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-900 truncate">{log.message}</p>
                </div>
                {log.anomaly_score && log.anomaly_score > 0.7 && (
                  <span className="badge bg-yellow-100 text-yellow-800 ml-2">
                    Anomaly
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default RealTimeLogs