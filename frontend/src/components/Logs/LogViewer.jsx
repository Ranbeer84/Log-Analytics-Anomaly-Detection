import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import Navbar from '../common/Navbar'
import Sidebar from '../common/Sidebar'
import LogFilters from './LogFilters'
import LogDetail from './LogDetail'
import { useLogs } from '../../hooks/useLogs'
import { formatTimestamp } from '../../utils/formatters'
import { LOG_LEVEL_COLORS } from '../../utils/constants'
import LoadingSpinner from '../common/LoadingSpinner'
import clsx from 'clsx'

const LogViewer = () => {
  const [selectedLog, setSelectedLog] = useState(null)
  const { logs, loading, pagination, updateFilters, nextPage, prevPage, goToPage } = useLogs()

  const handleFilterChange = (newFilters) => {
    updateFilters(newFilters)
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        
        <main className="flex-1 overflow-hidden flex">
          <div className="flex-1 flex flex-col">
            <div className="p-6 border-b border-gray-200 bg-white">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Log Viewer</h2>
              <LogFilters onFilterChange={handleFilterChange} />
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              {loading ? (
                <LoadingSpinner text="Loading logs..." />
              ) : logs.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  No logs found matching your filters
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    {logs.map((log) => (
                      <div
                        key={log.id || log.timestamp}
                        onClick={() => setSelectedLog(log)}
                        className={clsx(
                          'card cursor-pointer transition-all hover:shadow-md',
                          selectedLog?.id === log.id && 'ring-2 ring-primary-500'
                        )}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className={clsx('badge', LOG_LEVEL_COLORS[log.level])}>
                                {log.level}
                              </span>
                              <span className="text-sm text-gray-500">
                                {formatTimestamp(log.timestamp)}
                              </span>
                              {log.service && (
                                <span className="badge bg-gray-100 text-gray-800">
                                  {log.service}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-900">{log.message}</p>
                          </div>
                          {log.anomaly_score && log.anomaly_score > 0.7 && (
                            <span className="badge bg-yellow-100 text-yellow-800 ml-2">
                              Anomaly
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Pagination */}
                  <div className="mt-6 flex items-center justify-between">
                    <div className="text-sm text-gray-700">
                      Showing {((pagination.page - 1) * pagination.limit) + 1} to{' '}
                      {Math.min(pagination.page * pagination.limit, pagination.total)} of{' '}
                      {pagination.total} logs
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={prevPage}
                        disabled={pagination.page === 1}
                        className="btn btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <span className="text-sm text-gray-700">
                        Page {pagination.page} of {pagination.totalPages}
                      </span>
                      <button
                        onClick={nextPage}
                        disabled={pagination.page === pagination.totalPages}
                        className="btn btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>

          {selectedLog && (
            <LogDetail log={selectedLog} onClose={() => setSelectedLog(null)} />
          )}
        </main>
      </div>
    </div>
  )
}

export default LogViewer