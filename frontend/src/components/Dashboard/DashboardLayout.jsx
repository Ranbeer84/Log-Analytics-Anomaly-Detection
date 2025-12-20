import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import Navbar from '../common/Navbar'
import Sidebar from '../common/Sidebar'
import MetricsCards from './MetricsCards'
import RealTimeLogs from './RealTimeLogs'
import ErrorRateChart from '../Charts/ErrorRateChart'
import ResponseTimeChart from '../Charts/ResponseTimeChart'
import LogVolumeChart from '../Charts/LogVolumeChart'
import AnomalyHeatmap from '../Charts/AnomalyHeatmap'
import { useAnalytics } from '../../hooks/useAnalytics'
import { TIME_RANGES, REFRESH_INTERVALS } from '../../utils/constants'
import LoadingSpinner from '../common/LoadingSpinner'

const DashboardLayout = () => {
  const [timeRange, setTimeRange] = useState('24h')
  const [refreshInterval, setRefreshInterval] = useState(30000)
  const { overview, errorRate, responseTime, logVolume, anomalies, loading, refresh } = useAnalytics(timeRange)

  const handleRefresh = () => {
    refresh()
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        
        <main className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Header with controls */}
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-gray-900">Dashboard Overview</h2>
              <div className="flex items-center space-x-4">
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="input w-40"
                >
                  {TIME_RANGES.map((range) => (
                    <option key={range.value} value={range.value}>
                      {range.label}
                    </option>
                  ))}
                </select>
                
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  className="input w-32"
                >
                  {REFRESH_INTERVALS.map((interval) => (
                    <option key={interval.value} value={interval.value}>
                      {interval.label}
                    </option>
                  ))}
                </select>

                <button
                  onClick={handleRefresh}
                  disabled={loading}
                  className="btn btn-secondary flex items-center space-x-2"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  <span>Refresh</span>
                </button>
              </div>
            </div>

            {/* Metrics Cards */}
            {loading && !overview ? (
              <LoadingSpinner text="Loading dashboard..." />
            ) : (
              <>
                <MetricsCards overview={overview} />

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <ErrorRateChart data={errorRate} timeRange={timeRange} />
                  <ResponseTimeChart data={responseTime} timeRange={timeRange} />
                  <LogVolumeChart data={logVolume} timeRange={timeRange} />
                  <AnomalyHeatmap data={anomalies} timeRange={timeRange} />
                </div>

                {/* Real-time Logs */}
                <RealTimeLogs />
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default DashboardLayout