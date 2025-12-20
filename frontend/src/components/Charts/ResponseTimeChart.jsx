import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Clock } from 'lucide-react'
import { formatTimestamp } from '../../utils/formatters'

const ResponseTimeChart = ({ data, timeRange }) => {
  const formattedData = data?.map((item) => ({
    ...item,
    timestamp: formatTimestamp(item.timestamp, 'HH:mm'),
    avg: item.avg_response_time,
    p95: item.p95_response_time,
    p99: item.p99_response_time,
  })) || []

  return (
    <div className="card">
      <div className="flex items-center space-x-2 mb-4">
        <Clock className="w-5 h-5 text-green-600" />
        <h3 className="text-lg font-semibold text-gray-900">Response Time</h3>
      </div>

      {formattedData.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-500">
          No data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={formattedData}>
            <defs>
              <linearGradient id="colorAvg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorP95" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
            />
            <Area
              type="monotone"
              dataKey="avg"
              stroke="#22c55e"
              fillOpacity={1}
              fill="url(#colorAvg)"
              name="Avg"
            />
            <Area
              type="monotone"
              dataKey="p95"
              stroke="#f59e0b"
              fillOpacity={1}
              fill="url(#colorP95)"
              name="P95"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

export default ResponseTimeChart