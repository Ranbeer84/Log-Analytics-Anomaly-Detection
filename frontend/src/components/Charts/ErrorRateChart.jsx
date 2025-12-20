import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp } from 'lucide-react'
import { formatTimestamp } from '../../utils/formatters'

const ErrorRateChart = ({ data, timeRange }) => {
  const formattedData = data?.map((item) => ({
    ...item,
    timestamp: formatTimestamp(item.timestamp, 'HH:mm'),
    error_rate: (item.error_rate * 100).toFixed(2),
  })) || []

  return (
    <div className="card">
      <div className="flex items-center space-x-2 mb-4">
        <TrendingUp className="w-5 h-5 text-red-600" />
        <h3 className="text-lg font-semibold text-gray-900">Error Rate</h3>
      </div>

      {formattedData.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-500">
          No data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={formattedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Error Rate (%)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="error_rate"
              stroke="#ef4444"
              strokeWidth={2}
              dot={{ fill: '#ef4444', r: 4 }}
              activeDot={{ r: 6 }}
              name="Error Rate (%)"
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

export default ErrorRateChart