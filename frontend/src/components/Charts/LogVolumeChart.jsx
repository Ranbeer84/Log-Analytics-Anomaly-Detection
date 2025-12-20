import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { BarChart3 } from 'lucide-react'
import { formatTimestamp } from '../../utils/formatters'

const LogVolumeChart = ({ data, timeRange }) => {
  const formattedData = data?.map((item) => ({
    ...item,
    timestamp: formatTimestamp(item.timestamp, 'HH:mm'),
  })) || []

  return (
    <div className="card">
      <div className="flex items-center space-x-2 mb-4">
        <BarChart3 className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-900">Log Volume by Level</h3>
      </div>

      {formattedData.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-500">
          No data available
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={formattedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="timestamp"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar dataKey="info" stackId="a" fill="#3b82f6" name="INFO" />
            <Bar dataKey="warning" stackId="a" fill="#f59e0b" name="WARNING" />
            <Bar dataKey="error" stackId="a" fill="#ef4444" name="ERROR" />
            <Bar dataKey="critical" stackId="a" fill="#9333ea" name="CRITICAL" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

export default LogVolumeChart