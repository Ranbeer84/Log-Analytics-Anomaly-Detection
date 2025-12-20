import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from 'recharts'
import { AlertOctagon } from 'lucide-react'
import { formatTimestamp, formatRelativeTime } from '../../utils/formatters'

const AnomalyHeatmap = ({ data, timeRange }) => {
  const formattedData = data?.map((item) => ({
    ...item,
    x: new Date(item.timestamp).getTime(),
    y: item.anomaly_score * 100,
    z: item.severity || 1,
    timestamp: item.timestamp,
  })) || []

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900 mb-1">
            {formatRelativeTime(data.timestamp)}
          </p>
          <p className="text-sm text-gray-600">
            Score: <span className="font-medium">{data.y.toFixed(2)}%</span>
          </p>
          {data.message && (
            <p className="text-xs text-gray-500 mt-1 max-w-xs truncate">
              {data.message}
            </p>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <AlertOctagon className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Anomaly Detection</h3>
        </div>
        {formattedData.length > 0 && (
          <span className="badge bg-purple-100 text-purple-800">
            {formattedData.length} anomalies
          </span>
        )}
      </div>

      {formattedData.length === 0 ? (
        <div className="h-64 flex flex-col items-center justify-center text-gray-500">
          <AlertOctagon className="w-12 h-12 mb-2 opacity-50" />
          <p>No anomalies detected</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              name="Time"
              domain={['auto', 'auto']}
              tickFormatter={(value) => formatTimestamp(value, 'HH:mm')}
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Anomaly Score"
              unit="%"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Anomaly Score (%)', angle: -90, position: 'insideLeft' }}
            />
            <ZAxis type="number" dataKey="z" range={[50, 400]} />
            <Tooltip content={<CustomTooltip />} />
            <Scatter
              data={formattedData}
              fill="#9333ea"
              fillOpacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

export default AnomalyHeatmap