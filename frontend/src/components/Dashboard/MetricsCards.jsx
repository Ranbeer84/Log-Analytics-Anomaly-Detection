import { FileText, AlertTriangle, TrendingUp, Clock } from 'lucide-react'
import { formatNumber, formatDuration } from '../../utils/formatters'

const MetricsCards = ({ overview }) => {
  const metrics = [
    {
      title: 'Total Logs',
      value: formatNumber(overview?.total_logs || 0),
      change: overview?.logs_change || 0,
      icon: FileText,
      color: 'blue',
    },
    {
      title: 'Error Rate',
      value: `${(overview?.error_rate || 0).toFixed(2)}%`,
      change: overview?.error_rate_change || 0,
      icon: AlertTriangle,
      color: 'red',
    },
    {
      title: 'Avg Response Time',
      value: formatDuration(overview?.avg_response_time || 0),
      change: overview?.response_time_change || 0,
      icon: Clock,
      color: 'green',
    },
    {
      title: 'Anomalies Detected',
      value: formatNumber(overview?.anomalies || 0),
      change: overview?.anomalies_change || 0,
      icon: TrendingUp,
      color: 'purple',
    },
  ]

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-100 text-blue-600',
      red: 'bg-red-100 text-red-600',
      green: 'bg-green-100 text-green-600',
      purple: 'bg-purple-100 text-purple-600',
    }
    return colors[color] || colors.blue
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        const isPositive = metric.change >= 0
        const changeColor = isPositive ? 'text-green-600' : 'text-red-600'

        return (
          <div key={index} className="card hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-3 rounded-lg ${getColorClasses(metric.color)}`}>
                <Icon className="w-6 h-6" />
              </div>
              {metric.change !== 0 && (
                <div className={`flex items-center text-sm font-medium ${changeColor}`}>
                  <span>{isPositive ? '+' : ''}{metric.change.toFixed(1)}%</span>
                </div>
              )}
            </div>
            <h3 className="text-gray-600 text-sm font-medium mb-1">{metric.title}</h3>
            <p className="text-3xl font-bold text-gray-900">{metric.value}</p>
          </div>
        )
      })}
    </div>
  )
}

export default MetricsCards
