import { useState } from 'react'
import { Search, Filter, X } from 'lucide-react'
import { LOG_LEVELS } from '../../utils/constants'

const LogFilters = ({ onFilterChange }) => {
  const [filters, setFilters] = useState({
    search: '',
    level: '',
    service: '',
    start_date: '',
    end_date: '',
    anomaly_only: false,
  })

  const handleInputChange = (field, value) => {
    const newFilters = { ...filters, [field]: value }
    setFilters(newFilters)
    onFilterChange(newFilters)
  }

  const clearFilters = () => {
    const emptyFilters = {
      search: '',
      level: '',
      service: '',
      start_date: '',
      end_date: '',
      anomaly_only: false,
    }
    setFilters(emptyFilters)
    onFilterChange(emptyFilters)
  }

  const hasActiveFilters = Object.values(filters).some((value) => value !== '' && value !== false)

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-4">
        <div className="flex-1 relative">
          <input
            type="text"
            placeholder="Search logs..."
            value={filters.search}
            onChange={(e) => handleInputChange('search', e.target.value)}
            className="input pl-10 w-full"
          />
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
        </div>

        <select
          value={filters.level}
          onChange={(e) => handleInputChange('level', e.target.value)}
          className="input w-40"
        >
          <option value="">All Levels</option>
          {Object.values(LOG_LEVELS).map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>

        <input
          type="text"
          placeholder="Service"
          value={filters.service}
          onChange={(e) => handleInputChange('service', e.target.value)}
          className="input w-40"
        />

        {hasActiveFilters && (
          <button
            onClick={clearFilters}
            className="btn btn-secondary flex items-center space-x-2"
          >
            <X className="w-4 h-4" />
            <span>Clear</span>
          </button>
        )}
      </div>

      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-700">From:</label>
          <input
            type="datetime-local"
            value={filters.start_date}
            onChange={(e) => handleInputChange('start_date', e.target.value)}
            className="input w-52"
          />
        </div>

        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-700">To:</label>
          <input
            type="datetime-local"
            value={filters.end_date}
            onChange={(e) => handleInputChange('end_date', e.target.value)}
            className="input w-52"
          />
        </div>

        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={filters.anomaly_only}
            onChange={(e) => handleInputChange('anomaly_only', e.target.checked)}
            className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
          />
          <span className="text-sm text-gray-700">Anomalies only</span>
        </label>
      </div>
    </div>
  )
}

export default LogFilters
