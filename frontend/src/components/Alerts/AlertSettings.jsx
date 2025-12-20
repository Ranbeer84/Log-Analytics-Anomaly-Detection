import { useState } from 'react'
import { X, Plus, Trash2 } from 'lucide-react'
import { alertsApi } from '../../services/api'
import { ALERT_SEVERITIES } from '../../utils/constants'
import toast from 'react-hot-toast'

const AlertSettings = ({ onClose, onSave }) => {
  const [rules, setRules] = useState([
    {
      id: Date.now(),
      name: '',
      condition: 'error_rate',
      threshold: 5,
      severity: 'HIGH',
      enabled: true,
    },
  ])

  const addRule = () => {
    setRules([
      ...rules,
      {
        id: Date.now(),
        name: '',
        condition: 'error_rate',
        threshold: 5,
        severity: 'HIGH',
        enabled: true,
      },
    ])
  }

  const removeRule = (id) => {
    setRules(rules.filter((rule) => rule.id !== id))
  }

  const updateRule = (id, field, value) => {
    setRules(
      rules.map((rule) =>
        rule.id === id ? { ...rule, [field]: value } : rule
      )
    )
  }

  const handleSave = async () => {
    try {
      // Validate rules
      for (const rule of rules) {
        if (!rule.name) {
          toast.error('Please provide a name for all rules')
          return
        }
      }

      // Save rules to backend
      await Promise.all(
        rules.map((rule) => alertsApi.create(rule))
      )

      toast.success('Alert rules saved successfully')
      onSave()
      onClose()
    } catch (error) {
      toast.error('Failed to save alert rules')
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-gray-900">Alert Settings</h3>
            <p className="text-sm text-gray-600 mt-1">
              Configure alert rules and thresholds
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Rules */}
          <div className="space-y-4">
            {rules.map((rule, index) => (
              <div key={rule.id} className="card bg-gray-50">
                <div className="flex items-start justify-between mb-4">
                  <h4 className="font-medium text-gray-900">Rule {index + 1}</h4>
                  <button
                    onClick={() => removeRule(rule.id)}
                    className="p-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {/* Rule Name */}
                  <div className="col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Rule Name
                    </label>
                    <input
                      type="text"
                      value={rule.name}
                      onChange={(e) => updateRule(rule.id, 'name', e.target.value)}
                      placeholder="e.g., High Error Rate Alert"
                      className="input"
                    />
                  </div>

                  {/* Condition */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Condition
                    </label>
                    <select
                      value={rule.condition}
                      onChange={(e) => updateRule(rule.id, 'condition', e.target.value)}
                      className="input"
                    >
                      <option value="error_rate">Error Rate</option>
                      <option value="response_time">Response Time</option>
                      <option value="log_volume">Log Volume</option>
                      <option value="anomaly_score">Anomaly Score</option>
                    </select>
                  </div>

                  {/* Threshold */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Threshold
                    </label>
                    <input
                      type="number"
                      value={rule.threshold}
                      onChange={(e) => updateRule(rule.id, 'threshold', Number(e.target.value))}
                      className="input"
                      step="0.1"
                    />
                  </div>

                  {/* Severity */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Severity
                    </label>
                    <select
                      value={rule.severity}
                      onChange={(e) => updateRule(rule.id, 'severity', e.target.value)}
                      className="input"
                    >
                      {Object.values(ALERT_SEVERITIES).map((severity) => (
                        <option key={severity} value={severity}>
                          {severity}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Enabled */}
                  <div className="flex items-center">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={rule.enabled}
                        onChange={(e) => updateRule(rule.id, 'enabled', e.target.checked)}
                        className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      />
                      <span className="text-sm text-gray-700">Enabled</span>
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Add Rule Button */}
          <button
            onClick={addRule}
            className="btn btn-secondary w-full flex items-center justify-center space-x-2"
          >
            <Plus className="w-5 h-5" />
            <span>Add Rule</span>
          </button>

          {/* Notification Settings */}
          <div className="card bg-blue-50 border-blue-200">
            <h4 className="font-medium text-gray-900 mb-3">Notification Channels</h4>
            <div className="space-y-3">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700">Email Notifications</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  defaultChecked
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700">Slack Notifications</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="text-sm text-gray-700">PagerDuty Integration</span>
              </label>
            </div>
          </div>

          {/* Actions */}
          <div className="pt-4 border-t border-gray-200 flex justify-end space-x-3">
            <button onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button onClick={handleSave} className="btn btn-primary">
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AlertSettings