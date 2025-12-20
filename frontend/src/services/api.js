import axios from 'axios'
import toast from 'react-hot-toast'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const { status, data } = error.response
      
      switch (status) {
        case 401:
          toast.error('Unauthorized. Please login again.')
          // Handle logout
          break
        case 403:
          toast.error('Access forbidden')
          break
        case 404:
          toast.error('Resource not found')
          break
        case 500:
          toast.error('Server error. Please try again later.')
          break
        default:
          toast.error(data?.detail || 'An error occurred')
      }
    } else if (error.request) {
      toast.error('Network error. Please check your connection.')
    } else {
      toast.error('An unexpected error occurred')
    }
    
    return Promise.reject(error)
  }
)

// Logs API
export const logsApi = {
  getAll: (params) => api.get('/logs', { params }),
  getById: (id) => api.get(`/logs/${id}`),
  ingest: (data) => api.post('/logs/ingest', data),
  bulkIngest: (data) => api.post('/logs/bulk', data),
  search: (query) => api.post('/logs/search', query),
}

// Analytics API
export const analyticsApi = {
  getOverview: (timeRange) => api.get('/analytics/overview', { params: { time_range: timeRange } }),
  getErrorRate: (timeRange) => api.get('/analytics/error-rate', { params: { time_range: timeRange } }),
  getResponseTime: (timeRange) => api.get('/analytics/response-time', { params: { time_range: timeRange } }),
  getLogVolume: (timeRange) => api.get('/analytics/log-volume', { params: { time_range: timeRange } }),
  getTopErrors: (limit = 10) => api.get('/analytics/top-errors', { params: { limit } }),
  getAnomalies: (timeRange) => api.get('/analytics/anomalies', { params: { time_range: timeRange } }),
}

// Alerts API
export const alertsApi = {
  getAll: (params) => api.get('/alerts', { params }),
  getById: (id) => api.get(`/alerts/${id}`),
  create: (data) => api.post('/alerts', data),
  update: (id, data) => api.put(`/alerts/${id}`, data),
  delete: (id) => api.delete(`/alerts/${id}`),
  acknowledge: (id) => api.post(`/alerts/${id}/acknowledge`),
  resolve: (id) => api.post(`/alerts/${id}/resolve`),
}

// Health API
export const healthApi = {
  check: () => api.get('/health'),
  metrics: () => api.get('/health/metrics'),
}

export default api