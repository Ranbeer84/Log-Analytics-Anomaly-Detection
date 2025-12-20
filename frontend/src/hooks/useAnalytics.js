import { useState, useEffect, useCallback } from 'react'
import { analyticsApi } from '../services/api'
import toast from 'react-hot-toast'

export const useAnalytics = (timeRange = '24h') => {
  const [overview, setOverview] = useState(null)
  const [errorRate, setErrorRate] = useState([])
  const [responseTime, setResponseTime] = useState([])
  const [logVolume, setLogVolume] = useState([])
  const [topErrors, setTopErrors] = useState([])
  const [anomalies, setAnomalies] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchOverview = useCallback(async (range = timeRange) => {
    try {
      const response = await analyticsApi.getOverview(range)
      setOverview(response.data)
    } catch (err) {
      console.error('Failed to fetch overview:', err)
      toast.error('Failed to fetch overview data')
    }
  }, [timeRange])

  const fetchErrorRate = useCallback(async (range = timeRange) => {
    try {
      const response = await analyticsApi.getErrorRate(range)
      setErrorRate(response.data)
    } catch (err) {
      console.error('Failed to fetch error rate:', err)
    }
  }, [timeRange])

  const fetchResponseTime = useCallback(async (range = timeRange) => {
    try {
      const response = await analyticsApi.getResponseTime(range)
      setResponseTime(response.data)
    } catch (err) {
      console.error('Failed to fetch response time:', err)
    }
  }, [timeRange])

  const fetchLogVolume = useCallback(async (range = timeRange) => {
    try {
      const response = await analyticsApi.getLogVolume(range)
      setLogVolume(response.data)
    } catch (err) {
      console.error('Failed to fetch log volume:', err)
    }
  }, [timeRange])

  const fetchTopErrors = useCallback(async (limit = 10) => {
    try {
      const response = await analyticsApi.getTopErrors(limit)
      setTopErrors(response.data)
    } catch (err) {
      console.error('Failed to fetch top errors:', err)
    }
  }, [])

  const fetchAnomalies = useCallback(async (range = timeRange) => {
    try {
      const response = await analyticsApi.getAnomalies(range)
      setAnomalies(response.data)
    } catch (err) {
      console.error('Failed to fetch anomalies:', err)
    }
  }, [timeRange])

  const fetchAll = useCallback(async (range = timeRange) => {
    setLoading(true)
    setError(null)

    try {
      await Promise.all([
        fetchOverview(range),
        fetchErrorRate(range),
        fetchResponseTime(range),
        fetchLogVolume(range),
        fetchTopErrors(),
        fetchAnomalies(range),
      ])
    } catch (err) {
      setError(err.message)
      toast.error('Failed to fetch analytics data')
    } finally {
      setLoading(false)
    }
  }, [timeRange, fetchOverview, fetchErrorRate, fetchResponseTime, fetchLogVolume, fetchTopErrors, fetchAnomalies])

  const refresh = useCallback(() => {
    fetchAll(timeRange)
  }, [fetchAll, timeRange])

  useEffect(() => {
    fetchAll(timeRange)
  }, [timeRange])

  return {
    overview,
    errorRate,
    responseTime,
    logVolume,
    topErrors,
    anomalies,
    loading,
    error,
    refresh,
    fetchAll,
  }
}