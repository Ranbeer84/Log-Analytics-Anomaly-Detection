import { useEffect, useState, useCallback } from 'react'
import websocketService from '../services/websocket'

export const useWebSocket = (events = []) => {
  const [isConnected, setIsConnected] = useState(false)
  const [eventData, setEventData] = useState({})

  useEffect(() => {
    // Connect to WebSocket
    websocketService.connect()
    setIsConnected(websocketService.isConnected())

    // Set up event listeners
    const unsubscribers = events.map((event) =>
      websocketService.on(event, (data) => {
        setEventData((prev) => ({
          ...prev,
          [event]: data,
        }))
      })
    )

    // Update connection status
    const connectListener = websocketService.on('connect', () => {
      setIsConnected(true)
    })

    const disconnectListener = websocketService.on('disconnect', () => {
      setIsConnected(false)
    })

    // Cleanup
    return () => {
      unsubscribers.forEach((unsub) => unsub())
      connectListener()
      disconnectListener()
    }
  }, [events.join(',')])

  const subscribe = useCallback((channel) => {
    websocketService.subscribe(channel)
  }, [])

  const unsubscribe = useCallback((channel) => {
    websocketService.unsubscribe(channel)
  }, [])

  return {
    isConnected,
    eventData,
    subscribe,
    unsubscribe,
  }
}