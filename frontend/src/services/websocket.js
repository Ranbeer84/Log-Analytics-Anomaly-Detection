import { io } from 'socket.io-client'
import toast from 'react-hot-toast'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

class WebSocketService {
  constructor() {
    this.socket = null
    this.listeners = new Map()
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
  }

  connect() {
    if (this.socket?.connected) {
      console.log('WebSocket already connected')
      return
    }

    this.socket = io(WS_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: this.maxReconnectAttempts,
    })

    this.socket.on('connect', () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      toast.success('Connected to real-time updates')
    })

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason)
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, reconnect manually
        this.socket.connect()
      }
    })

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      this.reconnectAttempts++
      
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        toast.error('Failed to connect to real-time updates')
      }
    })

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error)
      toast.error('Real-time connection error')
    })

    // Set up default event listeners
    this.setupDefaultListeners()
  }

  setupDefaultListeners() {
    this.socket.on('new_log', (data) => {
      this.emit('new_log', data)
    })

    this.socket.on('new_anomaly', (data) => {
      this.emit('new_anomaly', data)
      toast.error(`Anomaly detected: ${data.message}`, { duration: 6000 })
    })

    this.socket.on('new_alert', (data) => {
      this.emit('new_alert', data)
      toast.error(`Alert: ${data.title}`, { duration: 6000 })
    })
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
      this.listeners.clear()
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event).push(callback)

    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(event)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  emit(event, data) {
    const callbacks = this.listeners.get(event)
    if (callbacks) {
      callbacks.forEach((callback) => callback(data))
    }
  }

  subscribe(channel) {
    if (this.socket?.connected) {
      this.socket.emit('subscribe', { channel })
    }
  }

  unsubscribe(channel) {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { channel })
    }
  }

  isConnected() {
    return this.socket?.connected || false
  }
}

// Create singleton instance
const websocketService = new WebSocketService()

export default websocketService