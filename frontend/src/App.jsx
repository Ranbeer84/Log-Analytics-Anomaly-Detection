import { Routes, Route, Navigate } from 'react-router-dom'
import DashboardLayout from './components/Dashboard/DashboardLayout'
import LogViewer from './components/Logs/LogViewer'
import AlertList from './components/Alerts/AlertList'
import ErrorBoundary from './components/common/ErrorBoundary'

function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardLayout />} />
        <Route path="/logs" element={<LogViewer />} />
        <Route path="/alerts" element={<AlertList />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </ErrorBoundary>
  )
}

export default App