import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import RecordPage from './pages/RecordPage.jsx'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/records/:recordId" element={<RecordPage />} />
      </Routes>
    </Router>
  )
}

export default App
