import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

/**
 * Hook to search datasets from the FastAPI backend.
 * @param {string} query - The search query
 * @param {object} filters - Optional filter object
 * @returns {object} { results, loading, error, totalResults }
 */
export function useSearch(query, filters = {}) {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [totalResults, setTotalResults] = useState(0)

  useEffect(() => {
    if (!query || query.trim() === '') {
      setResults([])
      setTotalResults(0)
      setError(null)
      return
    }

    const performSearch = async () => {
      setLoading(true)
      setError(null)

      try {
        const response = await axios.post(`${API_BASE_URL}/search`, {
          query: query.trim(),
          filters: Object.keys(filters).length > 0 ? filters : null,
        })

        setResults(response.data.results || [])
        setTotalResults(response.data.total_results || 0)
      } catch (err) {
        console.error('Search error:', err)
        setError(
          err.message === 'Network Error'
            ? 'Unable to connect to the search backend. Is localhost:8000 running?'
            : err.message || 'An error occurred during search.',
        )
        setResults([])
        setTotalResults(0)
      } finally {
        setLoading(false)
      }
    }

    performSearch()
  }, [query, filters])

  return { results, loading, error, totalResults }
}
