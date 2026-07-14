import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
    // Allow search if query is present OR if filters are set
    const hasQuery = query && query.trim() !== ''
    const hasSources = Array.isArray(filters.source) && filters.source.length > 0
    const hasTypes = Array.isArray(filters.types) && filters.types.length > 0
    const hasTemporalStart = filters.temporal_start
    const hasTemporalEnd = filters.temporal_end
    const hasBbox = filters.bbox

    if (!hasQuery && !hasSources && !hasTypes && !hasTemporalStart && !hasTemporalEnd && !hasBbox) {
      setResults([])
      setTotalResults(0)
      setError(null)
      return
    }

    const performSearch = async () => {
      setLoading(true)
      setError(null)

      try {
        const payload = {
          keywords: query.trim(),
          source: filters.source || null,
          types: filters.types || null,
          temporal_start: filters.temporal_start || null,
          temporal_end: filters.temporal_end || null,
          bbox: filters.bbox || null,
          limit: typeof filters.limit === 'number' ? filters.limit : 20,
          offset: typeof filters.offset === 'number' ? filters.offset : 0,
        }

        const response = await axios.post(`${API_BASE_URL}/api/v1/search`, payload)

        setResults(response.data.results || [])
        setTotalResults(response.data.total || 0)
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
