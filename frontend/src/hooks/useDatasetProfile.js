import { useEffect, useState } from 'react'
import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export function useDatasetProfile(datasetId) {
  const [profile, setProfile] = useState(null)
  const [isLoadingProfile, setIsLoadingProfile] = useState(false)
  const [profileError, setProfileError] = useState(null)

  useEffect(() => {
    if (!datasetId) {
      setProfile(null)
      setProfileError(null)
      setIsLoadingProfile(false)
      return
    }

    let cancelled = false

    const loadProfile = async () => {
      setIsLoadingProfile(true)
      setProfileError(null)

      try {
        const response = await axios.get(`${API_BASE_URL}/api/datasets/${encodeURIComponent(datasetId)}/profile`, {
          responseType: 'json',
          timeout: 30000,
        })

        if (!cancelled) {
          setProfile(response.data || null)
          setProfileError(null)
        }
      } catch (err) {
        if (!cancelled) {
          if (err.response?.status === 404) {
            setProfile(null)
            setProfileError('No stored profiling preview is available for this dataset yet.')
          } else {
            setProfile(null)
            setProfileError(
              err.message === 'Network Error'
                ? 'Unable to load the dataset profile. Is the backend running at localhost:8000?'
                : err.message || 'Unable to load dataset profile.',
            )
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoadingProfile(false)
        }
      }
    }

    loadProfile()

    return () => {
      cancelled = true
    }
  }, [datasetId])

  return { profile, isLoadingProfile, profileError }
}