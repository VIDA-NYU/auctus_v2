import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { MapContainer, Rectangle, TileLayer, useMap, useMapEvents } from 'react-leaflet'
import { useSearch } from '../hooks/useSearch'
import { ResultSnippet } from '../components/ResultSnippet'
import SearchBar from '../components/ui/SearchBar'
import FilterDropdown from '../components/ui/FilterDropdown'
import DatasetResults from '../components/DatasetResults'

const FILTER_CATEGORIES = [
  { id: 'source', label: 'Source' },
  { id: 'dataType', label: 'Data Types' },
  { id: 'date', label: 'Temporal' },
  { id: 'location', label: 'Spatial' },
]

const SOURCE_OPTIONS = [
  'Socrata',
  'CKAN',
  'Local',
  'NYC Open Data',
  'World Bank',
  'Zenodo',
  'CAL FIRE',
  'Inside Airbnb',
  'European Central Bank',
  'Our World in Data',
]

const TYPE_OPTIONS = ['spatial', 'numerical', 'temporal', 'categorical']

function MapClickTarget({ onBBoxChange }) {
  const clicksRef = useRef([])
  useMap()

  useMapEvents({
    click(e) {
      clicksRef.current.push([e.latlng.lng, e.latlng.lat])
      if (clicksRef.current.length === 2) {
        const [a, b] = clicksRef.current
        const minLon = Math.min(a[0], b[0])
        const minLat = Math.min(a[1], b[1])
        const maxLon = Math.max(a[0], b[0])
        const maxLat = Math.max(a[1], b[1])
        onBBoxChange([minLon, minLat, maxLon, maxLat])
        clicksRef.current = []
      }
    },
  })

  return null
}

function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [openDropdown, setOpenDropdown] = useState(null)
  const [editKeywords, setEditKeywords] = useState('')
  const [editFilters, setEditFilters] = useState({
    source: [],
    types: [],
    temporal_start: '',
    temporal_end: '',
    bbox: null,
  })
  const [selectedDataset, setSelectedDataset] = useState(null)

  const query = useMemo(() => searchParams.get('q') || '', [searchParams])

  const filtersFromParams = useMemo(() => {
    const f = {}
    const sources = searchParams.getAll('source')
    const types = searchParams.getAll('types')
    const temporalStart = searchParams.get('temporal_start')
    const temporalEnd = searchParams.get('temporal_end')
    const bboxStr = searchParams.get('bbox')

    if (sources && sources.length > 0) f.source = sources
    if (types && types.length > 0) f.types = types
    if (temporalStart) f.temporal_start = temporalStart
    if (temporalEnd) f.temporal_end = temporalEnd
    if (bboxStr) {
      const parts = bboxStr.split(',').map((v) => parseFloat(v))
      if (parts.length === 4 && parts.every((n) => !Number.isNaN(n))) {
        f.bbox = parts
      }
    }
    return f
  }, [searchParams.toString()])

  useEffect(() => {
    setEditKeywords(query)
    setEditFilters({
      source: filtersFromParams.source || [],
      types: filtersFromParams.types || [],
      temporal_start: filtersFromParams.temporal_start || '',
      temporal_end: filtersFromParams.temporal_end || '',
      bbox: filtersFromParams.bbox || null,
    })
  }, [query, filtersFromParams])

  const { results, loading, error, totalResults } = useSearch(query, filtersFromParams)

  useEffect(() => {
    if (Array.isArray(results) && results.length > 0) {
      setSelectedDataset(results[0])
    } else {
      setSelectedDataset(null)
    }
  }, [results])

  const handleSearch = () => {
    const params = new URLSearchParams()

    if (editKeywords.trim()) {
      params.set('q', editKeywords.trim())
    } else {
      params.set('q', '')
    }

    if (Array.isArray(editFilters.source) && editFilters.source.length > 0) {
      editFilters.source.forEach((s) => params.append('source', s))
    }
    if (Array.isArray(editFilters.types) && editFilters.types.length > 0) {
      editFilters.types.forEach((t) => params.append('types', t))
    }
    if (editFilters.temporal_start) params.set('temporal_start', editFilters.temporal_start)
    if (editFilters.temporal_end) params.set('temporal_end', editFilters.temporal_end)
    if (editFilters.bbox) params.set('bbox', editFilters.bbox.join(','))

    setSearchParams(params)
  }

  const getActiveCount = (filterId) => {
    switch (filterId) {
      case 'source':
        return Array.isArray(editFilters.source) ? editFilters.source.length : 0
      case 'dataType':
        return Array.isArray(editFilters.types) ? editFilters.types.length : 0
      case 'date':
        return (editFilters.temporal_start ? 1 : 0) + (editFilters.temporal_end ? 1 : 0)
      case 'location':
        return editFilters.bbox ? 1 : 0
      default:
        return 0
    }
  }

  const updateEditFilter = (key, value) => {
    setEditFilters((s) => ({ ...s, [key]: value }))
  }

  const toggleFilter = (id) => {
    setOpenDropdown((current) => (current === id ? null : id))
  }

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-white">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <SearchBar
            value={editKeywords}
            onChange={(e) => setEditKeywords(e.target.value)}
            onSearch={handleSearch}
            isCompact={true}
            showLogo={true}
            placeholder="Search datasets..."
          />

          <div className="mt-4 flex flex-wrap items-center gap-2">
            {FILTER_CATEGORIES.map(({ id, label }) => (
              <FilterDropdown
                key={id}
                label={label}
                activeCount={getActiveCount(id)}
                isOpen={openDropdown === id}
                onToggle={() => toggleFilter(id)}
              >
                {id === 'source' && (
                  <div className="grid min-w-72 grid-cols-2 gap-3">
                    {SOURCE_OPTIONS.map((opt) => (
                      <label key={opt} className="inline-flex cursor-pointer items-center gap-2 text-sm hover:text-[#64518c]">
                        <input
                          type="checkbox"
                          checked={Array.isArray(editFilters.source) && editFilters.source.includes(opt)}
                          onChange={() => {
                            const arr = Array.isArray(editFilters.source) ? [...editFilters.source] : []
                            const idx = arr.indexOf(opt)
                            if (idx === -1) arr.push(opt)
                            else arr.splice(idx, 1)
                            updateEditFilter('source', arr)
                          }}
                          className="h-4 w-4 cursor-pointer rounded border-slate-200 text-[#64518c] focus:ring-[#64518c]"
                        />
                        <span className="text-slate-700">{opt}</span>
                      </label>
                    ))}
                  </div>
                )}

                {id === 'dataType' && (
                  <div className="grid min-w-72 grid-cols-2 gap-3">
                    {TYPE_OPTIONS.map((opt) => (
                      <label key={opt} className="inline-flex cursor-pointer items-center gap-2 text-sm hover:text-[#64518c]">
                        <input
                          type="checkbox"
                          checked={Array.isArray(editFilters.types) && editFilters.types.includes(opt)}
                          onChange={() => {
                            const arr = Array.isArray(editFilters.types) ? [...editFilters.types] : []
                            const idx = arr.indexOf(opt)
                            if (idx === -1) arr.push(opt)
                            else arr.splice(idx, 1)
                            updateEditFilter('types', arr)
                          }}
                          className="h-4 w-4 cursor-pointer rounded border-slate-200 text-[#64518c] focus:ring-[#64518c]"
                        />
                        <span className="text-slate-700 capitalize">{opt}</span>
                      </label>
                    ))}
                  </div>
                )}

                {id === 'date' && (
                  <div className="flex min-w-72 flex-col gap-3">
                    <div>
                      <label className="mb-1 block text-xs font-medium text-slate-600">Start Date</label>
                      <input
                        type="date"
                        value={editFilters.temporal_start}
                        onChange={(e) => updateEditFilter('temporal_start', e.target.value)}
                        className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-[#64518c] focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-xs font-medium text-slate-600">End Date</label>
                      <input
                        type="date"
                        value={editFilters.temporal_end}
                        onChange={(e) => updateEditFilter('temporal_end', e.target.value)}
                        className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-[#64518c] focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
                      />
                    </div>
                  </div>
                )}

                {id === 'location' && (
                  <div className="flex min-w-72 flex-col gap-3">
                    <div className="h-56 w-full overflow-hidden rounded-xl border border-slate-200">
                      <MapContainer
                        center={[40.7128, -74.006]}
                        zoom={10}
                        className="h-full w-full"
                        whenReady={(mapInstance) => {
                          setTimeout(() => mapInstance.target.invalidateSize(), 100)
                        }}
                      >
                        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                        <MapClickTarget onBBoxChange={(bb) => updateEditFilter('bbox', bb)} />
                        {editFilters.bbox ? (
                          <Rectangle
                            bounds={[
                              [editFilters.bbox[1], editFilters.bbox[0]],
                              [editFilters.bbox[3], editFilters.bbox[2]],
                            ]}
                          />
                        ) : null}
                      </MapContainer>
                    </div>
                    <p className="px-1 text-xs text-slate-500">
                      Click two points on the map to draw a bounding box.
                    </p>
                  </div>
                )}
              </FilterDropdown>
            ))}
          </div>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 bg-white px-0 lg:px-6 py-8">
        <div className="min-h-0 w-full">
          {loading ? (
            <div className="space-y-4">
              <p className="text-center text-slate-600">
                Searching {totalResults.toLocaleString()}+ datasets...
              </p>
              <div className="space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="h-24 animate-pulse rounded-lg bg-slate-100 p-4" />
                ))}
              </div>
            </div>
          ) : error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center">
              <p className="text-sm font-medium text-red-900">{error}</p>
              <p className="mt-2 text-xs text-red-700">
                Please ensure the FastAPI backend is running at localhost:8000
              </p>
            </div>
          ) : results.length === 0 && !loading ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-12 text-center">
              <p className="text-lg font-medium text-slate-900">No datasets found</p>
              <p className="mt-2 text-sm text-slate-600">
                Try adjusting your search terms or filters to find what you're looking for.
              </p>
            </div>
          ) : (
            <div className="flex min-h-0 h-full flex-col">
              <p className="mb-2 text-sm text-slate-600">
                Found <span className="font-semibold text-slate-900">{totalResults}</span> dataset
                {totalResults !== 1 ? 's' : ''} for "<span className="font-semibold text-[#64518c]">{query}</span>"
              </p>
              <div className="min-h-0 flex-1">
              <DatasetResults results={results} selectedDataset={selectedDataset} setSelectedDataset={setSelectedDataset} />
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

export default ResultsPage
