import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import SearchBar from '../components/ui/SearchBar'
import FilterDropdown from '../components/ui/FilterDropdown'

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

function Home() {
  const navigate = useNavigate()

  const [stagedFilters, setStagedFilters] = useState({
    keywords: '',
    source: [],
    types: [],
    temporal_start: '',
    temporal_end: '',
    bbox: null,
  })
  const [openDropdown, setOpenDropdown] = useState(null)

  // MapClickTarget: Properly tracks two clicks for bounding box drawing using useMapEvents
  function MapClickTarget({ onBBoxChange }) {
    const clicksRef = useRef([])
    const mapInstance = useMap()

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
          clicksRef.current = [] // Reset for redraw
        }
      },
    })

    return null
  }

  const handleSearch = () => {
    // Check if at least ONE filter is set
    const hasKeywords = stagedFilters.keywords && stagedFilters.keywords.trim() !== ''
    const hasSources = Array.isArray(stagedFilters.source) && stagedFilters.source.length > 0
    const hasTypes = Array.isArray(stagedFilters.types) && stagedFilters.types.length > 0
    const hasTemporalStart = stagedFilters.temporal_start
    const hasTemporalEnd = stagedFilters.temporal_end
    const hasBbox = stagedFilters.bbox

    if (!hasKeywords && !hasSources && !hasTypes && !hasTemporalStart && !hasTemporalEnd && !hasBbox) {
      return // No filters set, don't proceed
    }

    const params = new URLSearchParams()
    if (hasKeywords) {
      params.set('q', stagedFilters.keywords.trim())
    } else {
      params.set('q', '') // Allow empty keywords for filter-only search
    }
    if (Array.isArray(stagedFilters.source) && stagedFilters.source.length > 0) {
      stagedFilters.source.forEach((s) => params.append('source', s))
    }
    if (Array.isArray(stagedFilters.types) && stagedFilters.types.length > 0) {
      stagedFilters.types.forEach((t) => params.append('types', t))
    }
    if (stagedFilters.temporal_start) params.set('temporal_start', stagedFilters.temporal_start)
    if (stagedFilters.temporal_end) params.set('temporal_end', stagedFilters.temporal_end)
    if (stagedFilters.bbox) params.set('bbox', stagedFilters.bbox.join(','))

    navigate(`/results?${params.toString()}`)
  }

  const getActiveCount = (filterId) => {
    switch (filterId) {
      case 'source':
        return Array.isArray(stagedFilters.source) ? stagedFilters.source.length : 0
      case 'dataType':
        return Array.isArray(stagedFilters.types) ? stagedFilters.types.length : 0
      case 'date':
        return (stagedFilters.temporal_start ? 1 : 0) + (stagedFilters.temporal_end ? 1 : 0)
      case 'location':
        return stagedFilters.bbox ? 1 : 0
      default:
        return 0
    }
  }

  const updateStaged = (key, value) => {
    setStagedFilters((s) => ({ ...s, [key]: value }))
  }

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-10 text-slate-800">
      <div className="mx-auto flex w-full max-w-5xl flex-col items-center gap-8 pt-20 sm:pt-24 pb-24">
        {/* Header Section */}
        <section className="flex w-full flex-col items-center text-center">
          <img
            src="/auctus-logo.min.56edd9aa.svg"
            alt="Auctus logo"
            className="h-14 w-14 sm:h-16 sm:w-16 object-contain"
          />
          <h1 className="mt-6 text-5xl font-semibold tracking-[0.04em] text-[#64518c] sm:text-6xl">
            Auctus
          </h1>
          <p className="mt-3 text-sm font-medium tracking-[0.18em] text-slate-500 uppercase sm:text-base">
            Dataset Discovery Engine
          </p>
        </section>

        {/* Search Section */}
        <section className="w-full max-w-3xl space-y-6">
          {/* Search Bar (no logo on home page) */}
          <SearchBar
            value={stagedFilters.keywords}
            onChange={(e) => updateStaged('keywords', e.target.value)}
            onSearch={handleSearch}
            isCompact={false}
            showLogo={false}
            placeholder="Search for datasets..."
          />

          {/* Filter Dropdowns */}
          <div className="flex flex-wrap items-center justify-center gap-3">
            {FILTER_CATEGORIES.map(({ id, label }) => (
              <FilterDropdown
                key={id}
                label={label}
                activeCount={getActiveCount(id)}
                isOpen={openDropdown === id}
                onToggle={() => setOpenDropdown(openDropdown === id ? null : id)}
              >
                {/* Source Filter */}
                {id === 'source' && (
                  <div className="grid grid-cols-2 gap-3 min-w-72">
                    {SOURCE_OPTIONS.map((opt) => (
                      <label key={opt} className="inline-flex items-center gap-2 text-sm cursor-pointer hover:text-amber-600">
                        <input
                          type="checkbox"
                          checked={Array.isArray(stagedFilters.source) && stagedFilters.source.includes(opt)}
                          onChange={() => {
                            const arr = Array.isArray(stagedFilters.source) ? [...stagedFilters.source] : []
                            const idx = arr.indexOf(opt)
                            if (idx === -1) arr.push(opt)
                            else arr.splice(idx, 1)
                            updateStaged('source', arr)
                          }}
                          className="h-4 w-4 rounded border-gray-300 text-amber-600 focus:ring-amber-500 cursor-pointer"
                        />
                        <span className="text-slate-700">{opt}</span>
                      </label>
                    ))}
                  </div>
                )}

                {/* Data Type Filter */}
                {id === 'dataType' && (
                  <div className="grid grid-cols-2 gap-3 min-w-72">
                    {TYPE_OPTIONS.map((opt) => (
                      <label key={opt} className="inline-flex items-center gap-2 text-sm cursor-pointer hover:text-amber-600">
                        <input
                          type="checkbox"
                          checked={Array.isArray(stagedFilters.types) && stagedFilters.types.includes(opt)}
                          onChange={() => {
                            const arr = Array.isArray(stagedFilters.types) ? [...stagedFilters.types] : []
                            const idx = arr.indexOf(opt)
                            if (idx === -1) arr.push(opt)
                            else arr.splice(idx, 1)
                            updateStaged('types', arr)
                          }}
                          className="h-4 w-4 rounded border-gray-300 text-amber-600 focus:ring-amber-500 cursor-pointer"
                        />
                        <span className="text-slate-700 capitalize">{opt}</span>
                      </label>
                    ))}
                  </div>
                )}

                {/* Temporal Filter */}
                {id === 'date' && (
                  <div className="flex flex-col gap-3 min-w-72">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Start Date</label>
                      <input
                        type="date"
                        value={stagedFilters.temporal_start}
                        onChange={(e) => updateStaged('temporal_start', e.target.value)}
                        className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-500/20"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">End Date</label>
                      <input
                        type="date"
                        value={stagedFilters.temporal_end}
                        onChange={(e) => updateStaged('temporal_end', e.target.value)}
                        className="w-full rounded border border-gray-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-amber-500 focus:outline-none focus:ring-2 focus:ring-amber-500/20"
                      />
                    </div>
                  </div>
                )}

                {/* Spatial Filter */}
                {id === 'location' && (
                  <div className="flex flex-col gap-2 min-w-max">
                    <div className="w-80 h-56 rounded-md overflow-hidden border border-gray-200">
                      <MapContainer
                        center={[40.7128, -74.006]}
                        zoom={10}
                        className="h-full w-full"
                        whenReady={(mapInstance) => {
                          setTimeout(() => mapInstance.target.invalidateSize(), 100)
                        }}
                      >
                        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                        <MapClickTarget onBBoxChange={(bb) => updateStaged('bbox', bb)} />
                        {stagedFilters.bbox ? (
                          <Rectangle
                            bounds={[
                              [stagedFilters.bbox[1], stagedFilters.bbox[0]],
                              [stagedFilters.bbox[3], stagedFilters.bbox[2]],
                            ]}
                          />
                        ) : null}
                      </MapContainer>
                    </div>
                    <p className="text-xs text-gray-500">Click two points to draw a bounding box</p>
                  </div>
                )}
              </FilterDropdown>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}

export default Home
