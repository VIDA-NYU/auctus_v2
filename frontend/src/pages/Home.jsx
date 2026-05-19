import { useMemo, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Calendar, Database, FileText, MapPin, Search } from 'lucide-react'
import { MapContainer, TileLayer, Rectangle, useMapEvents, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const filters = [
  {
    id: 'date',
    label: 'Date',
    icon: Calendar,
    placeholder: 'Select a date range',
    inputType: 'text',
  },
  {
    id: 'location',
    label: 'Location',
    icon: MapPin,
    placeholder: 'Enter a country, region, or city',
    inputType: 'text',
  },
  {
    id: 'source',
    label: 'Source',
    icon: Database,
    placeholder: 'Choose a source type',
    inputType: 'text',
  },
  {
    id: 'dataType',
    label: 'Data Type',
    icon: FileText,
    placeholder: 'Choose a data category',
    inputType: 'text',
  },
]

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
  const [activeFilter, setActiveFilter] = useState(null)

  const activeFilterConfig = useMemo(
    () => filters.find((filter) => filter.id === activeFilter),
    [activeFilter],
  )

  const ActiveFilterIcon = activeFilterConfig?.icon

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

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      handleSearch()
    }
  }

  const toggleFilter = (filterId) => {
    setActiveFilter((current) => (current === filterId ? null : filterId))
  }

  const updateStaged = (key, value) => {
    setStagedFilters((s) => ({ ...s, [key]: value }))
  }

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-10 text-slate-800">
      <div className="mx-auto flex w-full max-w-5xl flex-col items-center gap-8 pt-20 sm:pt-24 pb-24">
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

        <section className="w-full max-w-3xl space-y-4">
          <label
            htmlFor="landing-search"
            className="sr-only"
          >
            Search datasets
          </label>
          <div className="flex items-center gap-3 rounded-full border border-slate-200 bg-white px-5 py-4 shadow-[0_12px_30px_-18px_rgba(15,23,42,0.45)] transition-shadow focus-within:shadow-[0_16px_40px_-18px_rgba(100,81,140,0.35)]">
            {/* <Search className="h-5 w-5 shrink-0 text-slate-400" aria-hidden="true" /> */}
            <input
              id="landing-search"
              type="text"
              value={stagedFilters.keywords}
              onChange={(event) => updateStaged('keywords', event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search for datasets ..."
              className="w-full bg-transparent text-base text-slate-900 placeholder:text-slate-400 focus:outline-none sm:text-lg"
            />
            <button
              type="button"
              onClick={handleSearch}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full text-[#64518c] transition hover:bg-[#64518c]/10 focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2"
              aria-label="Search"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={2.5}
                stroke="currentColor"
                className="h-5 w-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.604 10.604Z"
                />
              </svg>
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3">
            {filters.map(({ id, label, icon: Icon }) => {
              const isActive = activeFilter === id

              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => toggleFilter(id)}
                  className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2 ${
                    isActive
                      ? 'border-[#64518c]/25 bg-[#64518c]/10 text-[#64518c] shadow-sm'
                      : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900'
                  }`}
                  aria-pressed={isActive}
                >
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {label}
                </button>
              )
            })}
          </div>

          {activeFilterConfig ? (
            <div className="mx-auto w-full max-w-2xl rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-600">
                {ActiveFilterIcon ? (
                  <ActiveFilterIcon className="h-4 w-4 text-[#64518c]" aria-hidden="true" />
                ) : null}
                {activeFilterConfig.label} filter
              </div>

              {activeFilterConfig.id === 'source' ? (
                (() => {
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

                  const toggleSource = (val) => {
                    setStagedFilters((s) => {
                      const arr = Array.isArray(s.source) ? [...s.source] : []
                      const idx = arr.indexOf(val)
                      if (idx === -1) arr.push(val)
                      else arr.splice(idx, 1)
                      return { ...s, source: arr }
                    })
                  }

                  return (
                    <div className="grid grid-cols-2 gap-2">
                      {SOURCE_OPTIONS.map((opt) => (
                        <label key={opt} className="inline-flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={Array.isArray(stagedFilters.source) && stagedFilters.source.includes(opt)}
                            onChange={() => toggleSource(opt)}
                            className="h-4 w-4 rounded border-slate-200 text-[#64518c] focus:ring-[#64518c]"
                          />
                          <span className="text-slate-700">{opt}</span>
                        </label>
                      ))}
                    </div>
                  )
                })()
              ) : activeFilterConfig.id === 'dataType' ? (
                (() => {
                  const TYPE_OPTIONS = ['spatial', 'numerical', 'temporal', 'categorical']

                  const toggleType = (val) => {
                    setStagedFilters((s) => {
                      const arr = Array.isArray(s.types) ? [...s.types] : []
                      const idx = arr.indexOf(val)
                      if (idx === -1) arr.push(val)
                      else arr.splice(idx, 1)
                      return { ...s, types: arr }
                    })
                  }

                  return (
                    <div className="grid grid-cols-2 gap-2">
                      {TYPE_OPTIONS.map((opt) => (
                        <label key={opt} className="inline-flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={Array.isArray(stagedFilters.types) && stagedFilters.types.includes(opt)}
                            onChange={() => toggleType(opt)}
                            className="h-4 w-4 rounded border-slate-200 text-[#64518c] focus:ring-[#64518c]"
                          />
                          <span className="text-slate-700">{opt}</span>
                        </label>
                      ))}
                    </div>
                  )
                })()
              ) : activeFilterConfig.id === 'date' ? (
                <div className="flex gap-2">
                  <input
                    type="date"
                    value={stagedFilters.temporal_start}
                    onChange={(e) => updateStaged('temporal_start', e.target.value)}
                    className="w-1/2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 focus:border-[#64518c] focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
                  />
                  <input
                    type="date"
                    value={stagedFilters.temporal_end}
                    onChange={(e) => updateStaged('temporal_end', e.target.value)}
                    className="w-1/2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 focus:border-[#64518c] focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
                  />
                </div>
              ) : activeFilterConfig.id === 'location' ? (
                <div className="w-full space-y-2">
                  <div className="w-full h-56 rounded-md overflow-hidden">
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
                  <p className="text-xs text-slate-500 px-1">Click two points on the map to draw a bounding box.</p>
                </div>
              ) : (
                <input
                  type="text"
                  value={stagedFilters[activeFilterConfig.id] ?? ''}
                  onChange={(event) => updateStaged(activeFilterConfig.id, event.target.value)}
                  placeholder={activeFilterConfig.placeholder}
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-[#64518c] focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
                />
              )}
            </div>
          ) : null}

          {/* <p className="text-center text-sm text-slate-400">
            Press Enter to search. Filters stay lightweight for fast academic discovery.
          </p> */}
        </section>
      </div>
    </main>
  )
}

export default Home
