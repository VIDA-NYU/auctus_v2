import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Calendar, Database, FileText, MapPin, Search } from 'lucide-react'

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
  const [query, setQuery] = useState('')
  const [activeFilter, setActiveFilter] = useState(null)
  const [filterValues, setFilterValues] = useState({})

  const activeFilterConfig = useMemo(
    () => filters.find((filter) => filter.id === activeFilter),
    [activeFilter],
  )

  const ActiveFilterIcon = activeFilterConfig?.icon

  const handleSearch = () => {
    // Navigation to results page
    if (query.trim()) {
      navigate(`/results?q=${encodeURIComponent(query)}`)
    }
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

  const updateFilterValue = (filterId, value) => {
    setFilterValues((current) => ({
      ...current,
      [filterId]: value,
    }))
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
            <Search className="h-5 w-5 shrink-0 text-slate-400" aria-hidden="true" />
            <input
              id="landing-search"
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search for datasets ..."
              className="w-full bg-transparent text-base text-slate-900 placeholder:text-slate-400 focus:outline-none sm:text-lg"
            />
            <button
              type="button"
              onClick={handleSearch}
              className="rounded-full bg-[#64518c] px-5 py-2.5 text-sm font-medium text-white transition hover:bg-[#56457a] focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2"
            >
              Search
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
              <input
                type={activeFilterConfig.inputType}
                value={filterValues[activeFilterConfig.id] ?? ''}
                onChange={(event) => updateFilterValue(activeFilterConfig.id, event.target.value)}
                placeholder={activeFilterConfig.placeholder}
                className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-[#64518c] focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#64518c]/20"
              />
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
