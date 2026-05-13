import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Calendar, Database, FileText, MapPin, Search } from 'lucide-react'
import { useSearch } from '../hooks/useSearch'
import { ResultSnippet } from '../components/ResultSnippet'

const filters = [
  {
    id: 'date',
    label: 'Date',
    icon: Calendar,
    placeholder: 'Select a date range',
  },
  {
    id: 'location',
    label: 'Location',
    icon: MapPin,
    placeholder: 'Enter a country, region, or city',
  },
  {
    id: 'source',
    label: 'Source',
    icon: Database,
    placeholder: 'Choose a source type',
  },
  {
    id: 'dataType',
    label: 'Data Type',
    icon: FileText,
    placeholder: 'Choose a data category',
  },
]

function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const [activeFilter, setActiveFilter] = useState(null)
  const [filterValues, setFilterValues] = useState({})

  const activeFilterConfig = useMemo(
    () => filters.find((filter) => filter.id === activeFilter),
    [activeFilter],
  )

  const ActiveFilterIcon = activeFilterConfig?.icon

  const { results, loading, error, totalResults } = useSearch(query, filterValues)

  const handleSearch = (newQuery) => {
    setSearchParams({ q: newQuery })
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      const newQuery = event.target.value
      if (newQuery.trim()) {
        handleSearch(newQuery)
      }
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
    <main className="min-h-screen bg-white">
      {/* Sticky Search Header */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white shadow-sm">
        <div className="mx-auto max-w-4xl px-6 py-4">
          <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
            <Search className="h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
            <input
              type="text"
              defaultValue={query}
              onKeyDown={handleKeyDown}
              placeholder="Search datasets..."
              className="flex-1 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
            />
            <button
              type="button"
              className="rounded-md bg-[#64518c] px-3 py-1.5 text-xs font-medium text-white transition hover:bg-[#56457a] focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2"
            >
              Search
            </button>
          </div>
        </div>
      </header>

      {/* Filter Bar */}
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-4xl px-6 py-3">
          <div className="flex flex-wrap items-center gap-2">
            {filters.map(({ id, label, icon: Icon }) => {
              const isActive = activeFilter === id

              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => toggleFilter(id)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2 ${
                    isActive
                      ? 'border-[#64518c]/25 bg-[#64518c]/10 text-[#64518c] shadow-sm'
                      : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900'
                  }`}
                  aria-pressed={isActive}
                >
                  <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Active Filter Input Panel */}
      {activeFilterConfig ? (
        <div className="border-b border-slate-200 bg-slate-50">
          <div className="mx-auto max-w-4xl px-6 py-3">
            <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white p-3">
              {ActiveFilterIcon ? (
                <ActiveFilterIcon className="h-4 w-4 shrink-0 text-[#64518c]" aria-hidden="true" />
              ) : null}
              <input
                type="text"
                value={filterValues[activeFilterConfig.id] ?? ''}
                onChange={(event) => updateFilterValue(activeFilterConfig.id, event.target.value)}
                placeholder={activeFilterConfig.placeholder}
                className="flex-1 bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
                autoFocus
              />
            </div>
          </div>
        </div>
      ) : null}

      {/* Results Content Area */}
      <div className="bg-white px-6 py-8">
        <div className="mx-auto max-w-4xl">
          {/* Loading State */}
          {loading ? (
            <div className="space-y-4">
              <p className="text-center text-slate-600">
                Searching {totalResults.toLocaleString()}+ datasets...
              </p>
              <div className="space-y-3">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="animate-pulse rounded-lg bg-slate-100 p-4 h-24" />
                ))}
              </div>
            </div>
          ) : error ? (
            /* Error State */
            <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center">
              <p className="text-sm font-medium text-red-900">{error}</p>
              <p className="mt-2 text-xs text-red-700">
                Please ensure the FastAPI backend is running at localhost:8000
              </p>
            </div>
          ) : results.length === 0 && !loading ? (
            /* Empty/Zero Results State */
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-12 text-center">
              <p className="text-lg font-medium text-slate-900">No datasets found</p>
              <p className="mt-2 text-sm text-slate-600">
                Try adjusting your search terms or filters to find what you're looking for.
              </p>
            </div>
          ) : (
            /* Results List */
            <div>
              <p className="mb-6 text-sm text-slate-600">
                Found <span className="font-semibold text-slate-900">{totalResults}</span> dataset
                {totalResults !== 1 ? 's' : ''} for "<span className="font-semibold">{query}</span>"
              </p>
              <div className="divide-y divide-slate-200 rounded-lg border border-slate-200 overflow-hidden">
                {results.map((dataset, idx) => (
                  <ResultSnippet key={idx} dataset={dataset} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

export default ResultsPage
