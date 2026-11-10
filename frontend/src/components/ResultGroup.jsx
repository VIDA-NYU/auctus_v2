import { useState } from 'react'
import { Link } from 'react-router-dom'

function partName(dataset) {
  const key = dataset?.source_record?.file_key || dataset?.title || dataset?.id || ''
  return String(key).split('/').pop()
}

function PartButton({ dataset, label, isSelected, onSelect, best = false }) {
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onSelect(dataset) }}
      className={`flex w-full items-baseline gap-2 rounded-md px-2 py-1 text-left text-sm transition focus:outline-none focus-visible:ring-2 focus-visible:ring-[#64518c] ${
        isSelected ? 'bg-[#efe6f6] text-[#3f2f63]' : best ? 'bg-[#f7f3fb] text-slate-800 hover:bg-[#efe6f6]' : 'text-slate-700 hover:bg-slate-50'
      }`}
      aria-pressed={isSelected}
    >
      {label && <span className="shrink-0 text-xs font-semibold uppercase tracking-wide text-[#64518c]">{label}</span>}
      <span className="min-w-0 break-words">{partName(dataset)}</span>
      {(dataset.types || []).slice(0, 1).map((t) => (
        <span key={t} className="ml-auto shrink-0 text-xs text-slate-500">{t}</span>
      ))}
    </button>
  )
}

/**
 * One search result per repository record: the record, its best-matching dataset
 * (highlighted, shown in the detail panel), and its other matching parts.
 */
export default function ResultGroup({ group, selectedDataset, setSelectedDataset }) {
  const [open, setOpen] = useState(false)
  const { best, matches = [], n_matches: nMatches = 1, n_parts: nParts, record_id: recordId } = group
  const isRecord = Boolean(recordId)
  const others = nMatches - 1

  return (
    <article className="space-y-2">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold text-slate-900">{group.title || best.title}</h3>
        {isRecord && (
          <Link
            to={`/records/${recordId}?domain=${encodeURIComponent(group.domain || '')}`}
            onClick={(e) => e.stopPropagation()}
            className="shrink-0 text-xs font-medium text-[#64518c] hover:underline"
          >
            record page
          </Link>
        )}
      </div>
      {isRecord && group.summary && <p className="text-xs text-slate-500">{group.summary}</p>}

      <div className="space-y-1">
        <PartButton dataset={best} label={isRecord ? 'Best match' : null} best
                    isSelected={selectedDataset === best} onSelect={setSelectedDataset} />
        {isRecord && (nParts || 0) > 1 && (
          <p className="px-2 text-xs text-slate-500">
            {nMatches} {nMatches === 1 ? 'part matches' : 'parts match'}{nParts ? ` of ${nParts}` : ''}
          </p>
        )}
        {others > 0 && (
          <>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setOpen((v) => !v) }}
              className="px-2 text-xs font-medium text-[#64518c] hover:underline"
              aria-expanded={open}
            >
              {open ? 'Hide' : 'Show'} {matches.length < others ? `the next ${matches.length}` : others} other matching part{others === 1 ? '' : 's'}
            </button>
            {open && (
              <div className="space-y-1 border-l-2 border-[#efe6f6] pl-2">
                {matches.map((d) => (
                  <PartButton key={d.id} dataset={d} isSelected={selectedDataset === d} onSelect={setSelectedDataset} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </article>
  )
}
