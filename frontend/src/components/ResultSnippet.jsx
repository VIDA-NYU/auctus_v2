import { ExternalLink } from 'lucide-react'

/**
 * ResultSnippet displays a single dataset result.
 * @param {object} dataset - The dataset object from the API
 */
export function ResultSnippet({ dataset }) {
  const {
    title = 'Untitled Dataset',
    description = 'No description available',
    types = [],
  } = dataset

  return (
    <article className="border-b border-slate-200 px-6 py-5 transition hover:bg-slate-50">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-slate-900 hover:text-[#64518c] transition">
            {title}
          </h3>
          <p className="mt-2 text-sm text-slate-600 leading-relaxed">
            {description}
          </p>
          {types && types.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {types.map((type, idx) => (
                <span
                  key={idx}
                  className="inline-block rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700"
                >
                  {type}
                </span>
              ))}
            </div>
          )}
        </div>
        <button
          type="button"
          className="mt-1 flex-shrink-0 rounded-lg p-2 text-slate-400 transition hover:bg-slate-200 hover:text-[#64518c] focus:outline-none focus:ring-2 focus:ring-[#64518c]"
          aria-label={`View details for ${title}`}
        >
          <ExternalLink className="h-5 w-5" aria-hidden="true" />
        </button>
      </div>
    </article>
  )
}
