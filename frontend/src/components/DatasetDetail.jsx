import React from 'react'

export default function DatasetDetail({ dataset }) {
  if (!dataset) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <p className="text-slate-600">Select a dataset to view full details.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">{dataset.title || dataset.name || dataset.id}</h2>
        <p className="mt-2 text-sm text-slate-700 whitespace-pre-wrap">{dataset.description || dataset.summary || 'No description available.'}</p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center rounded-full bg-[#fffafa] px-3 py-1 text-xs font-medium text-[#64518c] border border-slate-200">{dataset.source || dataset.publisher || 'Source'}</span>
        {Array.isArray(dataset.types || dataset.dataType)
          ? (dataset.types || dataset.dataType).map((t, i) => (
              <span key={i} className="inline-flex items-center rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-700 border border-slate-200">{t}</span>
            ))
          : (dataset.types || dataset.dataType) ? (
              <span className="inline-flex items-center rounded-full bg-slate-50 px-3 py-1 text-xs text-slate-700 border border-slate-200">{dataset.types || dataset.dataType}</span>
            ) : null}
      </div>

      <div className="pt-4">
        <h3 className="text-sm font-medium text-slate-800">Spatial</h3>
        {dataset.bbox || dataset.spatial || dataset.geometry ? (
          <div className="mt-3 h-40 w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-center text-sm text-slate-500">
            Map preview
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-500">No spatial data available for this dataset.</p>
        )}
      </div>
    </div>
  )
}
