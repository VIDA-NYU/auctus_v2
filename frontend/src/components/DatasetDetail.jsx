import React, { useMemo } from 'react'
import { useDatasetProfile } from '../hooks/useDatasetProfile.js'

function parseCsvString(csv) {
  if (!csv || typeof csv !== 'string') return null
  const lines = csv.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (lines.length < 2) return null

  const parseLine = (line) => {
    const cols = []
    let cur = ''
    let inQuotes = false
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"'
          i++
        } else {
          inQuotes = !inQuotes
        }
      } else if (ch === ',' && !inQuotes) {
        cols.push(cur)
        cur = ''
      } else {
        cur += ch
      }
    }
    cols.push(cur)
    return cols
  }

  const headers = parseLine(lines[0]).map((h) => h.trim())
  const rows = []
  for (let i = 1; i < lines.length; i++) {
    const cols = parseLine(lines[i])
    if (cols.length === 1 && cols[0] === '') continue
    const obj = {}
    for (let j = 0; j < headers.length; j++) {
      obj[headers[j]] = cols[j] !== undefined ? cols[j] : null
    }
    rows.push(obj)
  }
  return rows.length ? rows : null
}

function findSampleData(profile) {
  const visited = new Set()

  const walk = (value) => {
    if (!value || (typeof value !== 'object' && typeof value !== 'string') || visited.has(value)) return null
    // if it's a string that contains CSV sample lines, try parsing
    if (typeof value === 'string') {
      const parsed = parseCsvString(value)
      if (Array.isArray(parsed)) return parsed
      return null
    }
    visited.add(value)

    if (Array.isArray(value.sample_data)) return value.sample_data
    if (Array.isArray(value.sampleData)) return value.sampleData
    if (Array.isArray(value.rows)) return value.rows
    if (Array.isArray(value.data)) return value.data
    // legacy/profile payload: CSV text is stored under `sample`
    if (typeof value.sample === 'string') {
      const parsed = parseCsvString(value.sample)
      if (Array.isArray(parsed)) return parsed
    }

    if (value.profiler_metadata) {
      const nested = walk(value.profiler_metadata)
      if (nested) return nested
    }

    if (Array.isArray(value.columns)) {
      for (const column of value.columns) {
        const nested = walk(column)
        if (nested) return nested
      }
    }

    for (const nestedValue of Object.values(value)) {
      if (nestedValue && (typeof nestedValue === 'object' || typeof nestedValue === 'string')) {
        const nested = walk(nestedValue)
        if (nested) return nested
      }
    }

    return null
  }

  const found = walk(profile)
  return Array.isArray(found) ? found : null
}

function normalizeRows(sampleData) {
  if (!Array.isArray(sampleData)) return []
  return sampleData
    .map((row) => {
      if (row && typeof row === 'object' && !Array.isArray(row)) return row
      return { value: row }
    })
    .filter(Boolean)
}

export default function DatasetDetail({ dataset }) {
  const datasetId = dataset?.id || dataset?.dataset_id || null
  const { profile, isLoadingProfile, profileError } = useDatasetProfile(datasetId)

  const sampleRows = useMemo(() => normalizeRows(findSampleData(profile)), [profile])
  const tableHeaders = useMemo(() => {
    const headers = new Set()
    sampleRows.forEach((row) => {
      Object.keys(row || {}).forEach((key) => headers.add(key))
    })
    return Array.from(headers)
  }, [sampleRows])

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

      <div className="pt-4">
        <h3 className="text-sm font-medium text-slate-800">JIT Profile Preview</h3>

        {isLoadingProfile ? (
          <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
            Loading profiling preview...
          </div>
        ) : profileError ? (
          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            {profileError}
          </div>
        ) : sampleRows.length > 0 && tableHeaders.length > 0 ? (
          <div className="mt-3 overflow-hidden rounded-lg border border-slate-200 bg-white">
            <div className="max-h-96 overflow-auto">
              <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                <thead className="sticky top-0 bg-slate-50">
                  <tr>
                    {tableHeaders.map((header) => (
                      <th key={header} className="px-4 py-3 font-semibold text-slate-700">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {sampleRows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="align-top">
                      {tableHeaders.map((header) => (
                        <td key={header} className="max-w-xs px-4 py-3 text-slate-700">
                          <div className="max-h-24 overflow-auto whitespace-pre-wrap break-words">
                            {row?.[header] == null ? '—' : typeof row[header] === 'object' ? JSON.stringify(row[header]) : String(row[header])}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-500">
            No sample preview data was found in the stored profile.
          </p>
        )}
      </div>
    </div>
  )
}
