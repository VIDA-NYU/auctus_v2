import { useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import axios from 'axios'
import { ExternalLink } from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'

// Order the record's parts the way a reader scans a compendium: data first, then code, then docs.
const ROLE_ORDER = [
  'table', 'split table', 'map layer', 'raster map', 'image set (georeferenced)', 'image set', 'image',
  'model output', 'point clouds', '3D meshes', 'text corpus', 'spectra', 'code', 'documentation',
  'copy of an indexed layer', 'archive', 'not profiled', 'other',
]
const ROLE_HINT = {
  'copy of an indexed layer': 'The State Plane copy of a layer Auctus indexes in WGS84.',
  archive: 'A zip whose contents Auctus does not read (yet).',
  'not profiled': 'Indexed by its metadata only (too large or unreadable).',
}

// Row and column counts mean something for these; for a code or text collection they
// count the sampled files' summary table.
const TABULAR = new Set(['table', 'split table', 'map layer', 'spectra'])

function fileName(key) {
  return String(key || '').split('/').pop()
}

function Chip({ children }) {
  return (
    <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-700">
      {children}
    </span>
  )
}

function Section({ title, count, children }) {
  return (
    <section className="space-y-3 border-t border-slate-200 pt-6">
      <h2 className="text-base font-semibold text-slate-900">
        {title}
        {count != null && <span className="ml-2 text-sm font-normal text-slate-500">{count}</span>}
      </h2>
      {children}
    </section>
  )
}

export default function RecordPage() {
  const { recordId } = useParams()
  const [searchParams] = useSearchParams()
  const domain = searchParams.get('domain') || 'ultraviolet.library.nyu.edu'
  // Results are kept with the record they belong to, so a new recordId shows the
  // loading state without resetting anything inside the effect.
  const key = `${domain}:${recordId}`
  const [loaded, setLoaded] = useState({ key: null, record: null, error: null })

  useEffect(() => {
    let cancelled = false
    axios
      .get(`${API_BASE_URL}/api/v1/compendia/${encodeURIComponent(recordId)}`, { params: { domain }, timeout: 30000 })
      .then((res) => !cancelled && setLoaded({ key, record: res.data, error: null }))
      .catch((err) => !cancelled && setLoaded({ key, record: null, error: err.response?.status === 404
        ? 'This record has no compendium yet. It is built after a crawl.'
        : 'Could not reach the Auctus backend at localhost:8000.' }))
    return () => { cancelled = true }
  }, [key, recordId, domain])
  const record = loaded.key === key ? loaded.record : null
  const error = loaded.key === key ? loaded.error : null

  const partsById = useMemo(() => {
    const map = {}
    for (const p of record?.parts || []) if (p.dataset_id) map[p.dataset_id] = p
    return map
  }, [record])

  const groups = useMemo(() => {
    const byRole = {}
    for (const p of record?.parts || []) (byRole[p.role] ||= []).push(p)
    return Object.entries(byRole).sort(
      ([a], [b]) => (ROLE_ORDER.indexOf(a) + 1 || 99) - (ROLE_ORDER.indexOf(b) + 1 || 99),
    )
  }, [record])

  if (error) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-6 text-sm text-amber-900">{error}</div>
      </main>
    )
  }
  if (!record) {
    return (
      <main className="mx-auto max-w-5xl space-y-3 px-4 py-10 sm:px-6">
        {[...Array(3)].map((_, i) => <div key={i} className="h-20 animate-pulse rounded-lg bg-slate-100" />)}
      </main>
    )
  }

  const rel = record.relations || {}
  const partName = (id) => fileName(partsById[id]?.file_key || id)
  const codeReads = (rel.code_reads || []).filter((c) => c.part)

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-10 sm:px-6">
      <header className="space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-[#64518c]">Research compendium · {record.domain}</p>
        <h1 className="text-2xl font-semibold text-slate-900 text-balance">{record.title}</h1>
        <p className="text-sm text-slate-600">{record.summary}</p>
        <div className="flex flex-wrap items-center gap-2">
          {(record.subjects || []).slice(0, 8).map((s) => <Chip key={s}>{s}</Chip>)}
          <a href={record.landing_url} target="_blank" rel="noopener noreferrer"
             className="inline-flex items-center gap-1 text-sm font-medium text-[#64518c] hover:underline">
            Open in the repository <ExternalLink size={14} aria-hidden="true" />
          </a>
        </div>
        {record.description && (
          <p className="max-w-3xl whitespace-pre-wrap text-sm leading-relaxed text-slate-700">{record.description}</p>
        )}
      </header>

      <Section title="Parts" count={(record.parts || []).length}>
        <div className="space-y-5">
          {groups.map(([role, parts]) => (
            <div key={role}>
              <h3 className="text-sm font-medium text-slate-800">
                {role} <span className="font-normal text-slate-500">({parts.length})</span>
              </h3>
              {ROLE_HINT[role] && <p className="text-xs text-slate-500">{ROLE_HINT[role]}</p>}
              <ul className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200">
                {parts.slice(0, 50).map((p) => (
                  <li key={p.dataset_id || p.file_key} className="flex flex-wrap items-baseline justify-between gap-2 px-4 py-2 text-sm">
                    <span className="min-w-0 break-words text-slate-800" title={p.file_key}>{fileName(p.file_key)}</span>
                    <span className="flex gap-3 text-xs text-slate-500 tabular-nums">
                      {TABULAR.has(role) && p.rows != null && <span>{Number(p.rows).toLocaleString()} rows</span>}
                      {TABULAR.has(role) && p.columns != null && <span>{p.columns} columns</span>}
                      {p.partitions != null && <span>{p.partitions} partitions</span>}
                      {p.dataset_id ? (
                        <Link className="text-[#64518c] hover:underline"
                              to={`/results?q=${encodeURIComponent(fileName(p.file_key).replace(/\.[^.]+$/, ''))}`}>
                          find in search
                        </Link>
                      ) : p.url ? (
                        <a className="text-[#64518c] hover:underline" href={p.url} target="_blank" rel="noopener noreferrer">download</a>
                      ) : null}
                    </span>
                  </li>
                ))}
                {parts.length > 50 && <li className="px-4 py-2 text-xs text-slate-500">and {parts.length - 50} more</li>}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      {(rel.join_keys || []).length > 0 && (
        <Section title="Tables that join" count={rel.join_keys.length}>
          <p className="text-xs text-slate-500">Key-like columns two or more of the record's tables share (by name; values not yet compared).</p>
          <ul className="space-y-3">
            {rel.join_keys.map((k) => (
              <li key={k.column} className="rounded-lg border border-slate-200 p-3 text-sm">
                <p className="font-medium text-slate-900">
                  <code className="rounded bg-slate-100 px-1">{k.column}</code>
                  {k.label && <span className="ml-2 font-normal text-slate-600">{k.label}</span>}
                  <span className="ml-2 text-xs font-normal text-slate-500">{k.datasets.length} tables</span>
                </p>
                <p className="mt-1 text-xs text-slate-600 break-words">
                  {k.datasets.slice(0, 12).map(partName).join(' · ')}{k.datasets.length > 12 ? ` · and ${k.datasets.length - 12} more` : ''}
                </p>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {codeReads.length > 0 && (
        <Section title="Code that reads this record's data" count={codeReads.length}>
          <p className="text-xs text-slate-500">From a sample of the record's scripts.</p>
          <ul className="space-y-1 text-sm">
            {codeReads.map((c, i) => (
              <li key={i} className="text-slate-700">
                <code className="rounded bg-slate-100 px-1">{fileName(c.code)}</code> reads{' '}
                <code className="rounded bg-slate-100 px-1">{partName(c.part)}</code>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {(rel.related_records || []).length > 0 && (
        <Section title="Related records" count={rel.related_records.length}>
          <ul className="space-y-1 text-sm">
            {rel.related_records.map((r) => (
              <li key={r.record_id + r.relation} className="text-slate-700">
                <span className="text-slate-500">{r.relation_words || r.relation}</span>{' '}
                {r.indexed ? (
                  <Link className="font-medium text-[#64518c] hover:underline" to={`/records/${r.record_id}?domain=${domain}`}>{r.title || r.record_id}</Link>
                ) : (
                  <a className="font-medium text-[#64518c] hover:underline" href={`https://${domain}/records/${r.record_id}`}
                     target="_blank" rel="noopener noreferrer">{r.record_id}</a>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {(rel.publications || []).length > 0 && (
        <Section title="Publications" count={rel.publications.length}>
          <ul className="space-y-1 text-sm">
            {rel.publications.map((p) => (
              <li key={p.identifier} className="break-words text-slate-700">
                <span className="text-slate-500">{p.relation_words || p.relation}</span>{' '}
                <a className="text-[#64518c] hover:underline" target="_blank" rel="noopener noreferrer"
                   href={p.scheme === 'doi' ? `https://doi.org/${p.identifier}` : p.identifier}>{p.identifier}</a>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {(rel.versions || []).length > 0 && (
        <Section title="Other versions" count={rel.versions.length}>
          <ul className="flex flex-wrap gap-2">
            {rel.versions.map((v) => (
              <li key={v}><Link className="text-sm text-[#64518c] hover:underline" to={`/records/${v}?domain=${domain}`}>{v}</Link></li>
            ))}
          </ul>
        </Section>
      )}
    </main>
  )
}
