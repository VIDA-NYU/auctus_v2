import React from 'react'
import { ResultSnippet } from './ResultSnippet.jsx'
import DatasetDetail from './DatasetDetail.jsx'
import ResultGroup from './ResultGroup.jsx'

export default function DatasetResults({ results = [], groups = null, selectedDataset, setSelectedDataset }) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-6 md:flex-row">
      <div className="min-h-0 w-full md:w-4/12 md:max-w-[32rem] md:shrink-0">
        <div className="h-full min-h-0 overflow-y-auto rounded-lg border border-slate-200 divide-y divide-slate-200">
          {groups ? groups.map((group) => (
            <div
              key={group.key}
              role="button"
              tabIndex={0}
              onClick={() => setSelectedDataset(group.best)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setSelectedDataset(group.best)
              }}
              className={`cursor-pointer p-4 ${[group.best, ...(group.matches || [])].includes(selectedDataset) ? 'bg-[#fcfaff] ring-1 ring-[#64518c]/30' : 'hover:bg-[#faf7ff]'}`}
            >
              <ResultGroup group={group} selectedDataset={selectedDataset} setSelectedDataset={setSelectedDataset} />
            </div>
          )) : results.map((dataset, idx) => (
            <div
              key={dataset.id || idx}
              role="button"
              tabIndex={0}
              onClick={() => setSelectedDataset(dataset)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setSelectedDataset(dataset)
              }}
              className={`cursor-pointer p-3 ${selectedDataset === dataset ? 'bg-[#f7f3fb] ring-1 ring-[#64518c]/30' : 'hover:bg-[#faf7ff]'}`}
            >
              <ResultSnippet dataset={dataset} />
            </div>
          ))}
        </div>
      </div>

      <aside className="min-h-0 w-full md:w-8/12 md:flex-1">
        <div className="h-full min-h-0 overflow-y-auto rounded-lg border border-slate-200 bg-white p-6">
          <DatasetDetail dataset={selectedDataset} />
        </div>
      </aside>
    </div>
  )
}
