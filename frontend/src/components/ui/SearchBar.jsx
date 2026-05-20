import React from 'react';
import { Link } from 'react-router-dom';

/**
 * SearchBar Component
 * 
 * A minimalist, reusable search input with optional branding logo.
 * Uses premium purple design tokens and shadow styling.
 * 
 * Props:
 *   - value: string — Current search text
 *   - onChange: (e) => void — Handler for input changes
 *   - onSearch: () => void — Handler for search submission (Enter key or icon click)
 *   - showLogo: boolean — If true, render AUCTUS logo linked to "/"
 *   - placeholder: string (optional) — Placeholder text for the input
 */
export default function SearchBar({ value, onChange, onSearch, showLogo = false, isCompact = false, placeholder = "Search for datasets..." }) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      onSearch();
    }
  };

  const wrapperClasses = isCompact
    ? 'relative flex-1 flex items-center gap-3 rounded-full border border-slate-200 bg-white px-5 py-2 shadow-[0_12px_30px_-18px_rgba(15,23,42,0.45)] transition-shadow focus-within:shadow-[0_16px_40px_-18px_rgba(100,81,140,0.35)]'
    : 'relative flex-1 flex items-center gap-3 rounded-full border border-slate-200 bg-white px-5 py-4 shadow-[0_12px_30px_-18px_rgba(15,23,42,0.45)] transition-shadow focus-within:shadow-[0_16px_40px_-18px_rgba(100,81,140,0.35)]';

  const inputClasses = isCompact
    ? 'flex-1 bg-transparent text-sm sm:text-base text-slate-900 placeholder:text-slate-400 focus:outline-none'
    : 'flex-1 bg-transparent text-base sm:text-lg text-slate-900 placeholder:text-slate-400 focus:outline-none';

  const buttonClasses = isCompact
    ? 'inline-flex h-8 w-8 items-center justify-center rounded-full text-[#64518c] transition hover:bg-[#64518c]/10 focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2'
    : 'inline-flex h-10 w-10 items-center justify-center rounded-full text-[#64518c] transition hover:bg-[#64518c]/10 focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2';

  return (
    <div className="flex items-center gap-3 w-full">
      {/* AUCTUS Logo (conditional) */}
      {showLogo && (
        <Link
          to="/"
          className="flex-shrink-0 font-bold text-lg text-[#64518c] hover:text-[#56457a] transition-colors"
        >
          AUCTUS
        </Link>
      )}

      {/* Search Input Field - Premium styling with original shadows */}
      <div className={wrapperClasses}>
        <input
          type="text"
          value={value}
          onChange={onChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className={inputClasses}
        />

        {/* Magnifying Glass Icon (embedded) */}
        <button
          type="button"
          onClick={onSearch}
          className={buttonClasses}
          aria-label="Search"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.604 10.604Z"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
