import React, { useRef, useEffect } from 'react';

/**
 * FilterDropdown Component
 * 
 * A reusable dropdown button for filter categories.
 * Shows active state (with count) and floating panel with children options.
 * Uses premium purple design tokens matching the original design system.
 * 
 * Props:
 *   - label: string — Display name (e.g., "Source", "Data Types")
 *   - activeCount: number — Number of active filters (>0 shows active state)
 *   - isOpen: boolean — Whether the dropdown panel is visible
 *   - onToggle: () => void — Handler to toggle the dropdown
 *   - children: ReactNode — Filter options to render inside the panel
 */
export default function FilterDropdown({ label, activeCount = 0, isOpen, onToggle, children }) {
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        if (isOpen) {
          onToggle();
        }
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen, onToggle]);

  const isActive = activeCount > 0;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Toggle Button - Premium purple styling */}
      <button
        onClick={onToggle}
        className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-[#64518c] focus:ring-offset-2 ${
          isActive
            ? 'border-[#64518c]/25 bg-[#64518c]/10 text-[#64518c] shadow-sm font-semibold'
            : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900'
        }`}
      >
        {isActive ? `${label} (${activeCount})` : label}
      </button>

      {/* Backdrop Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30"
          onClick={onToggle}
          aria-hidden="true"
        />
      )}

      {/* Floating Dropdown Panel - Premium styling with shadow */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 rounded-2xl border border-slate-200 bg-white p-4 shadow-xl z-50 min-w-max">
          {children}
        </div>
      )}
    </div>
  );
}
