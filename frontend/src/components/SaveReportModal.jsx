import React, { useState } from 'react';
import { X, BookmarkPlus, Check } from 'lucide-react';
import { saveReport } from '../services/reportStorage';

export default function SaveReportModal({ isOpen, onClose, reportData, onSaved }) {
  const [title, setTitle] = useState(
    reportData?.title || `Audit: ${reportData?.transactionId || reportData?.output?.transaction_id || 'Transaction'}`
  );
  const [note, setNote] = useState('');

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    const saved = saveReport({
      ...reportData,
      title: title.trim() || `Report ${new Date().toLocaleDateString()}`,
      note: note.trim(),
    });
    if (onSaved) onSaved(saved);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-2xs animate-in fade-in duration-150">
      <div className="card-clean max-w-md w-full p-6 shadow-xl space-y-5 relative">
        <div className="flex items-center justify-between pb-3 border-b border-slate-200">
          <div className="flex items-center gap-2 text-indigo-700">
            <BookmarkPlus className="w-5 h-5" />
            <h3 className="text-base font-bold text-slate-900">Save to Audit Reports</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Report Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Unusual $1,200 purchase review"
              required
              className="input-clean text-sm"
              autoFocus
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Internal Review Note (Optional)
            </label>
            <textarea
              rows={3}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add details, analyst remarks, or verification notes..."
              className="input-clean text-sm resize-none"
            />
          </div>

          <div className="flex items-center justify-end gap-2 pt-3 border-t border-slate-200">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary text-xs py-2 px-3"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary text-xs py-2 px-4"
            >
              <Check className="w-4 h-4" />
              Save Report
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
