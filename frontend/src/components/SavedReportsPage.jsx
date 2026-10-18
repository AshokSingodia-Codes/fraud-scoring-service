import React, { useState } from 'react';
import {
  FileText,
  Search,
  Filter,
  Download,
  Upload,
  Trash2,
  Edit2,
  Eye,
  Calendar,
  X,
  Check,
  AlertTriangle
} from 'lucide-react';
import {
  getAllReports,
  deleteReport,
  updateReport,
  exportAllReportsJson,
  importReportsJson
} from '../services/reportStorage';
import { exportReportPdf, exportReportCsv, exportReportJson } from '../utils/pdfExport';
import { formatReasonExplanation, getDecisionBadgeInfo } from '../utils/reasonFormatter';

export default function SavedReportsPage({ onToast }) {
  const [reports, setReports] = useState(() => getAllReports());
  const [searchQuery, setSearchQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState('all');
  const [selectedReport, setSelectedReport] = useState(null);

  // Edit Title/Note State
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [editNote, setEditNote] = useState('');

  // Delete confirmation modal state
  const [deletingId, setDeletingId] = useState(null);

  const loadReports = () => {
    setReports(getAllReports());
  };

  const handleStartEdit = (report) => {
    setEditingId(report.id);
    setEditTitle(report.title || '');
    setEditNote(report.note || '');
  };

  const handleSaveEdit = (id) => {
    updateReport(id, { title: editTitle, note: editNote });
    setEditingId(null);
    loadReports();
    if (onToast) onToast("Report title and notes updated.");
  };

  const handleConfirmDelete = (id) => {
    deleteReport(id);
    setDeletingId(null);
    if (selectedReport?.id === id) setSelectedReport(null);
    loadReports();
    if (onToast) onToast("Report removed from vault.");
  };

  const handleImportBackup = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const count = importReportsJson(event.target.result);
        loadReports();
        if (onToast) onToast(`Successfully restored ${count} reports from backup!`);
      } catch (err) {
        alert(err.message);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  // Filtered reports list
  const filteredReports = reports.filter(r => {
    const matchSearch = searchQuery.trim() === '' ||
      r.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.transactionId?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.note?.toLowerCase().includes(searchQuery.toLowerCase());

    const matchDecision = decisionFilter === 'all' || r.decision === decisionFilter;
    return matchSearch && matchDecision;
  });

  return (
    <div className="space-y-8 py-2">
      {/* Title & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Saved Audit Reports
          </h1>
          <p className="text-sm text-slate-600">
            Private audit log stored securely in your browser. Print formal PDFs, export CSV spreadsheets, or restore from backup.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="btn-secondary text-xs py-2 px-3 cursor-pointer">
            <Upload className="w-3.5 h-3.5" />
            Import Backup (JSON)
            <input type="file" accept=".json" onChange={handleImportBackup} className="hidden" />
          </label>
          <button
            onClick={exportAllReportsJson}
            disabled={reports.length === 0}
            className="btn-secondary text-xs py-2 px-3 disabled:opacity-50"
            title="Download all saved audits as a single JSON backup file"
          >
            <Download className="w-3.5 h-3.5" />
            Export All (JSON)
          </button>
        </div>
      </div>

      {/* Search & Filter Header */}
      <div className="card-clean p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
            <input
              type="text"
              placeholder="Search reports by title, Transaction ID, or note..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-clean pl-9 text-xs"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 flex items-center gap-1">
              <Filter className="w-3.5 h-3.5" /> Filter Decision:
            </span>
            <select
              value={decisionFilter}
              onChange={(e) => setDecisionFilter(e.target.value)}
              className="input-clean text-xs py-1.5 px-3 w-auto"
            >
              <option value="all">All Verdicts</option>
              <option value="approve">Approve Only</option>
              <option value="review">Review Only</option>
              <option value="block">Block Only</option>
            </select>
          </div>
        </div>

        {/* Reports List */}
        {filteredReports.length === 0 ? (
          <div className="py-12 text-center space-y-3 text-slate-400">
            <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto text-slate-400">
              <FileText className="w-6 h-6" />
            </div>
            <h3 className="font-bold text-slate-700 text-sm">No Saved Reports Found</h3>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Whenever you check a single transaction or run a bulk batch, click "Save to Reports" to preserve it here.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredReports.map((report) => {
              const badge = getDecisionBadgeInfo(report.decision);
              const dateStr = new Date(report.savedAt).toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              });

              return (
                <div
                  key={report.id}
                  className="p-4 sm:p-5 rounded-xl border border-slate-200 bg-white hover:border-slate-300 hover:shadow-xs transition-all space-y-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="space-y-1 flex-1 min-w-[200px]">
                      {editingId === report.id ? (
                        <div className="space-y-2">
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="input-clean text-xs font-bold py-1 px-2"
                            placeholder="Report title"
                          />
                          <input
                            type="text"
                            value={editNote}
                            onChange={(e) => setEditNote(e.target.value)}
                            className="input-clean text-xs py-1 px-2"
                            placeholder="Notes..."
                          />
                          <div className="flex gap-2 pt-1">
                            <button
                              onClick={() => handleSaveEdit(report.id)}
                              className="btn-primary text-[11px] py-1 px-2.5"
                            >
                              <Check className="w-3 h-3" /> Save
                            </button>
                            <button
                              onClick={() => setEditingId(null)}
                              className="btn-secondary text-[11px] py-1 px-2.5"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-center gap-2">
                            <h3 className="text-sm font-bold text-slate-900">{report.title}</h3>
                            <button
                              onClick={() => handleStartEdit(report)}
                              className="text-slate-400 hover:text-slate-600"
                              title="Edit title & notes"
                            >
                              <Edit2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-500 font-mono">
                            <span>ID: {report.transactionId}</span>
                            <span>•</span>
                            <span className="flex items-center gap-1 font-sans">
                              <Calendar className="w-3 h-3 text-slate-400" /> {dateStr}
                            </span>
                            {report.input?.TransactionAmt && (
                              <>
                                <span>•</span>
                                <span className="font-sans font-semibold text-slate-700">
                                  ${Number(report.input.TransactionAmt).toFixed(2)}
                                </span>
                              </>
                            )}
                          </div>
                          {report.note && (
                            <p className="text-xs text-slate-600 bg-slate-50 p-2 rounded-lg border border-slate-100">
                              {report.note}
                            </p>
                          )}
                        </>
                      )}
                    </div>

                    <div className="flex flex-col items-end gap-1.5">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider border ${badge.colorClass}`}>
                        {badge.label}
                      </span>
                      {report.fraudProbability !== null && (
                        <span className="text-xs font-black text-slate-900">
                          {(report.fraudProbability * 100).toFixed(2)}% Risk
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Action Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 text-xs">
                    <button
                      onClick={() => setSelectedReport(report)}
                      className="text-indigo-600 hover:text-indigo-800 font-bold inline-flex items-center gap-1"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      View Full Audit
                    </button>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => exportReportPdf(report)}
                        className="btn-secondary text-[11px] py-1 px-2 text-slate-700"
                        title="Download printable Fraud Check Report (PDF)"
                      >
                        <Download className="w-3 h-3" />
                        PDF
                      </button>
                      <button
                        onClick={() => exportReportCsv(report)}
                        className="btn-secondary text-[11px] py-1 px-2 text-slate-700"
                        title="Export as CSV spreadsheet"
                      >
                        CSV
                      </button>
                      <button
                        onClick={() => exportReportJson(report)}
                        className="btn-secondary text-[11px] py-1 px-2 text-slate-700"
                        title="Export raw JSON"
                      >
                        JSON
                      </button>
                      <button
                        onClick={() => setDeletingId(report.id)}
                        className="p-1.5 text-slate-400 hover:text-rose-600 rounded-lg hover:bg-rose-50 transition-colors ml-1"
                        title="Delete report"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Report Detail Modal */}
      {selectedReport && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-2xs animate-in fade-in">
          <div className="card-clean max-w-2xl w-full p-6 sm:p-7 shadow-2xl space-y-6 relative max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div>
                <span className="text-[11px] font-mono text-slate-400">AUDIT REPORT INSPECTOR</span>
                <h3 className="text-lg font-bold text-slate-900">{selectedReport.title}</h3>
              </div>
              <button
                onClick={() => setSelectedReport(null)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Decision card */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex justify-between items-center">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Assigned Verdict</span>
                <span className="text-xl font-black text-slate-900 uppercase">
                  {selectedReport.decision} ({selectedReport.riskBand} Risk)
                </span>
                <span className="text-xs text-slate-500 block">
                  Reference: {selectedReport.transactionId}
                </span>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Fraud Probability</span>
                <span className="text-3xl font-black text-indigo-700">
                  {selectedReport.fraudProbability !== null ? `${(selectedReport.fraudProbability * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>
            </div>

            {/* Reasons if single check */}
            {selectedReport.reasons && selectedReport.reasons.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                  Recorded Explanations
                </h4>
                <div className="space-y-2">
                  {selectedReport.reasons.map((r, i) => {
                    const isUp = r.direction === 'increases_risk' || r.contribution > 0;
                    return (
                      <div
                        key={i}
                        className={`p-3 rounded-lg border text-xs space-y-1 ${
                          isUp ? 'bg-rose-50/50 border-rose-100' : 'bg-emerald-50/50 border-emerald-100'
                        }`}
                      >
                        <div className="flex justify-between font-bold">
                          <span className="text-slate-800">{r.label || r.feature}</span>
                          <span className={isUp ? 'text-rose-700' : 'text-emerald-700'}>
                            {isUp ? '+ Increased Risk' : '- Decreased Risk'}
                          </span>
                        </div>
                        <p className="text-slate-600 text-[11px]">
                          {formatReasonExplanation(r)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex justify-between items-center pt-3 border-t border-slate-200">
              <div className="flex gap-2">
                <button
                  onClick={() => exportReportPdf(selectedReport)}
                  className="btn-secondary text-xs py-2 px-3"
                >
                  <Download className="w-3.5 h-3.5" />
                  Print Fraud Check Report (PDF)
                </button>
                <button
                  onClick={() => exportReportCsv(selectedReport)}
                  className="btn-secondary text-xs py-2 px-3"
                >
                  Download CSV
                </button>
              </div>

              <button
                onClick={() => setSelectedReport(null)}
                className="btn-primary text-xs py-2 px-4"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deletingId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-2xs animate-in fade-in">
          <div className="card-clean max-w-sm w-full p-6 shadow-xl space-y-4 text-center">
            <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-600 flex items-center justify-center mx-auto">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">Delete this report?</h3>
              <p className="text-xs text-slate-500 mt-1">
                This will permanently delete this audit record from your browser storage. This action cannot be undone.
              </p>
            </div>
            <div className="flex gap-2 pt-2">
              <button
                onClick={() => setDeletingId(null)}
                className="btn-secondary text-xs py-2 px-3 flex-1"
              >
                Cancel
              </button>
              <button
                onClick={() => handleConfirmDelete(deletingId)}
                className="bg-rose-600 hover:bg-rose-700 text-white font-bold text-xs py-2 px-3 rounded-lg flex-1 transition-colors"
              >
                Yes, Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
