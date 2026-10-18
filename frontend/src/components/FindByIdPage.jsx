import React, { useState } from 'react';
import {
  Search,
  AlertCircle,
  Calendar,
  Download,
  BookmarkPlus,
  Code,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { getPredictionById } from '../services/api';
import { getDecisionBadgeInfo } from '../utils/reasonFormatter';
import { exportReportPdf } from '../utils/pdfExport';
import SaveReportModal from './SaveReportModal';

export default function FindByIdPage({ onToast }) {
  const [searchId, setSearchId] = useState('');
  const [loading, setLoading] = useState(false);
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [devSectionOpen, setDevSectionOpen] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    const id = searchId.trim();
    if (!id) return;

    setLoading(true);
    setError(null);
    setRecord(null);

    try {
      const data = await getPredictionById(id);
      setRecord(data);
    } catch (err) {
      setError(err.message || "Failed to find transaction audit log.");
    } finally {
      setLoading(false);
    }
  };

  const badge = record ? getDecisionBadgeInfo(record.decision) : null;
  const probPct = record ? (record.fraud_probability * 100).toFixed(2) : '0.00';
  const formattedDate = record?.timestamp
    ? new Date(record.timestamp * 1000).toLocaleString()
    : 'Recent';

  return (
    <div className="space-y-8 py-2">
      {/* Title Header */}
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Find a Past Result by Transaction ID
        </h1>
        <p className="text-sm text-slate-600">
          Search the audit database to retrieve previously scored transactions, decision logs, and timestamps.
        </p>
      </div>

      {/* Explainer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you need to provide
          </span>
          <p className="text-xs text-slate-600">
            The unique Transaction ID you supplied during scoring (e.g. <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-700">tx_demo_101</code>).
          </p>
        </div>
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you will get back
          </span>
          <p className="text-xs text-slate-600">
            The stored verdict, exact fraud probability score, risk category, model version, and exact audit timestamp.
          </p>
        </div>
      </div>

      {/* Search Bar Card */}
      <div className="card-clean p-6 sm:p-8 space-y-5">
        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-5 h-5 text-slate-400 absolute left-3.5 top-3.5" />
            <input
              type="text"
              placeholder="Enter Transaction ID (e.g. tx_demo_101, tx_api_001)..."
              value={searchId}
              onChange={(e) => setSearchId(e.target.value)}
              required
              className="input-clean pl-11 py-3 text-sm font-mono"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !searchId.trim()}
            className="btn-primary py-3 px-6 text-sm whitespace-nowrap"
          >
            {loading ? 'Searching Database...' : 'Search Audit Log'}
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 pt-1">
          <span>Quick search examples:</span>
          <button
            type="button"
            onClick={() => setSearchId('tx_demo_101')}
            className="underline hover:text-indigo-600 font-mono"
          >
            tx_demo_101
          </button>
          <span>•</span>
          <button
            type="button"
            onClick={() => setSearchId('tx_api_001')}
            className="underline hover:text-indigo-600 font-mono"
          >
            tx_api_001
          </button>
        </div>
      </div>

      {/* Error / Not Found Display */}
      {error && (
        <div className="card-clean p-8 text-center space-y-3 border-amber-200 bg-amber-50/50">
          <div className="w-12 h-12 bg-amber-100 text-amber-600 rounded-full flex items-center justify-center mx-auto">
            <AlertCircle className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-900">Transaction Not Found</h3>
          <p className="text-xs text-slate-600 max-w-md mx-auto">
            {error} If you just scored this transaction, verify that database logging is enabled in the backend service or try another ID.
          </p>
        </div>
      )}

      {/* Result Display */}
      {record && (
        <div className="card-clean p-6 sm:p-8 space-y-6 animate-in fade-in duration-200 max-w-2xl mx-auto">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
            <div>
              <span className="text-[11px] font-mono text-slate-400">DATABASE AUDIT RECORD</span>
              <h2 className="text-xl font-extrabold text-slate-900">{record.transaction_id}</h2>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider border ${badge.colorClass}`}>
              {badge.label}
            </span>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-5 rounded-2xl bg-slate-50 border border-slate-200">
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                Calculated Risk
              </span>
              <span className="text-3xl font-black text-slate-900">{probPct}%</span>
              <span className="text-xs text-slate-500 block">Fraud Probability</span>
            </div>

            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                Assigned Action
              </span>
              <span className="text-2xl font-black text-slate-900 uppercase">{record.decision}</span>
              <span className="text-xs text-slate-500 block">{record.risk_band} Risk Category</span>
            </div>

            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                Scoring Latency
              </span>
              <span className="text-2xl font-black text-slate-900">{record.latency_ms} ms</span>
              <span className="text-xs text-slate-500 block">Engine Latency</span>
            </div>
          </div>

          {/* Details Table */}
          <div className="space-y-3 text-xs">
            <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px]">
              Audit Log Metadata
            </h4>
            <div className="divide-y divide-slate-100 rounded-xl border border-slate-200 bg-white overflow-hidden">
              <div className="p-3 flex justify-between">
                <span className="text-slate-500">Transaction ID:</span>
                <span className="font-mono font-bold text-slate-800">{record.transaction_id}</span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-slate-500">Recorded Date & Time:</span>
                <span className="font-medium text-slate-800 flex items-center gap-1">
                  <Calendar className="w-3.5 h-3.5 text-slate-400" /> {formattedDate}
                </span>
              </div>
              <div className="p-3 flex justify-between">
                <span className="text-slate-500">Model Version:</span>
                <span className="font-mono text-slate-800">v{record.model_version}</span>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              onClick={() => exportReportPdf({
                transactionId: record.transaction_id,
                decision: record.decision,
                fraudProbability: record.fraud_probability,
                riskBand: record.risk_band,
                latencyMs: record.latency_ms,
                savedAt: record.timestamp ? new Date(record.timestamp * 1000).toISOString() : new Date().toISOString(),
              })}
              className="btn-secondary text-xs py-2.5 px-4 flex-1"
            >
              <Download className="w-4 h-4" />
              Download Fraud Check Report (PDF)
            </button>
            <button
              onClick={() => setSaveModalOpen(true)}
              className="btn-primary text-xs py-2.5 px-4 flex-1"
            >
              <BookmarkPlus className="w-4 h-4" />
              Save to My Reports
            </button>
          </div>
        </div>
      )}

      {/* Save Modal */}
      {saveModalOpen && record && (
        <SaveReportModal
          isOpen={saveModalOpen}
          onClose={() => setSaveModalOpen(false)}
          reportData={{
            type: 'single',
            transactionId: record.transaction_id,
            decision: record.decision,
            fraudProbability: record.fraud_probability,
            riskBand: record.risk_band,
            modelVersion: record.model_version,
            latencyMs: record.latency_ms,
            output: record,
          }}
          onSaved={() => {
            if (onToast) onToast("Saved database record to your personal Reports Vault.");
          }}
        />
      )}

      {/* Developer Collapsible */}
      <div className="card-clean p-4">
        <button
          type="button"
          onClick={() => setDevSectionOpen(!devSectionOpen)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-500 hover:text-slate-800"
        >
          <span className="flex items-center gap-2">
            <Code className="w-3.5 h-3.5" />
            For Developers: HTTP Audit Endpoint (<code className="text-indigo-600">GET /v1/predictions/&#123;id&#125;</code>)
          </span>
          {devSectionOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {devSectionOpen && (
          <div className="mt-3 pt-3 border-t border-slate-100 space-y-2 text-xs font-mono text-slate-600 animate-in fade-in">
            <p className="font-bold text-slate-800">Endpoint: GET /v1/predictions/{searchId || ':transaction_id'}</p>
            <p>Header: X-API-Key: demo-api-key-12345</p>
          </div>
        )}
      </div>
    </div>
  );
}
