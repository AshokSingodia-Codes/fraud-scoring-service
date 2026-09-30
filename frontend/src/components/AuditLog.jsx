import React, { useState } from 'react';
import { Search, FileText, CheckCircle2, AlertCircle } from 'lucide-react';
import { getPredictionById } from '../services/api';

export default function AuditLog() {
  const [searchId, setSearchId] = useState('');
  const [record, setRecord] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchId.trim()) return;
    setLoading(true);
    setError(null);
    setRecord(null);

    try {
      const data = await getPredictionById(searchId.trim());
      setRecord(data);
    } catch (err) {
      setError(err.response?.status === 404 ? 'No audit record found for this Transaction ID.' : err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl space-y-6">
      <div className="pb-4 border-b border-slate-800">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <FileText className="w-5 h-5 text-indigo-400" />
          Historical Prediction Audit Logs
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">Query predictions stored asynchronously in PostgreSQL / SQLite audit tables</p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
          <input
            type="text"
            placeholder="Enter Transaction ID (e.g. tx_live_test, tx_demo_101)..."
            value={searchId}
            onChange={(e) => setSearchId(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm rounded-xl transition-all disabled:opacity-50"
        >
          {loading ? 'Searching...' : 'Search Log'}
        </button>
      </form>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {record && (
        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <span className="text-xs text-slate-500 font-mono">Transaction ID</span>
              <h3 className="text-base font-bold text-slate-100">{record.transaction_id}</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-extrabold ${
              record.decision === 'approve' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' :
              record.decision === 'review' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30' : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
            }`}>
              {record.decision.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-slate-500 block">Fraud Probability</span>
              <span className="font-bold text-slate-200 text-sm">{(record.fraud_probability * 100).toFixed(2)}%</span>
            </div>
            <div>
              <span className="text-slate-500 block">Risk Band</span>
              <span className="font-bold text-indigo-400 text-sm">{record.risk_band}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Model Version</span>
              <span className="font-mono text-slate-300">{record.model_version}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Latency</span>
              <span className="font-mono text-slate-300">{record.latency_ms} ms</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
