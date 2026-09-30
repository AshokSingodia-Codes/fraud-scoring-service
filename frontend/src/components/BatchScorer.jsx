import React, { useState } from 'react';
import { Layers, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { scoreBatchTransactions } from '../services/api';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const SAMPLE_BATCH = [
  {
    transaction_id: "batch_tx_001",
    TransactionDT: 86400,
    TransactionAmt: 25.00,
    ProductCD: "W",
    card1: 1000,
    card4: "visa",
    card6: "debit",
    P_emaildomain: "gmail.com",
  },
  {
    transaction_id: "batch_tx_002",
    TransactionDT: 86500,
    TransactionAmt: 1450.00,
    ProductCD: "C",
    card1: 19800,
    card4: "mastercard",
    card6: "credit",
    P_emaildomain: "anonymous.com",
  },
  {
    transaction_id: "batch_tx_003",
    TransactionDT: 86600,
    TransactionAmt: 310.00,
    ProductCD: "H",
    card1: 5000,
    card4: "visa",
    card6: "credit",
    P_emaildomain: "yahoo.com",
  },
  {
    transaction_id: "batch_tx_004",
    TransactionDT: 86700,
    TransactionAmt: 12.99,
    ProductCD: "W",
    card1: 1000,
    card4: "visa",
    card6: "debit",
    P_emaildomain: "gmail.com",
  }
];

export default function BatchScorer() {
  const [jsonText, setJsonText] = useState(JSON.stringify(SAMPLE_BATCH, null, 2));
  const [loading, setLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [error, setError] = useState(null);

  const handleScoreBatch = async () => {
    setLoading(true);
    setError(null);
    try {
      const parsed = JSON.parse(jsonText);
      const res = await scoreBatchTransactions(parsed, false);
      setBatchResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Batch scoring failed");
    } finally {
      setLoading(false);
    }
  };

  // Aggregation metrics for chart
  const decisionCounts = { approve: 0, review: 0, block: 0 };
  if (batchResult && batchResult.results) {
    batchResult.results.forEach((r) => {
      decisionCounts[r.decision] = (decisionCounts[r.decision] || 0) + 1;
    });
  }

  const pieData = [
    { name: 'Approve', value: decisionCounts.approve, color: '#10b981' },
    { name: 'Review', value: decisionCounts.review, color: '#f59e0b' },
    { name: 'Block', value: decisionCounts.block, color: '#ef4444' },
  ].filter((d) => d.value > 0);

  return (
    <div className="space-y-6">
      <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-800">
          <div>
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              High-Throughput Batch Scoring Engine
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Submit up to 1,000 transaction payloads in a single API request</p>
          </div>
          <button
            onClick={() => setJsonText(JSON.stringify(SAMPLE_BATCH, null, 2))}
            className="px-3 py-1.5 rounded-lg bg-slate-800 text-xs text-slate-300 hover:bg-slate-700"
          >
            Reset Sample Batch
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Batch Transactions Array (JSON)</label>
            <textarea
              rows={8}
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              className="w-full font-mono text-xs bg-slate-950 border border-slate-800 rounded-xl p-3 text-indigo-300 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <button
            onClick={handleScoreBatch}
            disabled={loading}
            className="w-full py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2 transition-all disabled:opacity-50"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>PROCESS BATCH TRANSACTIONS</span>
              </>
            )}
          </button>

          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {batchResult && (
        <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              Batch Execution Summary
            </h3>
            <div className="flex items-center gap-4 text-xs">
              <span className="text-slate-400">Total Processed: <b className="text-slate-100">{batchResult.total_processed}</b></span>
              <span className="text-slate-400">Batch Latency: <b className="text-indigo-400">{batchResult.batch_latency_ms} ms</b></span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* Summary Donut Chart */}
            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={35} outerRadius={65} paddingAngle={4}>
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Decision Counts */}
            <div className="md:col-span-2 grid grid-cols-3 gap-3">
              <div className="bg-slate-950/60 border border-emerald-500/30 p-3.5 rounded-xl">
                <div className="text-xs font-semibold text-emerald-400">Approve</div>
                <div className="text-2xl font-black text-slate-100">{decisionCounts.approve}</div>
              </div>
              <div className="bg-slate-950/60 border border-amber-500/30 p-3.5 rounded-xl">
                <div className="text-xs font-semibold text-amber-400">Review</div>
                <div className="text-2xl font-black text-slate-100">{decisionCounts.review}</div>
              </div>
              <div className="bg-slate-950/60 border border-rose-500/30 p-3.5 rounded-xl">
                <div className="text-xs font-semibold text-rose-400">Block</div>
                <div className="text-2xl font-black text-slate-100">{decisionCounts.block}</div>
              </div>
            </div>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/40">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Transaction ID</th>
                  <th className="p-3">Fraud Prob</th>
                  <th className="p-3">Decision</th>
                  <th className="p-3">Risk Band</th>
                  <th className="p-3">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {batchResult.results.map((r, i) => (
                  <tr key={i} className="hover:bg-slate-900/50">
                    <td className="p-3 font-mono text-slate-200">{r.transaction_id}</td>
                    <td className="p-3 font-mono font-bold">{(r.fraud_probability * 100).toFixed(2)}%</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        r.decision === 'approve' ? 'bg-emerald-500/10 text-emerald-400' :
                        r.decision === 'review' ? 'bg-amber-500/10 text-amber-400' : 'bg-rose-500/10 text-rose-400'
                      }`}>
                        {r.decision.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 font-semibold">{r.risk_band}</td>
                    <td className="p-3 font-mono text-slate-400">{r.latency_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
