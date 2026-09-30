import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldX, Clock, Cpu, BarChart2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

export default function ScoreResults({ result }) {
  if (!result) {
    return (
      <div className="glass-card rounded-2xl p-8 border border-slate-800 text-center flex flex-col items-center justify-center min-h-[400px]">
        <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-600 mb-4">
          <BarChart2 className="w-8 h-8" />
        </div>
        <h3 className="text-slate-300 font-bold text-lg">No Active Score Output</h3>
        <p className="text-slate-500 text-xs max-w-sm mt-1">
          Select a transaction preset or enter parameters on the left and click "Score Transaction Now".
        </p>
      </div>
    );
  }

  const { fraud_probability, decision, risk_band, reasons, latency_ms, thresholds, transaction_id } = result;

  const probPct = (fraud_probability * 100).toFixed(2);

  // Decision badges & colors
  let badgeStyle = '';
  let badgeIcon = null;
  let decisionText = '';

  if (decision === 'approve') {
    badgeStyle = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 badge-glow-approve';
    badgeIcon = <ShieldCheck className="w-5 h-5" />;
    decisionText = 'APPROVE TRANSACTION';
  } else if (decision === 'review') {
    badgeStyle = 'bg-amber-500/10 text-amber-400 border-amber-500/30 badge-glow-review';
    badgeIcon = <AlertTriangle className="w-5 h-5" />;
    decisionText = 'MANUAL REVIEW REQUIRED';
  } else {
    badgeStyle = 'bg-rose-500/10 text-rose-400 border-rose-500/30 badge-glow-block';
    badgeIcon = <ShieldX className="w-5 h-5" />;
    decisionText = 'BLOCK TRANSACTION';
  }

  // Formatting reasons for Recharts
  const chartData = (reasons || []).map((r) => ({
    name: r.label.length > 22 ? r.label.slice(0, 20) + '...' : r.label,
    fullName: r.label,
    contribution: r.contribution,
    direction: r.direction,
    value: r.value,
  }));

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl space-y-6">
      {/* Top Banner & Decision Badge */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <span className="text-xs text-slate-500 font-mono">ID: {transaction_id}</span>
          <h2 className="text-xl font-extrabold text-slate-100">Scoring Engine Decision</h2>
        </div>
        <div className={`px-4 py-2 rounded-xl border text-sm font-extrabold flex items-center space-x-2 ${badgeStyle}`}>
          {badgeIcon}
          <span>{decisionText}</span>
        </div>
      </div>

      {/* Main Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Probability Card */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-400 mb-1">Calibrated Fraud Risk</div>
          <div className="text-3xl font-black text-slate-100">{probPct}%</div>
          <div className="text-[11px] text-slate-500 mt-1 font-mono">Prob: {fraud_probability.toFixed(4)}</div>
        </div>

        {/* Risk Band */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-400 mb-1">Assigned Risk Band</div>
          <div className={`text-2xl font-extrabold ${risk_band === 'High' ? 'text-rose-400' : risk_band === 'Medium' ? 'text-amber-400' : 'text-emerald-400'}`}>
            {risk_band} Risk
          </div>
          <div className="text-[11px] text-slate-500 mt-1">t_review: {thresholds?.t_review || 0.0496}</div>
        </div>

        {/* Latency */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between">
          <div className="text-xs font-semibold text-slate-400 mb-1 flex items-center gap-1">
            <Clock className="w-3.5 h-3.5 text-indigo-400" /> Pipeline Latency
          </div>
          <div className="text-2xl font-extrabold text-indigo-400">{latency_ms} ms</div>
          <div className="text-[11px] text-slate-500 font-mono">Engine: LightGBM + SHAP</div>
        </div>
      </div>

      {/* Probability Progress Bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-medium">
          <span>0.00 (Approve)</span>
          <span>t_review: {thresholds?.t_review}</span>
          <span>t_block: {thresholds?.t_block}</span>
          <span>1.00 (Block)</span>
        </div>
        <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              decision === 'block'
                ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                : decision === 'review'
                ? 'bg-gradient-to-r from-emerald-500 to-amber-500'
                : 'bg-gradient-to-r from-emerald-500 to-emerald-400'
            }`}
            style={{ width: `${Math.min(Math.max(fraud_probability * 100, 3), 100)}%` }}
          />
        </div>
      </div>

      {/* SHAP Reason Codes Visualization */}
      {reasons && reasons.length > 0 && (
        <div className="pt-4 border-t border-slate-800">
          <h3 className="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            Top-5 SHAP Per-Prediction Reason Codes
          </h3>

          <div className="h-56 w-full mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                <XAxis type="number" stroke="#64748b" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={130} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs">
                          <p className="font-bold text-slate-100">{data.fullName}</p>
                          <p className="text-slate-400 mt-1">Value: {data.value !== null ? String(data.value) : 'N/A'}</p>
                          <p className={data.direction === 'increases_risk' ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                            SHAP: {data.contribution > 0 ? '+' : ''}{data.contribution} ({data.direction === 'increases_risk' ? 'Increases Fraud Risk' : 'Lowers Fraud Risk'})
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.direction === 'increases_risk' ? '#ef4444' : '#10b981'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Breakdown Table */}
          <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/40">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900 text-slate-400 uppercase font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-2.5">Feature</th>
                  <th className="p-2.5">Observed Value</th>
                  <th className="p-2.5">SHAP Impact</th>
                  <th className="p-2.5">Direction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {reasons.map((r, i) => (
                  <tr key={i} className="hover:bg-slate-900/50">
                    <td className="p-2.5 font-semibold text-slate-200">{r.label}</td>
                    <td className="p-2.5 font-mono text-slate-400">{r.value !== null ? String(r.value) : 'null'}</td>
                    <td className={`p-2.5 font-mono font-bold ${r.direction === 'increases_risk' ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {r.contribution > 0 ? '+' : ''}{r.contribution}
                    </td>
                    <td className="p-2.5">
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${r.direction === 'increases_risk' ? 'bg-rose-500/10 text-rose-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                        {r.direction === 'increases_risk' ? '▲ INCREASES RISK' : '▼ LOWERS RISK'}
                      </span>
                    </td>
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
