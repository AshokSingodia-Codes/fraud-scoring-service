import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  BookmarkPlus,
  Code,
  Clock,
  Zap,
  Download
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';
import { scoreTransaction } from '../services/api';
import { formatReasonExplanation, getDecisionBadgeInfo } from '../utils/reasonFormatter';
import { exportReportPdf } from '../utils/pdfExport';
import SaveReportModal from './SaveReportModal';

const PRESET_SAFE = {
  transaction_id: "tx_sample_safe_101",
  TransactionDT: 86400,
  TransactionAmt: 34.50,
  ProductCD: "W",
  card1: 1000,
  card2: 555.0,
  card4: "visa",
  card6: "debit",
  addr1: 150.0,
  addr2: 87.0,
  P_emaildomain: "gmail.com",
  R_emaildomain: "gmail.com",
  DeviceType: "desktop",
  DeviceInfo: "Windows",
};

const PRESET_SUSPICIOUS = {
  transaction_id: "tx_sample_fraud_909",
  TransactionDT: 90000,
  TransactionAmt: 1850.00,
  ProductCD: "C",
  card1: 19800,
  card2: 100.0,
  card4: "mastercard",
  card6: "credit",
  addr1: 441.0,
  addr2: 87.0,
  P_emaildomain: "anonymous.com",
  R_emaildomain: "yahoo.com",
  DeviceType: "mobile",
  DeviceInfo: "iOS Device",
};

export default function CheckOnePage({ onToast }) {
  const [formData, setFormData] = useState(PRESET_SAFE);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [devSectionOpen, setDevSectionOpen] = useState(false);
  const [advancedJson, setAdvancedJson] = useState('{\n  "C1": 1,\n  "D1": 0\n}');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [saveModalOpen, setSaveModalOpen] = useState(false);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let parsed = value;
    if (type === 'number') {
      parsed = value === '' ? '' : parseFloat(value);
    }
    setFormData(prev => ({ ...prev, [name]: parsed }));
  };

  const handleFillSample = (sample) => {
    setFormData(sample);
    setError(null);
    if (onToast) onToast("Filled form with sample transaction values.");
  };

  const handleSubmit = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    setError(null);

    // Validate required fields
    if (!formData.transaction_id || !formData.transaction_id.toString().trim()) {
      setError("Please enter a Transaction ID before submitting.");
      return;
    }

    const amt = parseFloat(formData.TransactionAmt);
    if (formData.TransactionAmt === '' || isNaN(amt) || amt <= 0) {
      setError("Transaction amount must be a positive number greater than $0.00.");
      return;
    }

    setLoading(true);

    // Merge basic and optional advanced json if valid
    let payload = { ...formData };
    if (!payload.TransactionDT) {
      payload.TransactionDT = Math.floor(Date.now() / 1000) % (86400 * 180);
    }

    if (advancedOpen && advancedJson.trim()) {
      try {
        const extra = JSON.parse(advancedJson);
        payload = { ...payload, ...extra };
      } catch {
        setLoading(false);
        setError("The optional advanced JSON contains syntax errors. Please correct it or close the section.");
        return;
      }
    }

    try {
      const res = await scoreTransaction(payload, true);
      setResult(res);
    } catch (err) {
      setError(err.message || "Failed to score transaction. Please check your network and settings.");
    } finally {
      setLoading(false);
    }
  };

  const badgeInfo = result ? getDecisionBadgeInfo(result.decision) : null;
  const probPct = result ? (result.fraud_probability * 100).toFixed(2) : '0.00';

  // Prepare chart data for SHAP reasons
  const chartData = (result?.reasons || []).map(r => ({
    name: r.label && r.label.length > 20 ? r.label.substring(0, 18) + '...' : (r.label || r.feature),
    fullName: r.label || r.feature,
    contribution: Number(r.contribution.toFixed(3)),
    direction: r.direction,
    isRisk: r.direction === 'increases_risk' || r.contribution > 0,
  }));

  return (
    <div className="space-y-8 py-2">
      {/* Page Title & Context Header */}
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Check One Transaction
        </h1>
        <p className="text-sm text-slate-600">
          Evaluate an individual credit card purchase in real time to calculate fraud probability and clear recommendations.
        </p>
      </div>

      {/* What You Need / What You Get Back Explainer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you need to provide
          </span>
          <p className="text-xs text-slate-600">
            A Transaction ID, purchase amount in USD, card information, buyer email address, and device type.
          </p>
        </div>
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you will get back
          </span>
          <p className="text-xs text-slate-600">
            An automated decision (<strong>Approve</strong>, <strong>Review</strong>, or <strong>Block</strong>), calibrated risk percentage, and top contributing factors in plain language.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Form */}
        <div className="lg:col-span-7 card-clean p-6 sm:p-7 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
            <div>
              <h2 className="text-base font-bold text-slate-900">Transaction Information</h2>
              <p className="text-xs text-slate-500">Fill in the fields or choose a quick sample below</p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => handleFillSample(PRESET_SAFE)}
                className="text-xs px-2.5 py-1.5 rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 font-semibold transition-colors"
                title="Fill with standard, low-risk transaction attributes"
              >
                Safe Sample
              </button>
              <button
                type="button"
                onClick={() => handleFillSample(PRESET_SUSPICIOUS)}
                className="text-xs px-2.5 py-1.5 rounded-lg border border-rose-200 bg-rose-50 text-rose-800 hover:bg-rose-100 font-semibold transition-colors"
                title="Fill with high-risk, suspicious transaction attributes"
              >
                Suspicious Sample
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Row 1: ID & Amount */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Transaction ID *</span>
                  <span className="text-[10px] text-slate-400 font-normal">Reference number</span>
                </label>
                <input
                  type="text"
                  name="transaction_id"
                  value={formData.transaction_id || ''}
                  onChange={handleChange}
                  placeholder="e.g. tx_order_88291"
                  required
                  className="input-clean"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Amount ($ USD) *</span>
                  <span className="text-[10px] text-slate-400 font-normal">Purchase total</span>
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  name="TransactionAmt"
                  value={formData.TransactionAmt || ''}
                  onChange={handleChange}
                  placeholder="e.g. 150.00"
                  required
                  className="input-clean font-semibold text-slate-900"
                />
              </div>
            </div>

            {/* Row 2: Product & Card Network */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Product Category</span>
                  <span className="text-[10px] text-slate-400 font-normal">Type of merchant service</span>
                </label>
                <select
                  name="ProductCD"
                  value={formData.ProductCD || 'W'}
                  onChange={handleChange}
                  className="input-clean"
                >
                  <option value="W">W — Physical Retail Goods</option>
                  <option value="C">C — Digital Products / Coins</option>
                  <option value="H">H — Travel & Lodging</option>
                  <option value="R">R — Recurring / Streaming</option>
                  <option value="S">S — Subscriptions & Other</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Card Network & Type</span>
                  <span className="text-[10px] text-slate-400 font-normal">Credit or debit card</span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <select
                    name="card4"
                    value={formData.card4 || 'visa'}
                    onChange={handleChange}
                    className="input-clean"
                  >
                    <option value="visa">Visa</option>
                    <option value="mastercard">Mastercard</option>
                    <option value="american express">Amex</option>
                    <option value="discover">Discover</option>
                  </select>
                  <select
                    name="card6"
                    value={formData.card6 || 'debit'}
                    onChange={handleChange}
                    className="input-clean"
                  >
                    <option value="debit">Debit</option>
                    <option value="credit">Credit</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Row 3: Card Identifiers (card1, card2) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Card Number Hash (card1)</span>
                  <span className="text-[10px] text-slate-400 font-normal">1,000–19,999</span>
                </label>
                <input
                  type="number"
                  name="card1"
                  value={formData.card1 || ''}
                  onChange={handleChange}
                  placeholder="e.g. 1000"
                  className="input-clean"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Issuing Bank Code (card2)</span>
                  <span className="text-[10px] text-slate-400 font-normal">Optional</span>
                </label>
                <input
                  type="number"
                  name="card2"
                  value={formData.card2 || ''}
                  onChange={handleChange}
                  placeholder="e.g. 555.0"
                  className="input-clean"
                />
              </div>
            </div>

            {/* Row 4: Email Domains */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Buyer Email Domain</span>
                  <span className="text-[10px] text-slate-400 font-normal">e.g. gmail.com</span>
                </label>
                <input
                  type="text"
                  name="P_emaildomain"
                  value={formData.P_emaildomain || ''}
                  onChange={handleChange}
                  placeholder="e.g. gmail.com"
                  className="input-clean"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Delivery Email Domain</span>
                  <span className="text-[10px] text-slate-400 font-normal">Matches buyer?</span>
                </label>
                <input
                  type="text"
                  name="R_emaildomain"
                  value={formData.R_emaildomain || ''}
                  onChange={handleChange}
                  placeholder="e.g. gmail.com"
                  className="input-clean"
                />
              </div>
            </div>

            {/* Row 5: Device & Browser */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                  Device Hardware Type
                </label>
                <select
                  name="DeviceType"
                  value={formData.DeviceType || 'desktop'}
                  onChange={handleChange}
                  className="input-clean"
                >
                  <option value="desktop">Desktop / Laptop</option>
                  <option value="mobile">Mobile Smartphone</option>
                  <option value="tablet">Tablet</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">
                  Operating System / Info
                </label>
                <input
                  type="text"
                  name="DeviceInfo"
                  value={formData.DeviceInfo || 'Windows'}
                  onChange={handleChange}
                  placeholder="e.g. Windows, iOS Device, MacOS"
                  className="input-clean"
                />
              </div>
            </div>

            {/* Collapsible Advanced IEEE-CIS Fields */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setAdvancedOpen(!advancedOpen)}
                className="w-full py-2.5 px-3 rounded-lg border border-slate-200 bg-slate-50 text-xs font-semibold text-slate-700 flex items-center justify-between hover:bg-slate-100 transition-colors"
              >
                <span>Advanced Fields (Optional IEEE-CIS Features: C, D, V, id_xx)</span>
                {advancedOpen ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
              </button>

              {advancedOpen && (
                <div className="mt-3 p-3.5 bg-slate-50 rounded-lg border border-slate-200 space-y-2 animate-in fade-in">
                  <p className="text-[11px] text-slate-500">
                    You can supply additional raw IEEE-CIS columns like card counts (C1–C14), time intervals (D1–D15), or security features in JSON format:
                  </p>
                  <textarea
                    rows={3}
                    value={advancedJson}
                    onChange={(e) => setAdvancedJson(e.target.value)}
                    className="input-clean font-mono text-xs"
                    placeholder='{"C1": 1, "D1": 0, "V258": 1.0}'
                  />
                </div>
              )}
            </div>

            {error && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3.5 text-base shadow-sm"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Evaluating Risk Factors...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Check Transaction Risk Now</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Column: Result Presentation */}
        <div className="lg:col-span-5 space-y-6">
          {!result && !loading && (
            <div className="card-clean p-10 text-center space-y-4 text-slate-500">
              <div className="w-14 h-14 bg-indigo-50 text-indigo-500 rounded-2xl flex items-center justify-center mx-auto">
                <ShieldCheck className="w-7 h-7" />
              </div>
              <div className="space-y-1">
                <h3 className="font-bold text-slate-800 text-base">Ready to Score</h3>
                <p className="text-xs text-slate-500 max-w-xs mx-auto">
                  Enter details on the left and click "Check Transaction Risk Now" to view automated action and explanations.
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="card-clean p-10 text-center space-y-4">
              <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <div className="space-y-1">
                <h3 className="font-bold text-slate-800 text-base">Running AI Analysis...</h3>
                <p className="text-xs text-slate-500">Extracting 457 features and computing SHAP reason codes.</p>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="card-clean p-6 sm:p-7 space-y-6 animate-in fade-in duration-200">
              {/* Decision Header Banner */}
              <div className={`p-5 rounded-2xl border ${badgeInfo.bgLight} ${badgeInfo.border} space-y-3`}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-slate-500 font-semibold">
                    ID: {result.transaction_id}
                  </span>
                  <span className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider border ${badgeInfo.colorClass}`}>
                    {badgeInfo.label}
                  </span>
                </div>

                <div>
                  <h3 className={`text-xl font-extrabold ${badgeInfo.text}`}>
                    {result.decision.toUpperCase() === 'APPROVE' && 'Transaction Cleared'}
                    {result.decision.toUpperCase() === 'REVIEW' && 'Manual Review Recommended'}
                    {result.decision.toUpperCase() === 'BLOCK' && 'Transaction Declined (High Risk)'}
                  </h3>
                  <p className="text-xs text-slate-600 mt-1">
                    {badgeInfo.description}
                  </p>
                </div>
              </div>

              {/* Gauge & Key Numbers */}
              <div className="grid grid-cols-2 gap-4 p-4 rounded-xl bg-slate-50 border border-slate-100">
                <div>
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                    Fraud Probability
                  </span>
                  <div className="flex items-baseline gap-1 mt-0.5">
                    <span className="text-2xl sm:text-3xl font-black text-slate-900">{probPct}%</span>
                    <span className="text-xs text-slate-400">risk</span>
                  </div>
                </div>

                <div>
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
                    Risk Category
                  </span>
                  <div className="flex items-baseline gap-1 mt-0.5">
                    <span className="text-2xl sm:text-3xl font-black text-slate-900">{result.risk_band}</span>
                  </div>
                </div>
              </div>

              {/* Plain-English Top Reasons */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                    Top Contributing Factors
                  </h4>
                  <span className="text-[11px] text-slate-400 font-medium">Why the AI reached this result</span>
                </div>

                <div className="space-y-2.5">
                  {(result.reasons || []).slice(0, 5).map((r, i) => {
                    const isRisk = r.direction === 'increases_risk' || r.contribution > 0;
                    return (
                      <div
                        key={i}
                        className={`p-3 rounded-xl border text-xs space-y-1 ${
                          isRisk ? 'bg-rose-50/50 border-rose-100' : 'bg-emerald-50/50 border-emerald-100'
                        }`}
                      >
                        <div className="flex items-center justify-between font-bold">
                          <span className="text-slate-800 flex items-center gap-1.5">
                            {isRisk ? (
                              <ArrowUpRight className="w-3.5 h-3.5 text-rose-600" />
                            ) : (
                              <ArrowDownRight className="w-3.5 h-3.5 text-emerald-600" />
                            )}
                            {r.label || r.feature}
                          </span>
                          <span className={isRisk ? 'text-rose-700' : 'text-emerald-700'}>
                            {isRisk ? '+ Raised Risk' : '- Lowered Risk'}
                          </span>
                        </div>
                        <p className="text-slate-600 text-[11px] leading-relaxed">
                          {formatReasonExplanation(r)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Mini Contribution Chart */}
              {chartData.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">
                    Factor Impact Weights (SHAP Analysis)
                  </span>
                  <div className="h-32 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 10, fill: '#64748b' }} />
                        <Tooltip
                          formatter={(val) => [`Impact: ${val}`, 'Contribution']}
                          contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                        />
                        <ReferenceLine x={0} stroke="#cbd5e1" />
                        <Bar dataKey="contribution" radius={[4, 4, 4, 4]}>
                          {chartData.map((entry, idx) => (
                            <Cell key={idx} fill={entry.isRisk ? '#f43f5e' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Action Buttons: Download PDF & Save Report */}
              <div className="pt-2 flex flex-col sm:flex-row gap-2">
                <button
                  type="button"
                  onClick={() => exportReportPdf({
                    transactionId: result.transaction_id,
                    decision: result.decision,
                    fraudProbability: result.fraud_probability,
                    riskBand: result.risk_band,
                    latencyMs: result.latency_ms,
                    reasons: result.reasons,
                    savedAt: new Date().toISOString(),
                  })}
                  className="btn-secondary flex-1 py-2.5 text-xs text-slate-700 font-bold border-slate-300 hover:bg-slate-50"
                  title="Download printable one-page PDF Fraud Check Report"
                >
                  <Download className="w-4 h-4 text-slate-600" />
                  Download Fraud Check Report (PDF)
                </button>
                <button
                  type="button"
                  onClick={() => setSaveModalOpen(true)}
                  className="btn-primary flex-1 py-2.5 text-xs font-bold"
                >
                  <BookmarkPlus className="w-4 h-4" />
                  Save to Vault
                </button>
              </div>

              {/* Subtle Tech Footer */}
              <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" /> {result.latency_ms} ms
                </span>
                <span>Model v{result.model_version}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Developer Collapsible Drawer */}
      <div className="card-clean p-4">
        <button
          type="button"
          onClick={() => setDevSectionOpen(!devSectionOpen)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-500 hover:text-slate-800"
        >
          <span className="flex items-center gap-2">
            <Code className="w-3.5 h-3.5" />
            For Developers: HTTP API Endpoint Details (<code className="text-indigo-600">POST /v1/score</code>)
          </span>
          {devSectionOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {devSectionOpen && (
          <div className="mt-3 pt-3 border-t border-slate-100 space-y-2 text-xs font-mono text-slate-600 animate-in fade-in">
            <p className="font-bold text-slate-800">Endpoint: POST /v1/score?explain=true</p>
            <p>Header: X-API-Key: demo-api-key-12345</p>
            <div className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto text-[11px]">
              <pre>{JSON.stringify(formData, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>

      {/* Save Report Modal */}
      {saveModalOpen && (
        <SaveReportModal
          isOpen={saveModalOpen}
          onClose={() => setSaveModalOpen(false)}
          reportData={{
            type: 'single',
            transactionId: result.transaction_id,
            decision: result.decision,
            fraudProbability: result.fraud_probability,
            riskBand: result.risk_band,
            modelVersion: result.model_version,
            thresholds: result.thresholds,
            latencyMs: result.latency_ms,
            input: formData,
            output: result,
            reasons: result.reasons,
          }}
          onSaved={() => {
            if (onToast) onToast("Report saved to your local audit vault.");
          }}
        />
      )}
    </div>
  );
}
