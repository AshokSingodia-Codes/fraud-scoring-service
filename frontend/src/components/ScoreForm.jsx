import React, { useState } from 'react';
import { Zap, Sparkles, Code, Sliders } from 'lucide-react';

const PRESETS = {
  low: {
    transaction_id: "tx_preset_low_risk",
    TransactionDT: 86400,
    TransactionAmt: 34.50,
    ProductCD: "W",
    card1: 1000,
    card2: 555.0,
    card4: "visa",
    card6: "debit",
    P_emaildomain: "gmail.com",
    R_emaildomain: "gmail.com",
    DeviceType: "desktop",
    DeviceInfo: "Windows",
  },
  medium: {
    transaction_id: "tx_preset_review",
    TransactionDT: 88000,
    TransactionAmt: 320.00,
    ProductCD: "H",
    card1: 7500,
    card2: 321.0,
    card4: "visa",
    card6: "credit",
    P_emaildomain: "hotmail.com",
    R_emaildomain: "hotmail.com",
    DeviceType: "desktop",
    DeviceInfo: "MacOS",
  },
  high: {
    transaction_id: "tx_preset_high_risk",
    TransactionDT: 90000,
    TransactionAmt: 1850.00,
    ProductCD: "C",
    card1: 19800,
    card2: 100.0,
    card4: "mastercard",
    card6: "credit",
    P_emaildomain: "anonymous.com",
    R_emaildomain: "yahoo.com",
    DeviceType: "mobile",
    DeviceInfo: "iOS Device",
  },
};

export default function ScoreForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState(PRESETS.low);
  const [explain, setExplain] = useState(true);
  const [jsonMode, setJsonMode] = useState(false);
  const [rawJson, setRawJson] = useState(JSON.stringify(PRESETS.low, null, 2));

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    const val = type === 'number' ? (value === '' ? '' : parseFloat(value)) : value;
    const updated = { ...formData, [name]: val };
    setFormData(updated);
    setRawJson(JSON.stringify(updated, null, 2));
  };

  const handlePresetSelect = (key) => {
    const preset = PRESETS[key];
    setFormData(preset);
    setRawJson(JSON.stringify(preset, null, 2));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    let payload = formData;
    if (jsonMode) {
      try {
        payload = JSON.parse(rawJson);
      } catch (err) {
        alert("Invalid JSON format: " + err.message);
        return;
      }
    }
    onSubmit(payload, explain);
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl">
      {/* Header & Mode Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            Transaction Payload Input
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Select a sample preset or enter custom transaction parameters</p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950/60 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => setJsonMode(false)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              !jsonMode ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" /> Form
          </button>
          <button
            type="button"
            onClick={() => setJsonMode(true)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              jsonMode ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code className="w-3.5 h-3.5" /> JSON Editor
          </button>
        </div>
      </div>

      {/* Preset Quick Load Buttons */}
      <div className="mb-6">
        <label className="block text-xs font-medium text-slate-400 mb-2 flex items-center gap-1">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Quick Load Presets:
        </label>
        <div className="grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => handlePresetSelect('low')}
            className="px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-all text-center"
          >
            🟢 Low Risk E-Commerce
          </button>
          <button
            type="button"
            onClick={() => handlePresetSelect('medium')}
            className="px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-semibold hover:bg-amber-500/20 transition-all text-center"
          >
            🟡 Manual Review Needed
          </button>
          <button
            type="button"
            onClick={() => handlePresetSelect('high')}
            className="px-3 py-2 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-semibold hover:bg-rose-500/20 transition-all text-center"
          >
            🔴 High Risk Crypto
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {jsonMode ? (
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Raw JSON Object</label>
            <textarea
              rows={12}
              value={rawJson}
              onChange={(e) => setRawJson(e.target.value)}
              className="w-full font-mono text-xs bg-slate-950 border border-slate-800 rounded-xl p-3 text-indigo-300 focus:outline-none focus:border-indigo-500"
            />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Transaction ID</label>
              <input
                type="text"
                name="transaction_id"
                value={formData.transaction_id || ''}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Transaction Amt ($)</label>
              <input
                type="number"
                step="0.01"
                name="TransactionAmt"
                value={formData.TransactionAmt || ''}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
                required
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Product Code (ProductCD)</label>
              <select
                name="ProductCD"
                value={formData.ProductCD || 'W'}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value="W">W (Web Purchase)</option>
                <option value="C">C (Cross-Border / Crypto)</option>
                <option value="R">R (Risk Category)</option>
                <option value="H">H (High-Value Item)</option>
                <option value="S">S (Service)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Card 1 (Issuer ID)</label>
              <input
                type="number"
                name="card1"
                value={formData.card1 || ''}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Card Brand (card4)</label>
              <select
                name="card4"
                value={formData.card4 || 'visa'}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value="visa">Visa</option>
                <option value="mastercard">Mastercard</option>
                <option value="discover">Discover</option>
                <option value="american express">American Express</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Card Type (card6)</label>
              <select
                name="card6"
                value={formData.card6 || 'debit'}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value="debit">Debit</option>
                <option value="credit">Credit</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Purchaser Email Domain</label>
              <input
                type="text"
                name="P_emaildomain"
                value={formData.P_emaildomain || ''}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Recipient Email Domain</label>
              <input
                type="text"
                name="R_emaildomain"
                value={formData.R_emaildomain || ''}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Device Type</label>
              <select
                name="DeviceType"
                value={formData.DeviceType || 'desktop'}
                onChange={handleChange}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-slate-100 focus:border-indigo-500 focus:outline-none"
              >
                <option value="desktop">Desktop</option>
                <option value="mobile">Mobile</option>
              </select>
            </div>
          </div>
        )}

        {/* Explain Checkbox */}
        <div className="flex items-center justify-between pt-2">
          <label className="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-indigo-600 focus:ring-indigo-500 w-4 h-4"
            />
            <span>Compute SHAP Feature Explanations (+~60ms)</span>
          </label>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white font-bold text-sm shadow-xl shadow-indigo-600/30 hover:opacity-95 active:scale-[0.99] transition-all flex items-center justify-center space-x-2 disabled:opacity-50"
        >
          {loading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <>
              <Zap className="w-4 h-4 text-amber-300 fill-amber-300" />
              <span>SCORE TRANSACTION NOW</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
