import React, { useState } from 'react';
import { X, Key, Globe, CheckCircle2, AlertCircle, RotateCcw } from 'lucide-react';
import { getApiSettings, saveApiSettings, resetApiSettings, checkHealth } from '../services/api';

export default function SettingsModal({ isOpen, onClose, onSaveSuccess }) {
  const [apiUrl, setApiUrl] = useState(() => getApiSettings().apiUrl);
  const [apiKey, setApiKey] = useState(() => getApiSettings().apiKey);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  if (!isOpen) return null;

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      await checkHealth();
      setTestResult({ success: true, message: "Connected successfully! Fraud Scoring API is online." });
    } catch (err) {
      setTestResult({ success: false, message: `Could not reach API: ${err.message}` });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = (e) => {
    e.preventDefault();
    saveApiSettings(apiUrl, apiKey);
    if (onSaveSuccess) onSaveSuccess("Settings saved successfully.");
    onClose();
  };

  const handleReset = () => {
    resetApiSettings();
    const defaults = getApiSettings();
    setApiUrl(defaults.apiUrl);
    setApiKey(defaults.apiKey);
    setTestResult({ success: true, message: "Reset to default server configuration." });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="card-clean max-w-lg w-full p-6 shadow-xl space-y-6 relative">
        <div className="flex items-center justify-between pb-4 border-b border-slate-200">
          <div>
            <h3 className="text-lg font-bold text-slate-900">API Connection Settings</h3>
            <p className="text-xs text-slate-500 mt-0.5">Configure your server address and authentication key</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
            aria-label="Close settings"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
              <Globe className="w-4 h-4 text-slate-400" />
              API Server Base URL
            </label>
            <input
              type="url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              required
              className="input-clean text-sm"
            />
            <p className="text-xs text-slate-500 mt-1">
              Local default is <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-700">http://localhost:8000</code> or your hosted Render URL.
            </p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
              <Key className="w-4 h-4 text-slate-400" />
              X-API-Key (Secret Key)
            </label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="demo-api-key-12345"
              required
              className="input-clean text-sm font-mono"
            />
            <p className="text-xs text-slate-500 mt-1">
              Used to authorize requests with the fraud scoring backend. Never shared externally.
            </p>
          </div>

          {testResult && (
            <div className={`p-3 rounded-lg text-xs flex items-center gap-2 ${
              testResult.success ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-rose-50 text-rose-800 border border-rose-200'
            }`}>
              {testResult.success ? <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" /> : <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />}
              <span>{testResult.message}</span>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-slate-200">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={testing}
                className="btn-secondary text-xs py-2 px-3"
              >
                {testing ? 'Testing...' : 'Test Connection'}
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="btn-secondary text-xs py-2 px-3 text-slate-500"
                title="Reset to defaults"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset
              </button>
            </div>

            <div className="flex gap-2">
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
                Save Settings
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
