import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ScoreForm from './components/ScoreForm';
import ScoreResults from './components/ScoreResults';
import BatchScorer from './components/BatchScorer';
import AuditLog from './components/AuditLog';
import ModelInfo from './components/ModelInfo';
import { scoreTransaction } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [singleResult, setSingleResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleScoreSubmit = async (payload, explain) => {
    setLoading(true);
    try {
      const result = await scoreTransaction(payload, explain);
      setSingleResult(result);
    } catch (err) {
      alert("Scoring Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {activeTab === 'single' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-6">
              <ScoreForm onSubmit={handleScoreSubmit} loading={loading} />
            </div>
            <div className="lg:col-span-6">
              <ScoreResults result={singleResult} />
            </div>
          </div>
        )}

        {activeTab === 'batch' && <BatchScorer />}
        {activeTab === 'audit' && <AuditLog />}
        {activeTab === 'info' && <ModelInfo />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-wrap justify-between items-center gap-4">
          <div>IEEE-CIS Real-Time Fraud Detection Engine &copy; 2026</div>
          <div className="flex space-x-4">
            <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="hover:text-indigo-400">API Swagger Docs</a>
            <a href="http://127.0.0.1:8000/metrics" target="_blank" rel="noreferrer" className="hover:text-indigo-400">Prometheus Metrics</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
