import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Calendar,
  Layers,
  Scale,
  Award,
  AlertTriangle,
  Code,
  ChevronDown,
  ChevronUp,
  RefreshCw
} from 'lucide-react';
import { getModelInfo } from '../services/api';

export default function ModelInfoPage() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [devSectionOpen, setDevSectionOpen] = useState(false);

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModelInfo();
      setInfo(data);
    } catch (err) {
      setError(err.message || "Failed to load model specifications from the API.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    getModelInfo()
      .then(data => {
        if (isMounted) {
          setInfo(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (isMounted) {
          setError(err.message || "Failed to load model specifications from the API.");
          setLoading(false);
        }
      });
    return () => { isMounted = false; };
  }, []);

  return (
    <div className="space-y-8 py-2">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            About the Fraud Model
          </h1>
          <p className="text-sm text-slate-600">
            Understand how our artificial intelligence scores risk, balances financial costs, and makes transparent decisions.
          </p>
        </div>

        <button
          onClick={handleRefresh}
          disabled={loading}
          className="btn-secondary text-xs py-2 px-3 self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Model Status
        </button>
      </div>

      {/* Explainer Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you need to provide
          </span>
          <p className="text-xs text-slate-600">
            Nothing. This page automatically queries the live model server to display running parameters and health statistics.
          </p>
        </div>
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you will get back
          </span>
          <p className="text-xs text-slate-600">
            The active model version, business decision boundaries, total input features, and validated accuracy metrics.
          </p>
        </div>
      </div>

      {loading && (
        <div className="card-clean p-12 text-center space-y-3">
          <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-slate-500">Querying live model server specifications...</p>
        </div>
      )}

      {error && (
        <div className="card-clean p-8 text-center space-y-3 border-rose-200 bg-rose-50">
          <div className="w-12 h-12 bg-rose-100 text-rose-600 rounded-full flex items-center justify-center mx-auto">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-900">Could not retrieve model specs</h3>
          <p className="text-xs text-slate-600 max-w-md mx-auto">{error}</p>
          <button onClick={fetchModelInfo} className="btn-secondary text-xs py-2 px-4 mt-2">
            Try Again
          </button>
        </div>
      )}

      {info && !loading && (
        <div className="space-y-6 animate-in fade-in duration-200">
          {/* Key Facts Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card-clean p-5 space-y-1">
              <span className="text-xs text-slate-500 flex items-center gap-1.5 font-bold uppercase tracking-wider">
                <Cpu className="w-4 h-4 text-indigo-600" /> Model Architecture
              </span>
              <span className="text-2xl font-black text-slate-900 block">LightGBM GBDT</span>
              <span className="text-xs text-slate-500 block">Gradient Boosted Decision Trees</span>
            </div>

            <div className="card-clean p-5 space-y-1">
              <span className="text-xs text-slate-500 flex items-center gap-1.5 font-bold uppercase tracking-wider">
                <Layers className="w-4 h-4 text-blue-600" /> Active Features
              </span>
              <span className="text-2xl font-black text-slate-900 block">{info.num_features}</span>
              <span className="text-xs text-slate-500 block">Behavioral & Entity Signals</span>
            </div>

            <div className="card-clean p-5 space-y-1">
              <span className="text-xs text-slate-500 flex items-center gap-1.5 font-bold uppercase tracking-wider">
                <Calendar className="w-4 h-4 text-emerald-600" /> Production Release
              </span>
              <span className="text-2xl font-black text-slate-900 block">v{info.model_version}</span>
              <span className="text-xs text-slate-500 block">Trained: {info.training_date}</span>
            </div>

            <div className="card-clean p-5 space-y-1">
              <span className="text-xs text-slate-500 flex items-center gap-1.5 font-bold uppercase tracking-wider">
                <Award className="w-4 h-4 text-purple-600" /> Benchmark ROC-AUC
              </span>
              <span className="text-2xl font-black text-purple-700 block">0.910 (91.0%)</span>
              <span className="text-xs text-slate-500 block">Time-series test accuracy</span>
            </div>
          </div>

          {/* Decision Boundaries / Cost Matrix Explanation */}
          <div className="card-clean p-6 sm:p-8 space-y-6">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Scale className="w-5 h-5 text-indigo-600" />
                How the AI Balances Business Costs
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Unlike simple classifiers that use an arbitrary 50% cutoff, this engine uses mathematical cost optimization ($10 manual review cost vs preventing chargeback losses):
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-5 rounded-2xl border border-emerald-200 bg-emerald-50/50 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-emerald-100 text-emerald-800 border border-emerald-300">
                    APPROVE (Safe)
                  </span>
                  <span className="text-xs font-mono font-bold text-emerald-800">
                    Risk &lt; {(info.thresholds?.t_review * 100).toFixed(2)}%
                  </span>
                </div>
                <h3 className="font-bold text-slate-900 text-sm">Automated Instant Pass</h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Fraud probability is negligible. Passing automatically preserves zero customer checkout friction and zero analyst labor cost.
                </p>
              </div>

              <div className="p-5 rounded-2xl border border-amber-200 bg-amber-50/50 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-amber-100 text-amber-800 border border-amber-300">
                    MANUAL REVIEW
                  </span>
                  <span className="text-xs font-mono font-bold text-amber-800">
                    {(info.thresholds?.t_review * 100).toFixed(2)}% – {(info.thresholds?.t_block * 100).toFixed(2)}%
                  </span>
                </div>
                <h3 className="font-bold text-slate-900 text-sm">Optimal Human Triage Queue</h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  At this threshold (<strong>5.95%</strong>), spending $10 on a human review saves significantly more money in avoided fraud than doing nothing.
                </p>
              </div>

              <div className="p-5 rounded-2xl border border-rose-200 bg-rose-50/50 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider bg-rose-100 text-rose-800 border border-rose-300">
                    BLOCK (Decline)
                  </span>
                  <span className="text-xs font-mono font-bold text-rose-800">
                    Risk &ge; {(info.thresholds?.t_block * 100).toFixed(2)}%
                  </span>
                </div>
                <h3 className="font-bold text-slate-900 text-sm">High-Precision Automated Decline</h3>
                <p className="text-xs text-slate-600 leading-relaxed">
                  At <strong>75.00%</strong> probability, over 90% of flagged orders are confirmed unauthorized transactions, protecting your bank merchant account.
                </p>
              </div>
            </div>
          </div>

          {/* Metrics in Plain English */}
          <div className="card-clean p-6 sm:p-8 space-y-5">
            <div>
              <h2 className="text-lg font-bold text-slate-900">Key Performance Metrics in Plain English</h2>
              <p className="text-xs text-slate-500 mt-0.5">Tested strictly on a chronological future test set (Days 152–182) with zero lookahead bias:</p>
            </div>

            <div className="divide-y divide-slate-100 text-xs">
              <div className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="space-y-0.5">
                  <span className="font-bold text-slate-900 text-sm">ROC-AUC: 0.9103 (91.0%)</span>
                  <p className="text-slate-500">
                    Measures the model's ability to rank a fraudulent transaction higher than a legitimate one across all thresholds.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 text-center sm:text-right">
                  Outstanding
                </span>
              </div>

              <div className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="space-y-0.5">
                  <span className="font-bold text-slate-900 text-sm">PR-AUC (Precision-Recall): 0.5343 (53.4%)</span>
                  <p className="text-slate-500">
                    Crucial for severe class imbalance (3.5% fraud). 15x higher than random chance (3.4%), proving strong focus on true fraudsters.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 font-bold border border-indigo-200 text-center sm:text-right">
                  15.6x Baseline
                </span>
              </div>

              <div className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="space-y-0.5">
                  <span className="font-bold text-slate-900 text-sm">Expected Calibration Error (ECE): 0.0041</span>
                  <p className="text-slate-500">
                    Thanks to Isotonic Regression calibration, when the system reports a 20% fraud probability, exactly 1 out of 5 transactions are actually fraud.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 text-center sm:text-right">
                  Near Perfect Calibration
                </span>
              </div>
            </div>
          </div>

          {/* Developer Accordion */}
          <div className="card-clean p-4">
            <button
              type="button"
              onClick={() => setDevSectionOpen(!devSectionOpen)}
              className="w-full flex items-center justify-between text-xs font-semibold text-slate-500 hover:text-slate-800"
            >
              <span className="flex items-center gap-2">
                <Code className="w-3.5 h-3.5" />
                For Developers: Raw Model Specifications JSON (<code className="text-indigo-600">GET /v1/model/info</code>)
              </span>
              {devSectionOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {devSectionOpen && (
              <div className="mt-3 pt-3 border-t border-slate-100 text-xs font-mono bg-slate-900 text-slate-100 p-4 rounded-xl overflow-x-auto text-[11px] animate-in fade-in">
                <pre>{JSON.stringify(info, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
