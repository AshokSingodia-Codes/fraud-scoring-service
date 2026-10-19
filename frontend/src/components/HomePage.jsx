import React from 'react';
import {
  ArrowRight,
  Search,
  Layers,
  FileText,
  BarChart2,
  CheckCircle,
  AlertTriangle,
  Ban
} from 'lucide-react';

export default function HomePage({ onNavigate }) {
  const primaryFeatures = [
    {
      id: 'check-one',
      title: 'Score Transaction',
      description: 'Check a single card transaction in real time with automated decisions and risk reasons.',
      icon: Search,
      action: 'Start check',
    },
    {
      id: 'check-bulk',
      title: 'Batch Processing',
      description: 'Upload CSV or JSON files of transactions for bulk classification and export.',
      icon: Layers,
      action: 'Upload file',
    },
    {
      id: 'find-id',
      title: 'Audit Search',
      description: 'Search previously scored transactions by ID to view historical logs.',
      icon: FileText,
      action: 'Lookup ID',
    },
  ];

  return (
    <div className="space-y-8 py-2 max-w-5xl mx-auto">
      {/* Clean White Hero */}
      <section className="bg-white border border-slate-200 rounded-xl p-8 sm:p-10 shadow-xs">
        <div className="max-w-2xl space-y-3">
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            Real-Time Fraud Detection Engine
          </h1>
          <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
            Instant credit card transaction evaluation returning calibrated risk probabilities, cost-optimal policies, and explainable decision factors.
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              onClick={() => onNavigate('check-one')}
              className="btn-primary"
            >
              Check Transaction
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => onNavigate('check-bulk')}
              className="btn-secondary"
            >
              Batch Upload
            </button>
          </div>
        </div>

        {/* User-Focused Key Highlights (Non-technical) */}
        <div className="grid grid-cols-3 gap-4 mt-8 pt-6 border-t border-slate-100 max-w-lg">
          <div>
            <span className="block text-xl font-bold text-emerald-700">&lt; 15 ms</span>
            <span className="text-xs text-slate-500">Fast Decision</span>
          </div>
          <div>
            <span className="block text-xl font-bold text-slate-900">99.8%</span>
            <span className="text-xs text-slate-500">System Accuracy</span>
          </div>
          <div>
            <span className="block text-xl font-bold text-slate-900">Automated</span>
            <span className="text-xs text-slate-500">Risk Assessment</span>
          </div>
        </div>
      </section>

      {/* Decision Guidelines Reference (Informational Only) */}
      <section className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
            Decision Guidelines & Action Policy
          </h2>
          <span className="text-[11px] text-slate-600 bg-slate-100 px-2 py-0.5 rounded-full font-medium">
            Automated Policies
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 select-none">
          <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/50 flex items-start gap-3 cursor-default">
            <CheckCircle className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-emerald-900 uppercase">Approve</span>
                <span className="text-[10px] bg-emerald-200/60 text-emerald-900 px-1.5 py-0.2 rounded font-semibold">Risk &lt; 10%</span>
              </div>
              <p className="text-xs text-emerald-800 leading-relaxed">
                Negligible risk. Cleared for immediate transaction processing without friction.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/50 flex items-start gap-3 cursor-default">
            <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-amber-900 uppercase">Review</span>
                <span className="text-[10px] bg-amber-200/60 text-amber-900 px-1.5 py-0.2 rounded font-semibold">10% - 50%</span>
              </div>
              <p className="text-xs text-amber-800 leading-relaxed">
                Moderate risk or unusual device/location pattern. Queued for 2-step verification or agent check.
              </p>
            </div>
          </div>

          <div className="p-4 rounded-xl border border-rose-200 bg-rose-50/50 flex items-start gap-3 cursor-default">
            <Ban className="w-5 h-5 text-rose-600 mt-0.5 flex-shrink-0" />
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-rose-900 uppercase">Block</span>
                <span className="text-[10px] bg-rose-200/60 text-rose-900 px-1.5 py-0.2 rounded font-semibold">Risk &gt; 50%</span>
              </div>
              <p className="text-xs text-rose-800 leading-relaxed">
                Critical fraud likelihood. Automatically declined to eliminate chargeback losses.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Primary Actions Grid */}
      <section className="space-y-3">
        <h2 className="text-sm font-bold text-slate-700 uppercase tracking-wider">Features</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {primaryFeatures.map((f) => {
            const Icon = f.icon;
            return (
              <div
                key={f.id}
                onClick={() => onNavigate(f.id)}
                className="card-clean card-interactive p-5 flex flex-col justify-between cursor-pointer group"
              >
                <div className="space-y-2">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                      <Icon className="w-4 h-4" />
                    </div>
                    <h3 className="text-sm font-bold text-slate-900 group-hover:text-emerald-600 transition-colors">
                      {f.title}
                    </h3>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    {f.description}
                  </p>
                </div>
                <div className="pt-3 mt-3 border-t border-slate-100 flex items-center text-xs font-medium text-emerald-600">
                  <span>{f.action}</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
