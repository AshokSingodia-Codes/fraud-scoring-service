import React from 'react';
import {
  Search,
  Layers,
  FileText,
  Info,
  ArrowRight,
  ShieldAlert,
  CheckCircle2,
  Clock,
  Sparkles,
  Zap
} from 'lucide-react';

export default function HomePage({ onNavigate }) {
  const tools = [
    {
      id: 'check-one',
      title: 'Check One Transaction',
      badge: 'Most Popular',
      description: 'Enter purchase parameters like amount, card number, email, and device to get an instant fraud probability and plain-English reasons.',
      icon: Search,
      action: 'Start Single Check',
      color: 'bg-indigo-600',
    },
    {
      id: 'check-bulk',
      title: 'Check Many (Bulk Upload)',
      badge: 'Batch Processing',
      description: 'Upload a CSV or JSON spreadsheet of up to 1,000 transactions at once. Download filtered outcomes as Excel-ready CSV or executive PDF.',
      icon: Layers,
      action: 'Upload Batch File',
      color: 'bg-blue-600',
    },
    {
      id: 'find-id',
      title: 'Find Past Result by ID',
      badge: 'Audit Retrieval',
      description: 'Look up any previously scored transaction using its unique reference ID to verify decision logs and scoring timestamps.',
      icon: FileText,
      action: 'Lookup Transaction',
      color: 'bg-slate-700',
    },
    {
      id: 'saved-reports',
      title: 'Saved Reports Vault',
      badge: 'Private Storage',
      description: 'Review saved single and bulk audits stored locally on your device. Rename notes, print formal PDF Fraud Check Reports, or backup to JSON.',
      icon: FileText,
      action: 'View Saved Reports',
      color: 'bg-teal-600',
    },
    {
      id: 'model-info',
      title: 'About the Fraud Model',
      badge: 'Transparency',
      description: 'Read how the machine learning model balances business cost ($10 review vs fraud loss) with live PR-AUC and ROC-AUC metrics.',
      icon: Info,
      action: 'Inspect Specifications',
      color: 'bg-purple-600',
    },
  ];

  return (
    <div className="space-y-10 py-2">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-indigo-50/70 via-white to-white border border-indigo-100 rounded-3xl p-8 sm:p-12 shadow-xs relative overflow-hidden">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-100/70 text-indigo-800 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            Designed for Non-Technical Business & Operations Teams
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Stop fraudulent card charges before they cost your business.
          </h1>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed">
            Evaluate credit card transactions in under a quarter of a second. Get calibrated fraud risk percentages, automatic business decisions, and clear human explanations without touching a line of code.
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              onClick={() => onNavigate('check-one')}
              className="btn-primary py-3 px-6 text-sm"
            >
              <Zap className="w-4 h-4" />
              Check a Transaction Now
            </button>
            <button
              onClick={() => onNavigate('check-bulk')}
              className="btn-secondary py-3 px-6 text-sm"
            >
              <Layers className="w-4 h-4" />
              Bulk Upload File
            </button>
          </div>
        </div>

        {/* Feature highlight stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-10 pt-8 border-t border-slate-200/80">
          <div>
            <span className="block text-2xl font-black text-slate-900">590,000+</span>
            <span className="text-xs text-slate-500 font-medium">Trained Historical Patterns</span>
          </div>
          <div>
            <span className="block text-2xl font-black text-indigo-600">&lt; 250 ms</span>
            <span className="text-xs text-slate-500 font-medium">Instant Analysis Speed</span>
          </div>
          <div>
            <span className="block text-2xl font-black text-emerald-600">91.0%</span>
            <span className="text-xs text-slate-500 font-medium">ROC-AUC Test Accuracy</span>
          </div>
          <div>
            <span className="block text-2xl font-black text-slate-900">100%</span>
            <span className="text-xs text-slate-500 font-medium">Explainable Decisions</span>
          </div>
        </div>
      </section>

      {/* 3 Simple Steps */}
      <section className="space-y-4">
        <div className="text-center max-w-xl mx-auto space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900">How It Works in 3 Simple Steps</h2>
          <p className="text-sm text-slate-500">Zero complex setup. Clear answers whenever a transaction arrives.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="card-clean p-6 space-y-3 relative">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg">
              1
            </div>
            <h3 className="font-bold text-slate-900 text-base">Enter Purchase Details</h3>
            <p className="text-sm text-slate-600">
              Provide basic purchase facts (order amount, card issuer, email, and device). Test immediately with one-click presets.
            </p>
          </div>

          <div className="card-clean p-6 space-y-3 relative">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg">
              2
            </div>
            <h3 className="font-bold text-slate-900 text-base">Instant AI Evaluation</h3>
            <p className="text-sm text-slate-600">
              The model cross-references 457 risk factors, computes the exact probability of a chargeback, and selects the optimal action.
            </p>
          </div>

          <div className="card-clean p-6 space-y-3 relative">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg">
              3
            </div>
            <h3 className="font-bold text-slate-900 text-base">Understand the 'Why'</h3>
            <p className="text-sm text-slate-600">
              Get an Approve, Review, or Block decision accompanied by plain-English sentences revealing what raised or lowered the risk.
            </p>
          </div>
        </div>
      </section>

      {/* Legend: How to read the results */}
      <section className="card-clean p-6 sm:p-8 space-y-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900">How to Read the Results</h2>
          <p className="text-sm text-slate-500 mt-0.5">Every transaction is categorized into one of three clear actions:</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-xl border border-emerald-200 bg-emerald-50/60 space-y-2">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              APPROVE (Low Risk)
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Safe to Process</h3>
            <p className="text-xs text-slate-700 leading-relaxed">
              Fraud probability is negligible (&lt; 5.95%). The order is cleared for immediate processing with zero customer friction.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-amber-200 bg-amber-50/60 space-y-2">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-800 border border-amber-300">
              <Clock className="w-3.5 h-3.5 text-amber-600" />
              MANUAL REVIEW (Medium Risk)
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Human Inspection Advised</h3>
            <p className="text-xs text-slate-700 leading-relaxed">
              Borderline risk signals (5.95% to 75.0%). Routing to your operations queue prevents losing a good buyer while catching fraud.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-rose-200 bg-rose-50/60 space-y-2">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-100 text-rose-800 border border-rose-300">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
              BLOCK (High Risk)
            </div>
            <h3 className="font-bold text-slate-900 text-sm">Likely Fraudulent</h3>
            <p className="text-xs text-slate-700 leading-relaxed">
              Severe fraud probability (&ge; 75.0%). Automatically declined with 90%+ precision to prevent an expensive chargeback.
            </p>
          </div>
        </div>
      </section>

      {/* Available Tools Grid */}
      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Available Risk Tools</h2>
          <p className="text-sm text-slate-500">Select a tool below to begin:</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <div
                key={tool.id}
                onClick={() => onNavigate(tool.id)}
                className="card-clean card-interactive p-6 flex flex-col justify-between cursor-pointer group"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className={`w-10 h-10 rounded-xl ${tool.color} text-white flex items-center justify-center shadow-xs`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
                      {tool.badge}
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                    {tool.title}
                  </h3>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    {tool.description}
                  </p>
                </div>

                <div className="pt-4 mt-4 border-t border-slate-100 flex items-center text-xs font-bold text-indigo-600 group-hover:translate-x-0.5 transition-transform">
                  <span>{tool.action}</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1" />
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
