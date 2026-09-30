import React, { useEffect, useState } from 'react';
import { Cpu, CheckCircle2, Shield, BarChart2, Calendar } from 'lucide-react';
import { getModelInfo } from '../services/api';

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const data = await getModelInfo();
        setInfo(data);
      } catch (err) {
        console.error("Failed to load model info:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchInfo();
  }, []);

  if (loading) {
    return (
      <div className="glass-card rounded-2xl p-8 border border-slate-800 text-center flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mr-2" />
        <span className="text-slate-400 text-sm">Loading Model Specifications...</span>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="glass-card rounded-2xl p-6 border border-slate-800 text-slate-400 text-sm">
        Model Info unavailable. Ensure Backend API is running on port 8000.
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 shadow-2xl space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            Model Specification & Threshold Parameters
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Production system parameters and benchmark validation metrics</p>
        </div>
        <span className="px-3 py-1 bg-purple-500/10 border border-purple-500/30 text-purple-400 text-xs font-mono rounded-full font-bold">
          v{info.model_version}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div className="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
          <span className="text-slate-500 block mb-1 flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5 text-indigo-400" /> Training Date
          </span>
          <span className="text-base font-bold text-slate-200">{info.training_date}</span>
        </div>

        <div className="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
          <span className="text-slate-500 block mb-1 flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-purple-400" /> Feature Count
          </span>
          <span className="text-base font-bold text-slate-200">{info.num_features} Features</span>
        </div>

        <div className="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
          <span className="text-slate-500 block mb-1 flex items-center gap-1">
            <Shield className="w-3.5 h-3.5 text-amber-400" /> Review Threshold
          </span>
          <span className="text-base font-bold text-amber-400">t_review = {info.thresholds?.t_review}</span>
        </div>

        <div className="bg-slate-950/60 border border-slate-800 p-4 rounded-xl">
          <span className="text-slate-500 block mb-1 flex items-center gap-1">
            <BarChart2 className="w-3.5 h-3.5 text-rose-400" /> Block Threshold
          </span>
          <span className="text-base font-bold text-rose-400">t_block = {info.thresholds?.t_block}</span>
        </div>
      </div>

      {/* Headline Validation Metrics */}
      <div className="bg-slate-950/40 border border-slate-800 p-5 rounded-xl space-y-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Expected Test Performance Ranges
        </h3>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
            <span className="text-slate-400 block">ROC-AUC (Time Split)</span>
            <span className="text-sm font-bold text-indigo-400">{info.headline_metrics?.expected_roc_auc}</span>
          </div>
          <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
            <span className="text-slate-400 block">PR-AUC (Time Split)</span>
            <span className="text-sm font-bold text-purple-400">{info.headline_metrics?.expected_pr_auc}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
