import React, { useEffect, useState } from 'react';
import { ShieldAlert, Activity, Layers, FileText, CheckCircle2, XCircle, Cpu } from 'lucide-react';
import { checkHealth } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [isHealthy, setIsHealthy] = useState(null);

  useEffect(() => {
    const verifyHealth = async () => {
      try {
        await checkHealth();
        setIsHealthy(true);
      } catch {
        setIsHealthy(false);
      }
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'single', label: 'Single Scoring', icon: ShieldAlert },
    { id: 'batch', label: 'Batch Processor', icon: Layers },
    { id: 'audit', label: 'Audit Logs', icon: FileText },
    { id: 'info', label: 'Model Spec', icon: Cpu },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <ShieldAlert className="w-6 h-6 text-white" />
          </div>
          <div>
            <span className="text-xl font-extrabold gradient-text tracking-tight">FraudShield AI</span>
            <span className="hidden sm:inline-block ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              IEEE-CIS Engine
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 sm:space-x-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden md:inline">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* API Health Status */}
        <div className="flex items-center space-x-2 bg-slate-950/60 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
          {isHealthy === true ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-400 animate-pulse" />
              <span className="text-slate-300 hidden sm:inline">Backend Online</span>
            </>
          ) : isHealthy === false ? (
            <>
              <XCircle className="w-4 h-4 text-rose-500" />
              <span className="text-slate-300 hidden sm:inline">Backend Offline</span>
            </>
          ) : (
            <>
              <Activity className="w-4 h-4 text-amber-400 animate-spin" />
              <span className="text-slate-300 hidden sm:inline">Checking...</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
