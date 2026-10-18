import React, { useEffect, useState } from 'react';
import {
  ShieldCheck,
  Search,
  Layers,
  FileText,
  Info,
  Home,
  Settings,
  HelpCircle,
  Menu,
  X
} from 'lucide-react';
import { checkReady } from '../services/api';

export default function Navbar({ activeTab, setActiveTab, onOpenSettings, onOpenHelp }) {
  const [systemStatus, setSystemStatus] = useState({ online: false, checking: true, version: '' });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let mounted = true;

    const pollStatus = async () => {
      try {
        const data = await checkReady();
        if (mounted) {
          setSystemStatus({ online: true, checking: false, version: data.model_version || '1.0.0' });
        }
      } catch {
        if (mounted) {
          setSystemStatus({ online: false, checking: false, version: '' });
        }
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { id: 'home', label: 'Home', icon: Home, badge: null },
    { id: 'check-one', label: 'Check One Transaction', icon: Search, badge: 'Instant' },
    { id: 'check-bulk', label: 'Check Many (Bulk)', icon: Layers, badge: 'CSV/JSON' },
    { id: 'find-id', label: 'Find by ID', icon: FileText, badge: null },
    { id: 'saved-reports', label: 'Saved Reports', icon: FileText, badge: null },
    { id: 'model-info', label: 'About the Model', icon: Info, badge: null },
  ];

  const handleNavClick = (id) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-xs border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => handleNavClick('home')}>
            <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-xs">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-slate-900 tracking-tight text-lg">FraudShield AI</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Risk Engine
                </span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">Real-Time Credit Card Fraud Protection</p>
            </div>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700 font-semibold shadow-2xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-600' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold ${
                      isActive ? 'bg-indigo-200/60 text-indigo-800' : 'bg-slate-100 text-slate-500'
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right Header Actions: Status & Quick Controls */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* System Status Indicator */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                systemStatus.checking
                  ? 'bg-slate-50 text-slate-500 border-slate-200'
                  : systemStatus.online
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-rose-50 text-rose-700 border-rose-200'
              }`}
              title={systemStatus.online ? `Connected to Fraud Model v${systemStatus.version}` : 'Cannot reach backend API server'}
            >
              <span className={`w-2 h-2 rounded-full ${
                systemStatus.checking
                  ? 'bg-slate-400 animate-pulse'
                  : systemStatus.online
                  ? 'bg-emerald-500 ring-2 ring-emerald-200'
                  : 'bg-rose-500 ring-2 ring-rose-200'
              }`} />
              <span className="hidden sm:inline">
                {systemStatus.checking ? 'Connecting...' : systemStatus.online ? 'Online' : 'Offline'}
              </span>
            </div>

            {/* Help Drawer Trigger */}
            <button
              onClick={onOpenHelp}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Help & Glossary"
              aria-label="Open help guide"
            >
              <HelpCircle className="w-5 h-5" />
            </button>

            {/* Settings Trigger */}
            <button
              onClick={onOpenSettings}
              className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="API Connection Settings"
              aria-label="Open settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 text-slate-600 hover:bg-slate-100 rounded-lg"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="lg:hidden py-3 border-t border-slate-200 space-y-1 animate-in slide-in-from-top-2 duration-150">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700 font-semibold'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-600' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </header>
  );
}
