import React, { useEffect, useState } from 'react';
import {
  ShieldCheck,
  Search,
  Layers,
  FileText,
  BarChart2,
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
    { id: 'home', label: 'Overview' },
    { id: 'check-one', label: 'Single Check' },
    { id: 'check-bulk', label: 'Batch' },
    { id: 'find-id', label: 'Audit Log' },
    { id: 'saved-reports', label: 'Saved' },
  ];

  const handleNavClick = (id) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => handleNavClick('home')}>
            <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center shadow-sm">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <span className="font-bold text-slate-900 text-base tracking-tight">FraudShield</span>
          </div>

          {/* Clean Minimalist Nav */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-slate-100 text-slate-900 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Quick Actions */}
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                systemStatus.online ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
              title={systemStatus.online ? 'Service online' : 'Offline'}
            />

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-1.5 text-slate-600 hover:bg-slate-100 rounded-md"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden py-2 border-t border-slate-100 space-y-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNavClick(item.id)}
                className={`w-full text-left px-3 py-2 rounded-md text-xs font-medium ${
                  activeTab === item.id ? 'bg-slate-100 text-slate-900 font-bold' : 'text-slate-600'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </header>
  );
}
