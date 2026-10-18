import React, { useState } from 'react';
import Navbar from './components/Navbar';
import HomePage from './components/HomePage';
import CheckOnePage from './components/CheckOnePage';
import CheckBulkPage from './components/CheckBulkPage';
import FindByIdPage from './components/FindByIdPage';
import SavedReportsPage from './components/SavedReportsPage';
import ModelInfoPage from './components/ModelInfoPage';
import SettingsModal from './components/SettingsModal';
import HelpDrawer from './components/HelpDrawer';
import Toast from './components/Toast';
import ErrorBoundary from './components/ErrorBoundary';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans selection:bg-indigo-100 selection:text-indigo-800">
        {/* Top Navigation */}
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenHelp={() => setHelpOpen(true)}
        />

        {/* Main Page Content */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          {activeTab === 'home' && <HomePage onNavigate={setActiveTab} />}
          {activeTab === 'check-one' && <CheckOnePage onToast={showToast} />}
          {activeTab === 'check-bulk' && <CheckBulkPage onToast={showToast} />}
          {activeTab === 'find-id' && <FindByIdPage onToast={showToast} />}
          {activeTab === 'saved-reports' && <SavedReportsPage onToast={showToast} />}
          {activeTab === 'model-info' && <ModelInfoPage />}
        </main>

        {/* Global Footer */}
        <footer className="border-t border-slate-200 bg-white py-6 text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-700">FraudShield AI</span>
              <span>•</span>
              <span>Enterprise Fraud Scoring Engine &copy; 2026</span>
            </div>

            <div className="flex items-center gap-4 text-xs font-medium">
              <a
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="hover:text-indigo-600 transition-colors"
              >
                Swagger API Docs
              </a>
              <span>•</span>
              <a
                href="http://127.0.0.1:8000/metrics"
                target="_blank"
                rel="noreferrer"
                className="hover:text-indigo-600 transition-colors"
              >
                Prometheus Metrics
              </a>
              <span>•</span>
              <button
                onClick={() => setHelpOpen(true)}
                className="hover:text-indigo-600 transition-colors cursor-pointer"
              >
                Glossary & Help
              </button>
            </div>
          </div>
        </footer>

        {/* Modals & Overlays */}
        <SettingsModal
          isOpen={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onSaveSuccess={(msg) => showToast(msg)}
        />

        <HelpDrawer
          isOpen={helpOpen}
          onClose={() => setHelpOpen(false)}
        />

        <Toast
          message={toast?.message}
          type={toast?.type}
          onClose={() => setToast(null)}
        />
      </div>
    </ErrorBoundary>
  );
}
