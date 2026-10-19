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
      <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
        {/* Clean Navbar */}
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenHelp={() => setHelpOpen(true)}
        />

        {/* Content Body */}
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-6">
          {activeTab === 'home' && <HomePage onNavigate={setActiveTab} />}
          {activeTab === 'check-one' && <CheckOnePage onToast={showToast} />}
          {activeTab === 'check-bulk' && <CheckBulkPage onToast={showToast} />}
          {activeTab === 'find-id' && <FindByIdPage onToast={showToast} />}
          {activeTab === 'saved-reports' && <SavedReportsPage onToast={showToast} />}
          {activeTab === 'model-info' && <ModelInfoPage />}
        </main>

        {/* Minimal Footer */}
        <footer className="border-t border-slate-200 bg-white py-4 text-xs text-slate-500">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between">
            <span className="font-medium text-slate-700">FraudShield</span>
            <div className="flex gap-4">
              <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="hover:text-slate-900">
                API Docs
              </a>
              <a href="http://127.0.0.1:8000/metrics" target="_blank" rel="noreferrer" className="hover:text-slate-900">
                Metrics
              </a>
            </div>
          </div>
        </footer>

        {/* Modals */}
        <SettingsModal
          isOpen={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onToast={showToast}
        />
        <HelpDrawer
          isOpen={helpOpen}
          onClose={() => setHelpOpen(false)}
        />
        {toast && (
          <Toast
            message={toast.message}
            type={toast.type}
            onClose={() => setToast(null)}
          />
        )}
      </div>
    </ErrorBoundary>
  );
}
