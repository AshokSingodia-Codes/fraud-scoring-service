import { X, HelpCircle, BookOpen, AlertCircle, CheckCircle } from 'lucide-react';

export default function HelpDrawer({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-slate-900/30 backdrop-blur-2xs animate-in fade-in duration-150">
      <div className="absolute inset-y-0 right-0 max-w-md w-full bg-white shadow-2xl flex flex-col border-l border-slate-200">
        {/* Header */}
        <div className="p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2 text-indigo-700">
            <HelpCircle className="w-5 h-5" />
            <h3 className="text-base font-bold text-slate-900">FraudShield Help & Guide</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-200 transition-colors"
            aria-label="Close help"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm text-slate-600">
          {/* Quick Start */}
          <div className="space-y-2">
            <h4 className="font-bold text-slate-900 flex items-center gap-1.5 text-xs uppercase tracking-wider text-indigo-600">
              <CheckCircle className="w-4 h-4" />
              How the System Works in 3 Steps
            </h4>
            <ol className="list-decimal list-inside space-y-1.5 text-xs text-slate-600 pl-1">
              <li><strong className="text-slate-800">Enter purchase info:</strong> Provide the amount, card number, email, and device.</li>
              <li><strong className="text-slate-800">Instant AI evaluation:</strong> The LightGBM model compares features against 590,000 transaction patterns in sub-seconds.</li>
              <li><strong className="text-slate-800">Clear decision & why:</strong> Get an Approve, Review, or Block recommendation with plain-English reasons.</li>
            </ol>
          </div>

          {/* Glossary */}
          <div className="space-y-3 pt-3 border-t border-slate-200">
            <h4 className="font-bold text-slate-900 flex items-center gap-1.5 text-xs uppercase tracking-wider text-indigo-600">
              <BookOpen className="w-4 h-4" />
              Plain-English Glossary
            </h4>

            <div className="space-y-3">
              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="font-semibold text-slate-900 block text-xs">Fraud Probability</span>
                <p className="text-xs text-slate-600 mt-0.5">
                  A calibrated percentage from 0% to 100% predicting the likelihood that this card charge will result in an unauthorized dispute or chargeback.
                </p>
              </div>

              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="font-semibold text-slate-900 block text-xs">Risk Bands (Low, Medium, High)</span>
                <p className="text-xs text-slate-600 mt-0.5">
                  Categories that group transactions for operational triage:
                  <span className="block mt-1"><strong>Low:</strong> Almost certainly legitimate. Clear automatically.</span>
                  <span className="block"><strong>Medium:</strong> Borderline signals. Send to an analyst queue.</span>
                  <span className="block"><strong>High:</strong> Highly suspicious indicators. Decline immediately.</span>
                </p>
              </div>

              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="font-semibold text-slate-900 block text-xs">SHAP Reason Codes</span>
                <p className="text-xs text-slate-600 mt-0.5">
                  An explainability algorithm that measures how much each specific factor (e.g. card frequency, mismatched email) pushed the risk probability upward or downward.
                </p>
              </div>

              <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                <span className="font-semibold text-slate-900 block text-xs">Decision Thresholds</span>
                <p className="text-xs text-slate-600 mt-0.5">
                  The mathematical cutoffs derived from financial cost matrices ($10 manual review cost vs lost transaction amount) that determine when to switch from Approve to Review (5.95%) and Review to Block (75.00%).
                </p>
              </div>
            </div>
          </div>

          {/* Frequently Asked Questions */}
          <div className="space-y-3 pt-3 border-t border-slate-200">
            <h4 className="font-bold text-slate-900 flex items-center gap-1.5 text-xs uppercase tracking-wider text-indigo-600">
              <AlertCircle className="w-4 h-4" />
              Frequently Asked Questions
            </h4>

            <div className="space-y-2 text-xs">
              <details className="group border border-slate-200 rounded-lg p-2.5 open:bg-slate-50">
                <summary className="font-semibold text-slate-800 cursor-pointer list-none flex justify-between items-center">
                  Why does a $25 purchase get blocked?
                  <span className="text-slate-400 group-open:rotate-180 transition-transform">▼</span>
                </summary>
                <p className="text-slate-600 mt-2">
                  Fraudsters frequently test stolen card lists with low-dollar automated scripts. If the card number is brand new, email domains conflict, or browser identity is missing, the AI may flag high testing risk regardless of amount.
                </p>
              </details>

              <details className="group border border-slate-200 rounded-lg p-2.5 open:bg-slate-50">
                <summary className="font-semibold text-slate-800 cursor-pointer list-none flex justify-between items-center">
                  Where are saved reports stored?
                  <span className="text-slate-400 group-open:rotate-180 transition-transform">▼</span>
                </summary>
                <p className="text-slate-600 mt-2">
                  All saved reports are stored securely inside your browser's private storage (<code className="text-indigo-600">localStorage</code>). They stay completely private, never leave your device, and can be backed up as a JSON file anytime.
                </p>
              </details>

              <details className="group border border-slate-200 rounded-lg p-2.5 open:bg-slate-50">
                <summary className="font-semibold text-slate-800 cursor-pointer list-none flex justify-between items-center">
                  Can I export a PDF Fraud Check Report?
                  <span className="text-slate-400 group-open:rotate-180 transition-transform">▼</span>
                </summary>
                <p className="text-slate-600 mt-2">
                  Yes! In both "Check Many" and "Saved Reports", click the "Export PDF" button to download a formatted, printable one-page Fraud Check Report.
                </p>
              </details>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 text-center text-xs text-slate-400">
          FraudShield AI v1.0.0 • Help Center
        </div>
      </div>
    </div>
  );
}
