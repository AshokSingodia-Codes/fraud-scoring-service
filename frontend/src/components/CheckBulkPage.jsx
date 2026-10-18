import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Download,
  CheckCircle2,
  AlertTriangle,
  Search,
  Filter,
  BookmarkPlus,
  Eye,
  X,
  Code,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { scoreBatchTransactions } from '../services/api';
import { formatReasonExplanation, getDecisionBadgeInfo } from '../utils/reasonFormatter';
import { exportReportPdf, exportBatchReportPdf } from '../utils/pdfExport';
import SaveReportModal from './SaveReportModal';

export default function CheckBulkPage({ onToast }) {
  const [file, setFile] = useState(null);
  const [parsedRows, setParsedRows] = useState([]);
  const [validationError, setValidationError] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [results, setResults] = useState(null);
  const [selectedTx, setSelectedTx] = useState(null);
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [devSectionOpen, setDevSectionOpen] = useState(false);

  // Table filters & search
  const [searchQuery, setSearchQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState('all');
  const [riskBandFilter, setRiskBandFilter] = useState('all');

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) parseAndValidateFile(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) parseAndValidateFile(dropped);
  };

  const parseAndValidateFile = async (uploadedFile) => {
    setValidationError(null);
    setResults(null);
    setSelectedTx(null);

    const isCsv = uploadedFile.name.endsWith('.csv');
    const isJson = uploadedFile.name.endsWith('.json');

    if (!isCsv && !isJson) {
      setValidationError("Please select a .csv or .json file.");
      return;
    }

    setFile(uploadedFile);

    try {
      const text = await uploadedFile.text();
      let transactions = [];

      if (isJson) {
        const json = JSON.parse(text);
        transactions = Array.isArray(json) ? json : json.transactions || [];
        const missingField = transactions.find(t => !t.transaction_id || t.TransactionAmt === undefined || t.TransactionAmt === null);
        if (missingField) {
          setValidationError("Each JSON transaction must include 'transaction_id' and 'TransactionAmt'.");
          setParsedRows([]);
          return;
        }
      } else {
        // Simple CSV parser
        const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
        if (lines.length < 2) {
          setValidationError("CSV file must contain a header row and at least 1 transaction row.");
          setParsedRows([]);
          return;
        }

        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const requiredFields = ['transaction_id', 'TransactionAmt'];
        const missing = requiredFields.filter(f => !headers.includes(f));

        if (missing.length > 0) {
          setValidationError(`Missing required column headers: ${missing.join(', ')}. Use our template for guidance.`);
          setParsedRows([]);
          return;
        }

        for (let i = 1; i < lines.length; i++) {
          const vals = lines[i].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
          if (vals.length < headers.length) continue;

          const rowObj = {};
          headers.forEach((h, idx) => {
            let val = vals[idx];
            if (h === 'TransactionAmt' || h === 'card1' || h === 'card2' || h === 'TransactionDT') {
              val = val !== '' && !isNaN(val) ? Number(val) : val;
            }
            rowObj[h] = val;
          });

          // Ensure transaction_id exists and TransactionDT has fallback
          if (!rowObj.transaction_id || String(rowObj.transaction_id).trim() === '') {
            rowObj.transaction_id = `tx_row_${i}`;
          }
          if (!rowObj.TransactionDT || isNaN(rowObj.TransactionDT)) {
            rowObj.TransactionDT = 86400;
          }

          transactions.push(rowObj);
        }
      }

      if (transactions.length === 0) {
        setValidationError("No valid transaction rows found in the file.");
        setParsedRows([]);
        return;
      }

      if (transactions.length > 1000) {
        setValidationError(`File contains ${transactions.length} transactions, exceeding the 1,000 row safety limit. Please split the file into smaller batches.`);
        setParsedRows([]);
        return;
      }

      // Check for zero or negative amount in rows
      const invalidRow = transactions.find(t => isNaN(t.TransactionAmt) || Number(t.TransactionAmt) <= 0);
      if (invalidRow) {
        setValidationError(`Transaction ${invalidRow.transaction_id} contains an invalid or zero/negative amount (${invalidRow.TransactionAmt}). All transaction amounts must be greater than $0.00.`);
        setParsedRows([]);
        return;
      }

      setParsedRows(transactions);
    } catch (err) {
      setValidationError(`Failed to read file: ${err.message}`);
      setParsedRows([]);
    }
  };

  const handleProcess = async () => {
    if (!parsedRows || parsedRows.length === 0) return;
    setProcessing(true);
    setProgressPercent(20);

    try {
      // Step simulated progress
      const timer = setInterval(() => {
        setProgressPercent(prev => (prev < 90 ? prev + 15 : prev));
      }, 300);

      const res = await scoreBatchTransactions(parsedRows, true);
      clearInterval(timer);
      setProgressPercent(100);
      setResults(res);
      if (onToast) onToast(`Successfully evaluated ${res.total_processed} transactions!`);
    } catch (err) {
      setValidationError(err.message || "Failed to process batch transactions.");
    } finally {
      setProcessing(false);
    }
  };

  // Metrics summary
  const summary = React.useMemo(() => {
    if (!results?.results) return null;
    const list = results.results;
    const total = list.length;
    const approve = list.filter(r => r.decision === 'approve').length;
    const review = list.filter(r => r.decision === 'review').length;
    const block = list.filter(r => r.decision === 'block').length;

    return {
      total,
      approve,
      review,
      block,
      approvePct: total ? ((approve / total) * 100).toFixed(1) : 0,
      reviewPct: total ? ((review / total) * 100).toFixed(1) : 0,
      blockPct: total ? ((block / total) * 100).toFixed(1) : 0,
    };
  }, [results]);

  const pieData = summary ? [
    { name: 'Approve (Safe)', value: summary.approve, color: '#16a34a' },
    { name: 'Manual Review', value: summary.review, color: '#d97706' },
    { name: 'Block (Fraud)', value: summary.block, color: '#dc2626' },
  ].filter(d => d.value > 0) : [];

  // Filtered rows
  const filteredRows = React.useMemo(() => {
    if (!results?.results) return [];
    return results.results.filter(row => {
      const matchSearch = searchQuery.trim() === '' ||
        row.transaction_id.toLowerCase().includes(searchQuery.toLowerCase());
      const matchDecision = decisionFilter === 'all' || row.decision === decisionFilter;
      const matchBand = riskBandFilter === 'all' || row.risk_band === riskBandFilter;
      return matchSearch && matchDecision && matchBand;
    });
  }, [results, searchQuery, decisionFilter, riskBandFilter]);

  // Export results CSV
  const handleExportCsv = () => {
    if (!results?.results) return;
    const headers = ["transaction_id", "decision", "fraud_probability", "risk_band", "latency_ms"];
    const lines = results.results.map(r => [
      `"${r.transaction_id}"`,
      `"${r.decision}"`,
      r.fraud_probability,
      `"${r.risk_band}"`,
      r.latency_ms
    ].join(','));

    const csvStr = "data:text/csv;charset=utf-8," + [headers.join(','), ...lines].join('\n');
    const link = document.createElement('a');
    link.href = encodeURI(csvStr);
    link.download = `Batch_Results_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="space-y-8 py-2">
      {/* Title */}
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Check Many Transactions (Bulk Upload)
        </h1>
        <p className="text-sm text-slate-600">
          Upload and screen an entire batch of up to 1,000 credit card transactions at once using a CSV or JSON file.
        </p>
      </div>

      {/* Explainer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you need to provide
          </span>
          <p className="text-xs text-slate-600">
            A spreadsheet in CSV or JSON format containing your transaction rows. Download our sample template below if you need formatting help.
          </p>
        </div>
        <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs space-y-1.5">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
            What you will get back
          </span>
          <p className="text-xs text-slate-600">
            A complete decision breakdown (chart), interactive table with filters, per-row risk explanations, and export options (CSV/PDF).
          </p>
        </div>
      </div>

      {/* Upload Zone & Controls */}
      <div className="card-clean p-6 sm:p-8 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-900">Upload Transaction File</h2>
            <p className="text-xs text-slate-500">Supports standard .csv and .json files (up to 1,000 rows per run)</p>
          </div>
          <a
            href="/sample_bulk.csv"
            download="sample_fraud_transactions.csv"
            className="btn-secondary text-xs py-2 px-3 flex items-center gap-1.5 text-indigo-700 font-bold border-indigo-200 hover:bg-indigo-50"
            title="Download an example CSV template with pre-filled transactions"
          >
            <Download className="w-3.5 h-3.5 text-indigo-600" />
            Download CSV Template
          </a>
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-colors bg-slate-50/50 hover:bg-indigo-50/20 group"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv, .json"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="w-14 h-14 bg-indigo-50 text-indigo-600 group-hover:scale-105 rounded-2xl flex items-center justify-center mx-auto transition-transform">
            <UploadCloud className="w-7 h-7" />
          </div>
          <div className="mt-3 space-y-1">
            <p className="text-sm font-bold text-slate-800">
              {file ? file.name : "Click to choose a file or drag & drop here"}
            </p>
            <p className="text-xs text-slate-500">
              {file ? `${(file.size / 1024).toFixed(1)} KB — Ready to analyze` : "CSV or JSON files with up to 1,000 transactions"}
            </p>
          </div>
        </div>

        {validationError && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-center gap-2.5">
            <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {parsedRows.length > 0 && !results && (
          <div className="p-4 bg-indigo-50 border border-indigo-200 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2 text-indigo-900 font-semibold">
              <CheckCircle2 className="w-4 h-4 text-indigo-600" />
              <span>Loaded <strong>{parsedRows.length}</strong> valid transaction rows. Ready to run AI scoring.</span>
            </div>
            <button
              onClick={handleProcess}
              disabled={processing}
              className="btn-primary py-2 px-5 text-xs font-bold"
            >
              {processing ? `Evaluating (${progressPercent}%)...` : `Run AI Risk Analysis on ${parsedRows.length} Transactions`}
            </button>
          </div>
        )}

        {processing && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-500 font-medium">
              <span>Scoring transactions and calculating explanations...</span>
              <span>{progressPercent}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Batch Results Overview & Table */}
      {results && summary && (
        <div className="space-y-6 animate-in fade-in duration-200">
          {/* Executive Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card-clean p-5 space-y-1">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block">
                Total Evaluated
              </span>
              <span className="text-3xl font-black text-slate-900">{summary.total}</span>
              <span className="text-xs text-slate-400 block">Completed in {results.batch_latency_ms} ms</span>
            </div>

            <div className="card-clean p-5 border-emerald-200 bg-emerald-50/40 space-y-1">
              <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider block">
                Approved (Safe)
              </span>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-emerald-800">{summary.approve}</span>
                <span className="text-xs font-bold text-emerald-600">({summary.approvePct}%)</span>
              </div>
              <span className="text-xs text-emerald-700 block">Cleared without review</span>
            </div>

            <div className="card-clean p-5 border-amber-200 bg-amber-50/40 space-y-1">
              <span className="text-xs font-bold text-amber-700 uppercase tracking-wider block">
                Manual Review
              </span>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-amber-800">{summary.review}</span>
                <span className="text-xs font-bold text-amber-600">({summary.reviewPct}%)</span>
              </div>
              <span className="text-xs text-amber-700 block">Require analyst inspection</span>
            </div>

            <div className="card-clean p-5 border-rose-200 bg-rose-50/40 space-y-1">
              <span className="text-xs font-bold text-rose-700 uppercase tracking-wider block">
                Blocked (Fraud)
              </span>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-rose-800">{summary.block}</span>
                <span className="text-xs font-bold text-rose-600">({summary.blockPct}%)</span>
              </div>
              <span className="text-xs text-rose-700 block">Declined to prevent loss</span>
            </div>
          </div>

          {/* Visual Distribution Chart & Actions Banner */}
          <div className="card-clean p-6 flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-6">
              <div className="w-28 h-28">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      cx="50%"
                      cy="50%"
                      innerRadius={28}
                      outerRadius={48}
                      paddingAngle={3}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(val, name) => [`${val} transactions`, name]}
                      contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '11px' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-1.5 text-xs">
                <div className="font-bold text-slate-800">Batch Decision Breakdown</div>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-600" />
                  <span className="text-slate-600">Approve: <strong>{summary.approve}</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-600" />
                  <span className="text-slate-600">Manual Review: <strong>{summary.review}</strong></span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-600" />
                  <span className="text-slate-600">Blocked: <strong>{summary.block}</strong></span>
                </div>
              </div>
            </div>

            {/* Batch Action Buttons */}
            <div className="flex flex-wrap gap-2 w-full md:w-auto">
              <button
                onClick={handleExportCsv}
                className="btn-secondary text-xs py-2 px-3 flex-1 sm:flex-initial"
                title="Download results as CSV"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
              <button
                onClick={() => exportBatchReportPdf(results.results, summary)}
                className="btn-secondary text-xs py-2 px-3 flex-1 sm:flex-initial"
                title="Download printable PDF summary report"
              >
                <Download className="w-3.5 h-3.5" />
                Download Report (PDF)
              </button>
              <button
                onClick={() => setSaveModalOpen(true)}
                className="btn-primary text-xs py-2 px-4 flex-1 sm:flex-initial"
              >
                <BookmarkPlus className="w-3.5 h-3.5" />
                Save Batch to Reports
              </button>
            </div>
          </div>

          {/* Table Search & Filters */}
          <div className="card-clean p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="relative flex-1 min-w-[220px]">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Filter by Transaction ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input-clean pl-9 text-xs"
                />
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <Filter className="w-3.5 h-3.5" /> Filter Decision:
                </div>
                <select
                  value={decisionFilter}
                  onChange={(e) => setDecisionFilter(e.target.value)}
                  className="input-clean text-xs py-1.5 px-2.5 w-auto"
                >
                  <option value="all">All Decisions</option>
                  <option value="approve">Approve Only</option>
                  <option value="review">Review Only</option>
                  <option value="block">Block Only</option>
                </select>

                <select
                  value={riskBandFilter}
                  onChange={(e) => setRiskBandFilter(e.target.value)}
                  className="input-clean text-xs py-1.5 px-2.5 w-auto"
                >
                  <option value="all">All Risk Bands</option>
                  <option value="Low">Low Risk</option>
                  <option value="Medium">Medium Risk</option>
                  <option value="High">High Risk</option>
                </select>
              </div>
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 font-bold text-slate-700">
                    <th className="py-3 px-4">Transaction ID</th>
                    <th className="py-3 px-4">Decision</th>
                    <th className="py-3 px-4">Fraud Probability</th>
                    <th className="py-3 px-4">Risk Band</th>
                    <th className="py-3 px-4">Speed</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredRows.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-slate-400">
                        No transactions match the selected filter.
                      </td>
                    </tr>
                  ) : (
                    filteredRows.map((row) => {
                      const badge = getDecisionBadgeInfo(row.decision);
                      return (
                        <tr
                          key={row.transaction_id}
                          className="hover:bg-slate-50 transition-colors cursor-pointer"
                          onClick={() => setSelectedTx(row)}
                        >
                          <td className="py-3 px-4 font-mono font-medium text-slate-900">
                            {row.transaction_id}
                          </td>
                          <td className="py-3 px-4">
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${badge.colorClass}`}>
                              {badge.label}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-semibold text-slate-800">
                            {(row.fraud_probability * 100).toFixed(2)}%
                          </td>
                          <td className="py-3 px-4 text-slate-600">
                            {row.risk_band}
                          </td>
                          <td className="py-3 px-4 text-slate-400 font-mono text-[11px]">
                            {row.latency_ms} ms
                          </td>
                          <td className="py-3 px-4 text-right">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedTx(row);
                              }}
                              className="text-indigo-600 hover:text-indigo-800 font-bold inline-flex items-center gap-1"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              Inspect
                            </button>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-slate-400 text-right">
              Showing {filteredRows.length} of {results.results.length} total rows. Click any row to view full SHAP reasons.
            </p>
          </div>
        </div>
      )}

      {/* Row Detail Drawer / Modal */}
      {selectedTx && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-2xs animate-in fade-in">
          <div className="card-clean max-w-lg w-full p-6 shadow-2xl space-y-5 relative max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div>
                <span className="text-[11px] font-mono text-slate-400">TRANSACTION INSPECTOR</span>
                <h3 className="text-lg font-bold text-slate-900">{selectedTx.transaction_id}</h3>
              </div>
              <button
                onClick={() => setSelectedTx(null)}
                className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 rounded-xl border bg-slate-50 border-slate-200 flex justify-between items-center">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Assigned Decision</span>
                <span className="text-lg font-black text-slate-900 uppercase">
                  {selectedTx.decision} ({selectedTx.risk_band} Risk)
                </span>
              </div>
              <div className="text-right">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Fraud Probability</span>
                <span className="text-2xl font-black text-indigo-700">
                  {(selectedTx.fraud_probability * 100).toFixed(2)}%
                </span>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Top Factors Influencing This Score
              </h4>
              <div className="space-y-2">
                {(selectedTx.reasons || []).map((r, i) => {
                  const isUp = r.direction === 'increases_risk' || r.contribution > 0;
                  return (
                    <div
                      key={i}
                      className={`p-3 rounded-lg border text-xs space-y-1 ${
                        isUp ? 'bg-rose-50/50 border-rose-100' : 'bg-emerald-50/50 border-emerald-100'
                      }`}
                    >
                      <div className="flex justify-between font-bold">
                        <span className="text-slate-800">{r.label || r.feature}</span>
                        <span className={isUp ? 'text-rose-700' : 'text-emerald-700'}>
                          {isUp ? '+ Increased Risk' : '- Decreased Risk'}
                        </span>
                      </div>
                      <p className="text-slate-600 text-[11px]">
                        {formatReasonExplanation(r)}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="flex justify-between items-center pt-3 border-t border-slate-200">
              <button
                onClick={() => exportReportPdf({
                  transactionId: selectedTx.transaction_id,
                  decision: selectedTx.decision,
                  fraudProbability: selectedTx.fraud_probability,
                  riskBand: selectedTx.risk_band,
                  latencyMs: selectedTx.latency_ms,
                  reasons: selectedTx.reasons,
                  savedAt: new Date().toISOString(),
                })}
                className="btn-secondary text-xs py-2 px-3"
              >
                <Download className="w-3.5 h-3.5" />
                Download Fraud Check Report (PDF)
              </button>
              <button
                onClick={() => setSelectedTx(null)}
                className="btn-primary text-xs py-2 px-4"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Save Entire Batch Report Modal */}
      {saveModalOpen && results && (
        <SaveReportModal
          isOpen={saveModalOpen}
          onClose={() => setSaveModalOpen(false)}
          reportData={{
            type: 'bulk',
            title: `Bulk Check: ${results.total_processed} Transactions`,
            decision: summary.block > 0 ? 'review' : 'approve',
            riskBand: summary.block > 0 ? 'High / Mixed' : 'Low',
            latencyMs: results.batch_latency_ms,
            bulkResults: results.results,
            note: `${summary.approve} Approved, ${summary.review} Review, ${summary.block} Blocked.`,
          }}
          onSaved={() => {
            if (onToast) onToast("Saved batch run to your Reports Vault.");
          }}
        />
      )}

      {/* Developer Accordion */}
      <div className="card-clean p-4">
        <button
          type="button"
          onClick={() => setDevSectionOpen(!devSectionOpen)}
          className="w-full flex items-center justify-between text-xs font-semibold text-slate-500 hover:text-slate-800"
        >
          <span className="flex items-center gap-2">
            <Code className="w-3.5 h-3.5" />
            For Developers: HTTP Batch Endpoint (<code className="text-indigo-600">POST /v1/score/batch</code>)
          </span>
          {devSectionOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {devSectionOpen && (
          <div className="mt-3 pt-3 border-t border-slate-100 space-y-2 text-xs font-mono text-slate-600 animate-in fade-in">
            <p className="font-bold text-slate-800">Endpoint: POST /v1/score/batch?explain=false</p>
            <p>Header: X-API-Key: demo-api-key-12345</p>
            <p>Body structure: {`{ "transactions": [ { "transaction_id": "tx1", "TransactionAmt": 100, ... } ] }`}</p>
          </div>
        )}
      </div>
    </div>
  );
}
