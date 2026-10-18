/**
 * Local report persistence service using browser localStorage.
 * Ensures audit reports survive page reloads and browser sessions.
 * Never stores API keys or credentials.
 */

const STORAGE_KEY_REPORTS = 'fraud_saved_reports_v1';

export function getAllReports() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_REPORTS);
    if (!raw) return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch (err) {
    console.error("Failed to read saved reports from localStorage:", err);
    return [];
  }
}

export function saveReport(reportData) {
  const reports = getAllReports();
  const id = `rep_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  
  const newReport = {
    id,
    savedAt: new Date().toISOString(),
    title: reportData.title || `Transaction ${reportData.transactionId || 'Check'}`,
    note: reportData.note || '',
    type: reportData.type || 'single', // 'single' or 'bulk'
    transactionId: reportData.transactionId || reportData.output?.transaction_id || 'unknown',
    decision: reportData.decision || reportData.output?.decision || 'unknown',
    fraudProbability: reportData.fraudProbability ?? reportData.output?.fraud_probability ?? null,
    riskBand: reportData.riskBand || reportData.output?.risk_band || 'Unknown',
    modelVersion: reportData.modelVersion || reportData.output?.model_version || '1.0.0',
    thresholds: reportData.thresholds || reportData.output?.thresholds || null,
    latencyMs: reportData.latencyMs ?? reportData.output?.latency_ms ?? null,
    input: sanitizeInput(reportData.input),
    output: sanitizeOutput(reportData.output),
    reasons: reportData.reasons || reportData.output?.reasons || [],
    bulkResults: reportData.bulkResults || null,
  };

  reports.unshift(newReport);
  localStorage.setItem(STORAGE_KEY_REPORTS, JSON.stringify(reports));
  return newReport;
}

export function updateReport(id, updates) {
  const reports = getAllReports();
  const idx = reports.findIndex(r => r.id === id);
  if (idx === -1) return null;

  reports[idx] = {
    ...reports[idx],
    title: updates.title !== undefined ? updates.title : reports[idx].title,
    note: updates.note !== undefined ? updates.note : reports[idx].note,
    updatedAt: new Date().toISOString(),
  };

  localStorage.setItem(STORAGE_KEY_REPORTS, JSON.stringify(reports));
  return reports[idx];
}

export function deleteReport(id) {
  const reports = getAllReports();
  const filtered = reports.filter(r => r.id !== id);
  localStorage.setItem(STORAGE_KEY_REPORTS, JSON.stringify(filtered));
  return true;
}

export function exportAllReportsJson() {
  const reports = getAllReports();
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reports, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `fraud_reports_backup_${new Date().toISOString().slice(0, 10)}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

export function importReportsJson(jsonString) {
  try {
    const imported = JSON.parse(jsonString);
    if (!Array.isArray(imported)) {
      throw new Error("Invalid backup file: Expected a JSON array of reports.");
    }
    const existing = getAllReports();
    const existingIds = new Set(existing.map(r => r.id));
    
    let addedCount = 0;
    const merged = [...existing];

    for (const item of imported) {
      if (item && item.id && !existingIds.has(item.id)) {
        merged.push(item);
        existingIds.add(item.id);
        addedCount++;
      } else if (item && !item.id) {
        item.id = `rep_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        merged.push(item);
        addedCount++;
      }
    }

    // Sort by savedAt descending
    merged.sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
    localStorage.setItem(STORAGE_KEY_REPORTS, JSON.stringify(merged));
    return addedCount;
  } catch (err) {
    throw new Error(`Failed to import reports: ${err.message}`);
  }
}

/**
 * Remove any sensitive API credentials from saved payload
 */
function sanitizeInput(input) {
  if (!input) return null;
  const clone = { ...input };
  delete clone.apiKey;
  delete clone.api_key;
  delete clone['X-API-Key'];
  return clone;
}

function sanitizeOutput(output) {
  if (!output) return null;
  const clone = { ...output };
  return clone;
}
