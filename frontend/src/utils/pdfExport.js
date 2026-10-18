import { jsPDF } from 'jspdf';
import { formatReasonExplanation } from './reasonFormatter';

/**
 * Generate a professional printable PDF audit report for a transaction
 */
export function exportReportPdf(report) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'pt',
    format: 'a4',
  });

  const margin = 40;
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 45;

  // Header Title
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42); // slate-900
  doc.text("FraudShield AI — Fraud Check Report", margin, y);
  y += 18;

  // Subtitle
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(100, 116, 139); // slate-500
  doc.text(`Generated on: ${new Date(report.savedAt || Date.now()).toLocaleString()} | Reference ID: ${report.id || 'N/A'}`, margin, y);
  y += 14;

  // Divider line
  doc.setDrawColor(226, 232, 240); // slate-200
  doc.setLineWidth(1);
  doc.line(margin, y, pageWidth - margin, y);
  y += 20;

  // Big Decision Banner
  const decision = (report.decision || 'unknown').toUpperCase();
  let bannerBg = [240, 253, 244]; // emerald-50
  let bannerBorder = [187, 247, 208]; // emerald-200
  let bannerText = [21, 128, 61]; // emerald-700
  let bannerLabel = "DECISION: APPROVED (SAFE TRANSACTION)";

  if (decision === 'REVIEW') {
    bannerBg = [255, 251, 235]; // amber-50
    bannerBorder = [253, 230, 138]; // amber-200
    bannerText = [180, 83, 9]; // amber-700
    bannerLabel = "DECISION: MANUAL REVIEW REQUIRED";
  } else if (decision === 'BLOCK') {
    bannerBg = [254, 242, 242]; // rose-50
    bannerBorder = [254, 202, 202]; // rose-200
    bannerText = [185, 28, 28]; // rose-700
    bannerLabel = "DECISION: BLOCKED (HIGH FRAUD RISK)";
  }

  // Draw Banner Rect
  doc.setFillColor(...bannerBg);
  doc.setDrawColor(...bannerBorder);
  doc.roundedRect(margin, y, pageWidth - (margin * 2), 40, 6, 6, 'FD');

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(...bannerText);
  doc.text(bannerLabel, margin + 14, y + 25);
  y += 55;

  // Key Summary Grid
  doc.setFillColor(248, 250, 252); // slate-50
  doc.setDrawColor(226, 232, 240);
  doc.roundedRect(margin, y, pageWidth - (margin * 2), 52, 6, 6, 'FD');

  const probStr = report.fraudProbability !== null && report.fraudProbability !== undefined
    ? `${(report.fraudProbability * 100).toFixed(2)}%`
    : 'N/A';

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);

  // Column 1
  doc.text("FRAUD PROBABILITY", margin + 16, y + 18);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(15, 23, 42);
  doc.text(probStr, margin + 16, y + 38);

  // Column 2
  const col2X = margin + 140;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text("RISK BAND", col2X, y + 18);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(15, 23, 42);
  doc.text(String(report.riskBand || 'Standard'), col2X, y + 38);

  // Column 3
  const col3X = margin + 270;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text("TRANSACTION AMOUNT", col3X, y + 18);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(15, 23, 42);
  const amt = report.input?.TransactionAmt ? `$${Number(report.input.TransactionAmt).toFixed(2)}` : 'N/A';
  doc.text(amt, col3X, y + 38);

  // Column 4
  const col4X = margin + 410;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text("PROCESSING SPEED", col4X, y + 18);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(14);
  doc.setTextColor(15, 23, 42);
  doc.text(report.latencyMs ? `${report.latencyMs} ms` : 'Instant', col4X, y + 38);

  y += 70;

  // Transaction Input Breakdown
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text("Transaction Parameters", margin, y);
  y += 14;

  const inputFields = [
    ["Transaction ID", report.transactionId || report.input?.transaction_id || 'N/A'],
    ["Product Category", report.input?.ProductCD || 'General (W)'],
    ["Card Issuer Info", `Card ID: ${report.input?.card1 || 'N/A'} (${report.input?.card4 || 'card'}, ${report.input?.card6 || 'debit'})`],
    ["Customer Email", report.input?.P_emaildomain || 'N/A'],
    ["Device & Operating System", `${report.input?.DeviceType || 'Desktop'} — ${report.input?.DeviceInfo || 'Windows'}`],
    ["Review Notes", report.note || 'None entered'],
  ];

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  for (const [key, val] of inputFields) {
    doc.setTextColor(100, 116, 139);
    doc.text(`${key}:`, margin, y);
    doc.setTextColor(30, 41, 59);
    doc.text(String(val), margin + 140, y);
    y += 14;
  }

  y += 10;

  // Key Influencing Reasons Section
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text("Top Factors Explaining This Decision", margin, y);
  y += 14;

  const reasons = report.reasons || [];
  if (reasons.length === 0) {
    doc.setFont("helvetica", "italic");
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.text("No specific elevated risk drivers were identified for this order.", margin, y);
    y += 16;
  } else {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);

    for (let i = 0; i < Math.min(reasons.length, 5); i++) {
      const r = reasons[i];
      const isUp = r.direction === 'increases_risk' || r.contribution > 0;
      const directionTag = isUp ? "[+ RISK INCREASE]" : "[- RISK DECREASE]";
      const plainText = formatReasonExplanation(r);

      doc.setFont("helvetica", "bold");
      if (isUp) {
        doc.setTextColor(185, 28, 28); // red
      } else {
        doc.setTextColor(21, 128, 61); // green
      }
      doc.text(`${i + 1}. ${directionTag} ${r.label || r.feature}:`, margin, y);
      y += 11;

      doc.setFont("helvetica", "normal");
      doc.setTextColor(51, 65, 85);
      const splitText = doc.splitTextToSize(`   ${plainText}`, pageWidth - (margin * 2) - 10);
      doc.text(splitText, margin, y);
      y += (splitText.length * 11) + 4;
    }
  }

  // Footer Disclaimer
  const footerY = doc.internal.pageSize.getHeight() - 30;
  doc.setDrawColor(226, 232, 240);
  doc.line(margin, footerY - 10, pageWidth - margin, footerY - 10);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(148, 163, 184);
  doc.text("FraudShield AI Engine | Autonomous Real-Time Fraud Check Report", margin, footerY);

  // Save PDF
  const filename = `Fraud_Check_Report_${report.transactionId || 'Check'}_${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}

/**
 * Export a report as a formatted CSV file
 */
export function exportReportCsv(report) {
  const headers = ["TransactionID", "Decision", "FraudProbability", "RiskBand", "Amount", "SavedAt", "TopReasons"];
  const reasonsSummary = (report.reasons || []).map(r => `${r.label} (${r.direction})`).join("; ");

  const row = [
    `"${report.transactionId || ''}"`,
    `"${report.decision || ''}"`,
    report.fraudProbability !== null ? report.fraudProbability : "",
    `"${report.riskBand || ''}"`,
    report.input?.TransactionAmt || "",
    `"${report.savedAt || ''}"`,
    `"${reasonsSummary}"`,
  ];

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), row.join(",")].join("\n");
  const link = document.createElement("a");
  link.setAttribute("href", encodeURI(csvContent));
  link.setAttribute("download", `Report_${report.transactionId || 'check'}.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/**
 * Export a report as JSON
 */
export function exportReportJson(report) {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
  const link = document.createElement('a');
  link.setAttribute("href", dataStr);
  link.setAttribute("download", `Report_${report.transactionId || 'check'}.json`);
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/**
 * Generate a PDF executive summary for batch transaction screening
 */
export function exportBatchReportPdf(results, summary) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'pt',
    format: 'a4',
  });

  const margin = 40;
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 45;

  // Header Title
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42);
  doc.text("FraudShield AI — Batch Fraud Check Report", margin, y);
  y += 18;

  // Subtitle
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(100, 116, 139);
  doc.text(`Generated on: ${new Date().toLocaleString()} | Total Transactions: ${summary.total}`, margin, y);
  y += 14;

  // Divider line
  doc.setDrawColor(226, 232, 240);
  doc.line(margin, y, pageWidth - margin, y);
  y += 20;

  // Summary Metrics Grid
  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(226, 232, 240);
  doc.roundedRect(margin, y, pageWidth - (margin * 2), 65, 6, 6, 'FD');

  const colWidth = (pageWidth - (margin * 2)) / 4;

  // Total
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text("TOTAL SCREENED", margin + 12, y + 20);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(15, 23, 42);
  doc.text(String(summary.total), margin + 12, y + 45);

  // Approved
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(21, 128, 61);
  doc.text("APPROVED (SAFE)", margin + colWidth + 12, y + 20);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text(`${summary.approve} (${summary.approvePct}%)`, margin + colWidth + 12, y + 45);

  // Review
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(180, 83, 9);
  doc.text("MANUAL REVIEW", margin + (colWidth * 2) + 12, y + 20);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text(`${summary.review} (${summary.reviewPct}%)`, margin + (colWidth * 2) + 12, y + 45);

  // Blocked
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(185, 28, 28);
  doc.text("BLOCKED (HIGH RISK)", margin + (colWidth * 3) + 12, y + 20);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text(`${summary.block} (${summary.blockPct}%)`, margin + (colWidth * 3) + 12, y + 45);

  y += 85;

  // Sample High Risk Transactions Table
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(15, 23, 42);
  doc.text("High-Priority / Flagged Transactions Breakdown", margin, y);
  y += 14;

  const flagged = (results.results || []).filter(r => r.decision === 'block' || r.decision === 'review');
  const displayList = flagged.length > 0 ? flagged.slice(0, 18) : (results.results || []).slice(0, 18);

  // Table header
  doc.setFillColor(241, 245, 249);
  doc.rect(margin, y, pageWidth - (margin * 2), 20, 'F');
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(71, 85, 105);
  doc.text("TRANSACTION ID", margin + 8, y + 14);
  doc.text("DECISION", margin + 180, y + 14);
  doc.text("FRAUD PROBABILITY", margin + 280, y + 14);
  doc.text("RISK BAND", margin + 400, y + 14);
  y += 22;

  // Table rows
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);

  for (const row of displayList) {
    if (y > doc.internal.pageSize.getHeight() - 50) {
      doc.addPage();
      y = 45;
    }

    doc.setTextColor(30, 41, 59);
    doc.text(String(row.transaction_id || 'N/A'), margin + 8, y + 10);

    const isBlock = row.decision === 'block';
    const isReview = row.decision === 'review';
    if (isBlock) doc.setTextColor(185, 28, 28);
    else if (isReview) doc.setTextColor(180, 83, 9);
    else doc.setTextColor(21, 128, 61);

    doc.text(String(row.decision || '').toUpperCase(), margin + 180, y + 10);

    doc.setTextColor(15, 23, 42);
    doc.text(`${(row.fraud_probability * 100).toFixed(2)}%`, margin + 280, y + 10);
    doc.text(String(row.risk_band || 'Standard'), margin + 400, y + 10);

    doc.setDrawColor(241, 245, 249);
    doc.line(margin, y + 14, pageWidth - margin, y + 14);
    y += 16;
  }

  // Footer
  const footerY = doc.internal.pageSize.getHeight() - 25;
  doc.setDrawColor(226, 232, 240);
  doc.line(margin, footerY - 8, pageWidth - margin, footerY - 8);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(148, 163, 184);
  doc.text(`FraudShield AI Batch Report | ${summary.total} Evaluated | Processing Latency: ${results.batch_latency_ms} ms`, margin, footerY);

  const filename = `Batch_Fraud_Check_Report_${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
}

