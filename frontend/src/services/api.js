import axios from 'axios';

// Storage key for user-configured API endpoint
const STORAGE_KEY_API_URL = 'fraud_api_base_url';

export function getApiSettings() {
  const envUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const storedUrl = localStorage.getItem(STORAGE_KEY_API_URL);

  const rawUrl = storedUrl && storedUrl.trim() !== '' ? storedUrl.trim() : envUrl;
  const cleanUrl = rawUrl.replace(/\/+$/, '');

  return {
    apiUrl: cleanUrl,
  };
}

export function saveApiSettings(apiUrl) {
  if (apiUrl) localStorage.setItem(STORAGE_KEY_API_URL, apiUrl.trim());
}

export function resetApiSettings() {
  localStorage.removeItem(STORAGE_KEY_API_URL);
}

function createClient() {
  const { apiUrl } = getApiSettings();
  return axios.create({
    baseURL: apiUrl,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 30000,
  });
}

/**
 * Check backend liveness probe
 */
export async function checkHealth() {
  const { apiUrl } = getApiSettings();
  const res = await axios.get(`${apiUrl}/health`, { timeout: 5000 });
  return res.data;
}

/**
 * Check backend readiness probe
 */
export async function checkReady() {
  const { apiUrl } = getApiSettings();
  const res = await axios.get(`${apiUrl}/ready`, { timeout: 5000 });
  return res.data;
}

/**
 * Score a single transaction in real time
 */
export async function scoreTransaction(payload, explain = true) {
  const client = createClient();
  try {
    const res = await client.post(`/v1/score?explain=${explain}`, payload);
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * Score multiple transactions in bulk
 */
export async function scoreBatchTransactions(transactions, explain = false) {
  const client = createClient();
  try {
    const res = await client.post(`/v1/score/batch?explain=${explain}`, { transactions });
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * Query an audit record by transaction ID
 */
export async function getPredictionById(transactionId) {
  const client = createClient();
  try {
    const res = await client.get(`/v1/predictions/${encodeURIComponent(transactionId)}`);
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * List historical predictions with pagination and filters
 */
export async function listPredictions(limit = 50, offset = 0, decision = null, riskBand = null) {
  const client = createClient();
  try {
    const params = { limit, offset };
    if (decision && decision !== 'all') params.decision = decision;
    if (riskBand && riskBand !== 'all') params.risk_band = riskBand;
    const res = await client.get('/v1/predictions', { params });
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * Update a prediction audit record (e.g. notes or decision status)
 */
export async function updatePredictionById(transactionId, updateData) {
  const client = createClient();
  try {
    const res = await client.patch(`/v1/predictions/${encodeURIComponent(transactionId)}`, updateData);
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * Delete a prediction audit record
 */
export async function deletePredictionById(transactionId) {
  const client = createClient();
  try {
    const res = await client.delete(`/v1/predictions/${encodeURIComponent(transactionId)}`);
    return res.data;
  } catch (err) {
    throw parseApiError(err);
  }
}

/**
 * Formats API errors into helpful plain English
 */
function parseApiError(err) {
  if (err.response) {
    const status = err.response.status;
    const detail = err.response.data?.detail;

    if (status === 404) {
      return new Error("No record found with this Transaction ID in the database.");
    }
    if (status === 429) {
      return new Error("Too many requests sent in a short window. Please wait a moment before trying again.");
    }
    if (status === 503) {
      return new Error("The machine learning model is currently loading or offline. Please check back in a minute.");
    }

    if (typeof detail === 'string') {
      return new Error(detail);
    } else if (Array.isArray(detail)) {
      const msg = detail.map(d => `${d.loc ? d.loc.join('.') : 'field'}: ${d.msg}`).join(', ');
      return new Error(`Validation issue: ${msg}`);
    }
    return new Error(`Server returned error ${status}: ${err.response.statusText || 'Unknown error'}`);
  } else if (err.request) {
    return new Error("Cannot reach the fraud service. Check that the backend is running and your API Base URL is correct.");
  }
  return err;
}
