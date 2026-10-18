import axios from 'axios';

// Storage keys for user-configured API endpoints and credentials
const STORAGE_KEY_API_URL = 'fraud_api_base_url';
const STORAGE_KEY_API_KEY = 'fraud_api_key';

export function getApiSettings() {
  const envUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const envKey = import.meta.env.VITE_API_KEY || 'demo-api-key-12345';

  const storedUrl = localStorage.getItem(STORAGE_KEY_API_URL);
  const storedKey = localStorage.getItem(STORAGE_KEY_API_KEY);

  return {
    apiUrl: storedUrl && storedUrl.trim() !== '' ? storedUrl.trim() : envUrl,
    apiKey: storedKey && storedKey.trim() !== '' ? storedKey.trim() : envKey,
  };
}

export function saveApiSettings(apiUrl, apiKey) {
  if (apiUrl) localStorage.setItem(STORAGE_KEY_API_URL, apiUrl.trim());
  if (apiKey) localStorage.setItem(STORAGE_KEY_API_KEY, apiKey.trim());
}

export function resetApiSettings() {
  localStorage.removeItem(STORAGE_KEY_API_URL);
  localStorage.removeItem(STORAGE_KEY_API_KEY);
}

function createClient() {
  const { apiUrl, apiKey } = getApiSettings();
  return axios.create({
    baseURL: apiUrl,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
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
 * Fetch model specifications and operational thresholds
 */
export async function getModelInfo() {
  const client = createClient();
  try {
    const res = await client.get('/v1/model/info');
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

    if (status === 401) {
      return new Error("Access key missing. Please open 'Settings' and enter your API Key.");
    }
    if (status === 403) {
      return new Error("Invalid API key. Please check your API key in 'Settings' and try again.");
    }
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
