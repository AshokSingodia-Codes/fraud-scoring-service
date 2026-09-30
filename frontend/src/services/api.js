import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'demo-api-key-12345';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  },
  timeout: 15000,
});

export const checkHealth = async () => {
  const res = await axios.get(`${API_BASE_URL}/health`);
  return res.data;
};

export const scoreTransaction = async (payload, explain = true) => {
  const res = await api.post(`/v1/score?explain=${explain}`, payload);
  return res.data;
};

export const scoreBatchTransactions = async (transactions, explain = false) => {
  const res = await api.post(`/v1/score/batch?explain=${explain}`, { transactions });
  return res.data;
};

export const getModelInfo = async () => {
  const res = await api.get('/v1/model/info');
  return res.data;
};

export const getPredictionById = async (txId) => {
  const res = await api.get(`/v1/predictions/${txId}`);
  return res.data;
};

export default api;
