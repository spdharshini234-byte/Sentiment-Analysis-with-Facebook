// All backend calls go through this file. The base URL is configurable:
//   VITE_API_URL=https://my-backend.onrender.com   (separate hosting)
//   empty                                          (same origin: FastAPI serves the UI, or the Vite dev proxy)
const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the backend. Make sure the server is running and VITE_API_URL is correct.");
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON response */
  }
  if (!res.ok) {
    throw new Error((body && (body.error || body.detail)) || `Request failed (${res.status}).`);
  }
  return body;
}

const qs = (params = {}) => {
  const p = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return p.length ? "?" + new URLSearchParams(p).toString() : "";
};

export const api = {
  health: () => request("/api/health"),
  dashboard: () => request("/api/dashboard"),
  distribution: () => request("/api/sentiment-distribution"),
  trend: (params) => request("/api/sentiment-trend" + qs(params)),
  commonWords: (params) => request("/api/common-words" + qs(params)),
  keywordSentiment: (params) => request("/api/keyword-sentiment" + qs(params)),
  negativeFeedback: (params) => request("/api/negative-feedback" + qs(params)),
  positiveFeedback: (params) => request("/api/positive-feedback" + qs(params)),
  modelPerformance: () => request("/api/model-performance"),
  datasetInfo: () => request("/api/dataset-info"),
  analyze: (text) =>
    request("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }),
  batch: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return request("/api/batch-predict", { method: "POST", body: fd });
  },
};
