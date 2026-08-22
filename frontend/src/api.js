// In Docker/production, nginx proxies /api/* to the FastAPI backend, so the
// default is a relative path. In local `npm run dev`, point straight at the
// backend on :8000 unless VITE_API_URL overrides it.
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return res.json();
}

export function fetchHealth() {
  return getJSON("/health");
}

export function fetchMetrics() {
  return getJSON("/metrics");
}

export function fetchNextFlow() {
  return getJSON("/stream/next");
}

export function simulateCategory(category) {
  return getJSON(`/simulate/${category}`);
}
