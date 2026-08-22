import { useCallback, useEffect, useState } from "react";
import { fetchHealth, fetchMetrics, fetchNextFlow, simulateCategory } from "./api.js";
import MetricCards from "./components/MetricCards.jsx";
import AlertFeed from "./components/AlertFeed.jsx";
import TrafficChart from "./components/TrafficChart.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import SimulationControls from "./components/SimulationControls.jsx";

const MAX_FEED_LENGTH = 40;
const POLL_INTERVAL_MS = 1200;

export default function App() {
  const [flows, setFlows] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [connected, setConnected] = useState(false);
  const [totals, setTotals] = useState({ total: 0, alerts: 0 });
  const [error, setError] = useState(null);
  const [autoMode, setAutoMode] = useState(true);
  const [injecting, setInjecting] = useState(false);

  const addFlow = useCallback((flow) => {
    setFlows((prev) => [flow, ...prev].slice(0, MAX_FEED_LENGTH));
    setTotals((prev) => ({
      total: prev.total + 1,
      alerts: prev.alerts + (flow.is_ddos ? 1 : 0),
    }));
  }, []);

  // Initial metrics + health check, once.
  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => setMetrics(null));
    fetchHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  // Automatic random-traffic polling — only runs while autoMode is on.
  // Re-subscribes whenever autoMode flips, so pausing genuinely stops
  // the requests (not just the UI updates).
  useEffect(() => {
    if (!autoMode) return undefined;

    let cancelled = false;

    async function poll() {
      try {
        const flow = await fetchNextFlow();
        if (cancelled) return;
        setConnected(true);
        setError(null);
        addFlow(flow);
      } catch (e) {
        if (cancelled) return;
        setConnected(false);
        setError(e.message);
      }
    }

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [autoMode, addFlow]);

  async function handleInject(category) {
    setInjecting(true);
    try {
      const flow = await simulateCategory(category);
      setConnected(true);
      setError(null);
      addFlow(flow);
    } catch (e) {
      setConnected(false);
      setError(e.message);
    } finally {
      setInjecting(false);
    }
  }

  const alertRate = totals.total > 0 ? (totals.alerts / totals.total) * 100 : 0;

  return (
    <div className="app">
      <header>
        <h1>🚀 CI/CD Deployment Test</h1>
         
	<p className="muted">Live network flow monitoring — ML-scored in real time</p>
      </header>

      {error && (
        <div className="banner-error">
          Can't reach the API ({error}). Is the backend running?
        </div>
      )}

      <MetricCards
        total={totals.total}
        alerts={totals.alerts}
        alertRate={alertRate}
        connected={connected}
      />

      <SimulationControls
        autoMode={autoMode}
        onToggleAuto={() => setAutoMode((prev) => !prev)}
        onInject={handleInject}
        injecting={injecting}
      />

      <div className="grid-2">
        <TrafficChart flows={flows} />
        <ModelComparison metrics={metrics} />
      </div>

      <AlertFeed flows={flows} />
    </div>
  );
}
