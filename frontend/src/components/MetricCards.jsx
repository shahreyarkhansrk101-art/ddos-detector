export default function MetricCards({ total, alerts, alertRate, connected }) {
  const cards = [
    { label: "Flows scored", value: total.toLocaleString() },
    { label: "DDoS alerts", value: alerts.toLocaleString() },
    { label: "Alert rate", value: `${alertRate.toFixed(1)}%` },
    { label: "Status", value: connected ? "Live" : "Connecting..." },
  ];

  return (
    <div className="metric-grid">
      {cards.map((c) => (
        <div className="metric-card" key={c.label}>
          <p className="metric-label">{c.label}</p>
          <p className={`metric-value ${c.label === "Status" ? (connected ? "ok" : "pending") : ""}`}>
            {c.value}
          </p>
        </div>
      ))}
    </div>
  );
}
