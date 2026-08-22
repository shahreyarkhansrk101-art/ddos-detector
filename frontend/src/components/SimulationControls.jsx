const ATTACK_TYPES = [
  { key: "benign", label: "Benign traffic" },
  { key: "flash_crowd", label: "Flash crowd" },
  { key: "syn_flood", label: "SYN flood" },
  { key: "udp_flood", label: "UDP flood" },
  { key: "http_flood", label: "HTTP flood" },
];

export default function SimulationControls({ autoMode, onToggleAuto, onInject, injecting }) {
  return (
    <div className="panel controls">
      <div className="controls-row">
        <h2>Simulation controls</h2>
        <button
          className={`btn ${autoMode ? "btn-active" : ""}`}
          onClick={onToggleAuto}
        >
          {autoMode ? "Pause auto traffic" : "Resume auto traffic"}
        </button>
      </div>
      <p className="muted small">
        {autoMode
          ? "Random traffic is streaming automatically. Pause it to inject specific flows manually."
          : "Auto traffic paused. Use the buttons below to inject a specific flow on demand."}
      </p>
      <div className="controls-buttons">
        {ATTACK_TYPES.map((t) => (
          <button
            key={t.key}
            className="btn"
            disabled={injecting}
            onClick={() => onInject(t.key)}
          >
            Inject: {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}
