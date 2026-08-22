function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function AlertFeed({ flows }) {
  return (
    <div className="panel">
      <h2>Live traffic feed</h2>
      <div className="feed">
        {flows.length === 0 && <p className="muted">Waiting for traffic...</p>}
        {flows.map((f, i) => (
          <div key={i} className={`feed-row ${f.is_ddos ? "alert" : ""}`}>
            <span className="feed-time">{formatTime(f.timestamp)}</span>
            <span className="feed-proto">{f.protocol}</span>
            <span className="feed-rate">{f.packet_rate.toLocaleString()} pkt/s</span>
            <span className="feed-ips">{f.unique_src_ips} src IPs</span>
            <span className="feed-confidence">{(f.confidence * 100).toFixed(0)}%</span>
            <span className={`feed-status ${f.is_ddos ? "alert" : "ok"}`}>
              {f.is_ddos ? "DDoS" : "benign"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
