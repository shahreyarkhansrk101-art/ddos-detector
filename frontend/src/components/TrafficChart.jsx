import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

export default function TrafficChart({ flows }) {
  const data = flows
    .slice()
    .reverse()
    .map((f, i) => ({
      idx: i,
      packet_rate: Math.round(f.packet_rate),
      alert: f.is_ddos ? Math.round(f.packet_rate) : null,
    }));

  return (
    <div className="panel">
      <h2>Packet rate over time</h2>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
          <XAxis dataKey="idx" stroke="#8892a4" tick={false} />
          <YAxis stroke="#8892a4" width={60} />
          <Tooltip
            contentStyle={{ background: "#1a1e27", border: "1px solid #2a2f3a" }}
            labelFormatter={() => ""}
            formatter={(value) => [`${value} pkt/s`, "rate"]}
          />
          <Line
            type="monotone"
            dataKey="packet_rate"
            stroke="#4c8dff"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="alert"
            stroke="#e5484d"
            strokeWidth={0}
            dot={{ r: 3, fill: "#e5484d" }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
