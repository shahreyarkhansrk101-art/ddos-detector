import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function ModelComparison({ metrics }) {
  if (!metrics) return null;

  const modelNames = Object.keys(metrics).filter((k) => k !== "best_model");
  const data = modelNames.map((name) => ({
    name: name.replace(/_/g, " "),
    accuracy: metrics[name].accuracy,
    precision: metrics[name].precision,
    recall: metrics[name].recall,
    f1: metrics[name].f1_score,
    roc_auc: metrics[name].roc_auc,
  }));

  return (
    <div className="panel">
      <h2>
        Model comparison{" "}
        <span className="muted small">
          (best: {metrics.best_model?.replace(/_/g, " ")})
        </span>
      </h2>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
          <XAxis dataKey="name" stroke="#8892a4" fontSize={12} />
          <YAxis domain={[0, 1]} stroke="#8892a4" width={40} />
          <Tooltip contentStyle={{ background: "#1a1e27", border: "1px solid #2a2f3a" }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="accuracy" fill="#4c8dff" />
          <Bar dataKey="precision" fill="#7c5cff" />
          <Bar dataKey="recall" fill="#22c3aa" />
          <Bar dataKey="f1" fill="#f2b84b" />
          <Bar dataKey="roc_auc" fill="#e5484d" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
