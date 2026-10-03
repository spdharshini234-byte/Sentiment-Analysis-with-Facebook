import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Empty } from "./State.jsx";
import { LABELS, SENTIMENT_COLORS } from "../theme.js";

export default function TrendChart({ points, mode = "count", height = 320 }) {
  if (!points || !points.length) return <Empty />;
  const suffix = mode === "percent" ? "_pct" : "";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={points} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
        <XAxis dataKey="period" tick={{ fontSize: 12, fill: "var(--muted)" }} minTickGap={24} />
        <YAxis tick={{ fontSize: 12, fill: "var(--muted)" }} unit={mode === "percent" ? "%" : ""} />
        <Tooltip />
        <Legend />
        {LABELS.map((k) => (
          <Line key={k} type="monotone" dataKey={k + suffix} name={k} stroke={SENTIMENT_COLORS[k]} strokeWidth={2} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
