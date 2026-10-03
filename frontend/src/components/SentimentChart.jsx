import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { Empty } from "./State.jsx";
import { LABELS, SENTIMENT_COLORS, fmt } from "../theme.js";

export default function SentimentChart({ counts, height = 260 }) {
  const data = LABELS.map((k) => ({ name: k, value: counts?.[k] || 0 })).filter((d) => d.value > 0);
  if (!data.length) return <Empty />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="85%" paddingAngle={2} stroke="none">
          {data.map((d) => <Cell key={d.name} fill={SENTIMENT_COLORS[d.name]} />)}
        </Pie>
        <Tooltip formatter={(v, n) => [fmt(v), n]} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
