import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Empty } from "./State.jsx";

export default function KeywordChart({ words, color = "#1877f2", height = 340 }) {
  if (!words || !words.length) return <Empty />;
  const data = words.map((w) => ({ word: w.word, posts: w.count }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "var(--muted)" }} />
        <YAxis type="category" dataKey="word" width={90} tick={{ fontSize: 12, fill: "var(--text)" }} />
        <Tooltip formatter={(v) => [v, "posts containing the word"]} />
        <Bar dataKey="posts" fill={color} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
