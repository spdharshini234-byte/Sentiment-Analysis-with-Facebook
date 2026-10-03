import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api.js";
import { Async, Card, Empty, PageHeader, useApi } from "../components/State.jsx";
import MetricCard from "../components/MetricCard.jsx";
import SentimentChart from "../components/SentimentChart.jsx";
import { LABELS, SENTIMENT_COLORS, fmt, pct } from "../theme.js";

export default function Pulse() {
  const state = useApi(() => api.distribution());
  return (
    <>
      <PageHeader title="Sentiment Pulse" subtitle="Counts and shares of each sentiment, computed from the dataset." />
      <Async state={state}>
        {(d) => {
          const cat = (d.by_category || []).map((r) => ({ category: r.category.replace(/_/g, " "), ...Object.fromEntries(LABELS.map((k) => [k, r[`${k}_pct`]])) }));
          return (
            <>
              <div className="grid four">
                <MetricCard label="Total comments" value={fmt(d.total)} sub={`overall: ${d.overall_sentiment}`} />
                {LABELS.map((k) => (
                  <MetricCard key={k} tone={k} label={k.toUpperCase()} value={fmt(d.counts[k])} sub={pct(d.percentages[k])} progress={d.percentages[k]} />
                ))}
              </div>
              <div className="grid two top">
                <Card title="Distribution"><SentimentChart counts={d.counts} height={300} /></Card>
                <Card title="Model confidence" subtitle={`${d.confidence.scope}, ${fmt(d.confidence.total)} posts`}>
                  {["high", "medium", "low"].map((k) => (
                    <div key={k} className="prob">
                      <span className="prob-name">{k}</span>
                      <div className="bar"><div className="bar-fill" style={{ width: `${d.confidence.percentages[k]}%` }} /></div>
                      <span className="prob-val">{fmt(d.confidence.counts[k])} ({pct(d.confidence.percentages[k])})</span>
                    </div>
                  ))}
                  <p className="muted small top">Thresholds - high: {d.confidence.thresholds.high}, medium: {d.confidence.thresholds.medium}, low: {d.confidence.thresholds.low}</p>
                </Card>
              </div>
              <Card title="Sentiment mix by post category" subtitle="Share of posts in each category" className="top">
                {cat.length === 0 ? <Empty /> : (
                  <ResponsiveContainer width="100%" height={Math.max(260, cat.length * 38)}>
                    <BarChart data={cat} layout="vertical" stackOffset="expand" margin={{ left: 24 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" horizontal={false} />
                      <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 12, fill: "var(--muted)" }} />
                      <YAxis type="category" dataKey="category" width={120} tick={{ fontSize: 12, fill: "var(--text)" }} />
                      <Tooltip formatter={(v) => `${Number(v).toFixed(1)}%`} />
                      <Legend />
                      {LABELS.map((k) => <Bar key={k} dataKey={k} stackId="a" fill={SENTIMENT_COLORS[k]} />)}
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </Card>
              <Card title="Dataset labels vs model predictions" subtitle="Model predicted counts over all posts" className="top">
                <table className="table">
                  <thead><tr><th>Sentiment</th><th>Dataset label</th><th>Model prediction</th></tr></thead>
                  <tbody>
                    {LABELS.map((k) => (
                      <tr key={k}><td><span className={`tag ${k}`}>{k}</span></td><td>{fmt(d.counts[k])}</td><td>{fmt(d.model_predicted.counts[k])}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          );
        }}
      </Async>
    </>
  );
}
