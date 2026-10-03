import { useState } from "react";
import { api } from "../services/api.js";
import { Async, Card, Empty, PageHeader, useApi } from "../components/State.jsx";
import TrendChart from "../components/TrendChart.jsx";
import { LABELS, SENTIMENT_COLORS } from "../theme.js";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function Trend() {
  const [granularity, setGranularity] = useState("month");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [mode, setMode] = useState("count");
  const state = useApi(() => api.trend({ granularity, start, end }), [granularity, start, end]);

  return (
    <>
      <PageHeader title="Sentiment Trend Monitor" subtitle="How sentiment changes over time (uses the dataset's timestamp column)." />
      <Card className="filters">
        <label>Group by
          <select value={granularity} onChange={(e) => setGranularity(e.target.value)}>
            <option value="day">Day</option><option value="week">Week</option><option value="month">Month</option>
          </select>
        </label>
        <label>From <input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
        <label>To <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        <label>Y axis
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="count">Post count</option><option value="percent">Percentage</option>
          </select>
        </label>
        <button className="btn" onClick={() => { setStart(""); setEnd(""); }}>Reset dates</button>
      </Card>
      <Async state={state}>
        {(d) => {
          if (!d.available) {
            return (
              <>
                <Card className="top"><Empty message="Time-based sentiment trend is unavailable because the source dataset does not contain a usable date/time field." /></Card>
                <Card title={d.fallback?.title || "Sentiment by record range"} className="top">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={d.fallback?.data || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="range" tick={{ fontSize: 11 }} /><YAxis /><Tooltip /><Legend />
                      {LABELS.map((k) => <Bar key={k} dataKey={k} stackId="a" fill={SENTIMENT_COLORS[k]} />)}
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </>
            );
          }
          return (
            <Card className="top" title={`Sentiment per ${d.granularity}`} subtitle={d.date_range ? `Data covers ${d.date_range.min} to ${d.date_range.max}` : ""}>
              {d.points.length === 0 ? <Empty message={d.message || "No posts fall inside the selected date range."} /> : <TrendChart points={d.points} mode={mode} />}
            </Card>
          );
        }}
      </Async>
    </>
  );
}
