import { useRef, useState } from "react";
import { api } from "../services/api.js";
import { Card, ErrorBox } from "./State.jsx";
import SentimentChart from "./SentimentChart.jsx";
import MetricCard from "./MetricCard.jsx";
import { fmt, pct } from "../theme.js";

function toCsv(columns, rows) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.join(","), ...rows.map((r) => columns.map((c) => esc(r[c])).join(","))].join("\r\n");
}

export default function BatchAnalyzer() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [res, setRes] = useState(null);

  async function run() {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setBusy(true);
    setError(null);
    setRes(null);
    try {
      setRes(await api.batch(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function download() {
    const blob = new Blob(["\ufeff" + toCsv(res.columns, res.rows)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentiment_results_${(file?.name || "batch").replace(/\.csv$/i, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const preview = res ? res.rows.slice(0, 50) : [];
  const show = res ? ["predicted_sentiment", "confidence", res.detected_text_column] : [];

  return (
    <div>
      <Card title="Upload a CSV of comments" subtitle="The text column is detected automatically. Limit: 5 MB / 5,000 rows.">
        <div className="row gap wrap">
          <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={(e) => { setFile(e.target.files?.[0] || null); setError(null); }} />
          <button className="btn primary" onClick={run} disabled={busy}>{busy ? "Analyzing..." : "Analyze file"}</button>
          {res && <button className="btn" onClick={download}>Download results (CSV)</button>}
        </div>
        {error && <ErrorBox message={error} />}
      </Card>

      {res && (
        <>
          <div className="grid four top">
            <MetricCard label="Rows analyzed" value={fmt(res.analyzed_rows)} sub={`${fmt(res.skipped_rows)} skipped (empty)`} />
            <MetricCard label="Text column" value={res.detected_text_column} sub="auto-detected" />
            <MetricCard label="Mean confidence" value={pct((res.summary.mean_confidence || 0) * 100)} />
            <MetricCard label="Largest group" value={Object.entries(res.summary.counts).sort((a, b) => b[1] - a[1])[0][0]} />
          </div>
          {res.truncated && <p className="warn">Only the first 5,000 rows were analyzed.</p>}
          <div className="grid two top">
            <Card title="Predicted sentiment"><SentimentChart counts={res.summary.counts} /></Card>
            <Card title={`Preview (first ${preview.length} rows)`} className="scroll">
              <table className="table">
                <thead><tr><th>Comment</th><th>Sentiment</th><th>Confidence</th></tr></thead>
                <tbody>
                  {preview.map((r, i) => (
                    <tr key={i}>
                      <td>{r[res.detected_text_column]}</td>
                      <td><span className={`tag ${r.predicted_sentiment}`}>{r.predicted_sentiment || "-"}</span></td>
                      <td>{r.confidence === null ? "-" : pct(r.confidence * 100, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
