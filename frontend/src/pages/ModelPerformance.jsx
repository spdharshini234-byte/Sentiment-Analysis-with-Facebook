import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api.js";
import { Async, Card, PageHeader, useApi } from "../components/State.jsx";
import MetricCard from "../components/MetricCard.jsx";
import { LABELS, f3 } from "../theme.js";

function ConfusionMatrix({ matrix, labels }) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <table className="cm">
      <thead><tr><th></th>{labels.map((l) => <th key={l}>pred. {l}</th>)}</tr></thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th>actual {labels[i]}</th>
            {row.map((v, j) => <td key={j} style={{ background: `rgba(24,119,242,${0.1 + 0.8 * (v / max)})`, color: v / max > 0.5 ? "#fff" : "inherit" }}>{v}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelPerformance() {
  const state = useApi(() => api.modelPerformance());
  return (
    <>
      <PageHeader title="Model Performance Center" subtitle="Four models compared on the same train / validation / test split." />
      <Async state={state}>
        {(d) => {
          const m = d.final_test_metrics;
          const chart = d.models.map((r) => ({ model: r.model.replace(" (majority class)", ""), Validation: +r.validation.f1_macro.toFixed(3), Test: +r.test.f1_macro.toFixed(3) }));
          return (
            <>
              <div className="grid four">
                <MetricCard label="Selected model" value={d.selected_model} sub={d.selection_rule} />
                <MetricCard label="Test accuracy" value={f3(m.accuracy)} sub={`${m.n} unseen posts`} />
                <MetricCard label="Test macro-F1" value={f3(m.f1_macro)} sub={`weighted F1 ${f3(m.f1_weighted)}`} />
                <MetricCard label="Baseline macro-F1" value={f3(d.baseline.f1_macro)} sub={`lift +${f3(d.lift_over_baseline_f1_macro)}`} />
              </div>
              <Card title="Model comparison" className="top">
                <div className="scroll">
                  <table className="table">
                    <thead><tr><th>Model</th><th>CV macro-F1</th><th>Val accuracy</th><th>Val macro-F1</th><th>Test accuracy</th><th>Test precision (macro)</th><th>Test recall (macro)</th><th>Test macro-F1</th><th>Test weighted F1</th></tr></thead>
                    <tbody>
                      {d.models.map((r) => (
                        <tr key={r.model} className={r.model === d.selected_model ? "hl" : ""}>
                          <td><strong>{r.model}</strong>{r.model === d.selected_model && " (selected)"}</td>
                          <td>{r.cv_f1_macro === null ? "-" : f3(r.cv_f1_macro)}</td>
                          <td>{f3(r.validation.accuracy)}</td><td>{f3(r.validation.f1_macro)}</td>
                          <td>{f3(r.test.accuracy)}</td><td>{f3(r.test.precision_macro)}</td><td>{f3(r.test.recall_macro)}</td>
                          <td>{f3(r.test.f1_macro)}</td><td>{f3(r.test.f1_weighted)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
              <div className="grid two top">
                <Card title="Macro-F1 by model">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={chart}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="model" tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
                      <Bar dataKey="Validation" fill="#1877f2" /><Bar dataKey="Test" fill="#f28e2b" />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
                <Card title="Confusion matrix" subtitle="Selected model, test split (rows = actual)">
                  <ConfusionMatrix matrix={m.confusion_matrix} labels={d.labels} />
                </Card>
              </div>
              <div className="grid two top">
                <Card title="Per-class results (test)">
                  <table className="table">
                    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
                    <tbody>{LABELS.slice().sort().map((k) => (
                      <tr key={k}><td><span className={`tag ${k}`}>{k}</span></td><td>{f3(m.per_class[k].precision)}</td><td>{f3(m.per_class[k].recall)}</td><td>{f3(m.per_class[k].f1)}</td><td>{m.per_class[k].support}</td></tr>
                    ))}</tbody>
                  </table>
                </Card>
                <Card title="Overfitting and success criteria">
                  <p><strong>{d.overfitting_check.verdict}</strong></p>
                  <p className="muted small">Train macro-F1 {f3(d.overfitting_check.train_f1_macro)}, validation {f3(d.overfitting_check.validation_f1_macro)}, test {f3(d.overfitting_check.test_f1_macro)}. {d.overfitting_check.underfitting_note}</p>
                  <ul className="plain">
                    <li>{d.success_criteria.criterion_1}: <strong>{d.success_criteria.criterion_1_met ? "met" : "not met"}</strong></li>
                    <li>{d.success_criteria.criterion_2}: <strong>{d.success_criteria.criterion_2_met ? "met" : "not met"}</strong> (ceiling {d.success_criteria.label_consistency_ceiling_accuracy_pct}%)</li>
                  </ul>
                </Card>
              </div>
              <Card title="Error analysis" subtitle="Largest confusions and sample misclassified test posts" className="top">
                <p>{d.top_confusions.map((c) => `${c.count} x actual ${c.true} predicted as ${c.predicted}`).join("; ")}.</p>
                <div className="scroll">
                  <table className="table">
                    <thead><tr><th>Comment</th><th>Category</th><th>Actual</th><th>Predicted</th></tr></thead>
                    <tbody>{d.error_examples.map((e, i) => (
                      <tr key={i}><td>{e.text}</td><td>{String(e.category).replace(/_/g, " ")}</td><td><span className={`tag ${e.sentiment}`}>{e.sentiment}</span></td><td><span className={`tag ${e.pred}`}>{e.pred}</span></td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </Card>
            </>
          );
        }}
      </Async>
    </>
  );
}
