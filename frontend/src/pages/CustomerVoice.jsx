import { api } from "../services/api.js";
import { Async, Card, Empty, PageHeader, useApi } from "../components/State.jsx";
import { pct } from "../theme.js";

export default function CustomerVoice() {
  const state = useApi(() => api.negativeFeedback({ limit: 25 }));
  return (
    <>
      <PageHeader title="Customer Voice - Top Customer Complaints" subtitle="Comments classified as negative, sorted by model confidence." />
      <Async state={state}>
        {(d) => (
          <Card title="Top complaints" subtitle={d.definition}>
            {d.items.length === 0 ? <Empty /> : (
              <div className="scroll">
                <table className="table">
                  <thead><tr><th>Comment</th><th>Sentiment</th><th>Confidence</th><th>Category</th><th>Latest date</th><th>Times seen</th></tr></thead>
                  <tbody>
                    {d.items.map((c, i) => (
                      <tr key={i}>
                        <td>{c.comment}</td>
                        <td><span className="tag negative">{c.predicted_sentiment}</span></td>
                        <td>{pct(c.confidence * 100)}</td>
                        <td>{(c.category || "-").replace(/_/g, " ")}</td>
                        <td>{c.latest_date || "-"}</td>
                        <td>{c.occurrences}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="muted small top">
              Emotion is not shown because the dataset has no emotion field. Platform/source is represented by <em>{d.source_field}</em>.
            </p>
          </Card>
        )}
      </Async>
    </>
  );
}
