import { useState } from "react";
import { api } from "../services/api.js";
import { Async, Card, Empty, PageHeader, useApi } from "../components/State.jsx";
import KeywordChart from "../components/KeywordChart.jsx";
import { SENTIMENT_COLORS, pct } from "../theme.js";

export default function Keywords() {
  const [filter, setFilter] = useState("");
  const words = useApi(() => api.commonWords({ limit: 15, sentiment: filter }), [filter]);
  const ks = useApi(() => api.keywordSentiment({ top_n: 15, min_count: 30 }));
  const color = filter ? SENTIMENT_COLORS[filter] : "#1877f2";

  return (
    <>
      <PageHeader title="Keyword Intelligence" subtitle="Frequent words and what sentiment they carry. Stop-words, URLs and symbols are removed." />
      <div className="grid two">
        <Card
          title="Most common words"
          right={
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="">All posts</option><option value="positive">Positive only</option>
              <option value="neutral">Neutral only</option><option value="negative">Negative only</option>
            </select>
          }
        >
          <Async state={words}>{(d) => <KeywordChart words={d.words} color={color} />}</Async>
        </Card>
        <Card title="TF-IDF keywords" subtitle="Words with the highest average TF-IDF weight">
          <Async state={words}>
            {(d) => d.tfidf_keywords.length === 0 ? <Empty /> : (
              <ul className="wordlist">
                {d.tfidf_keywords.map((w) => <li key={w.word}><strong>{w.word}</strong><span className="muted small">score {w.score}</span></li>)}
              </ul>
            )}
          </Async>
        </Card>
      </div>
      <Card title="Sentiment x keyword insights" subtitle="For each frequent keyword: how its posts split across sentiments" className="top">
        <Async state={ks}>
          {(d) => d.keywords.length === 0 ? <Empty /> : (
            <div className="scroll">
              <table className="table">
                <thead><tr><th>Keyword</th><th>Posts</th><th>Positive</th><th>Neutral</th><th>Negative</th><th>Split</th></tr></thead>
                <tbody>
                  {d.keywords.map((k) => (
                    <tr key={k.keyword}>
                      <td><strong>{k.keyword}</strong></td><td>{k.count}</td>
                      <td>{pct(k.positive_pct)}</td><td>{pct(k.neutral_pct)}</td><td>{pct(k.negative_pct)}</td>
                      <td>
                        <div className="stack">
                          <span style={{ width: `${k.positive_pct}%`, background: SENTIMENT_COLORS.positive }} />
                          <span style={{ width: `${k.neutral_pct}%`, background: SENTIMENT_COLORS.neutral }} />
                          <span style={{ width: `${k.negative_pct}%`, background: SENTIMENT_COLORS.negative }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Async>
      </Card>
    </>
  );
}
