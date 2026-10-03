import { api } from "../services/api.js";
import { Async, Card, Empty, PageHeader, useApi } from "../components/State.jsx";
import MetricCard from "../components/MetricCard.jsx";
import SentimentChart from "../components/SentimentChart.jsx";
import KeywordChart from "../components/KeywordChart.jsx";
import InsightPanel from "../components/InsightPanel.jsx";
import { fmt, pct } from "../theme.js";

function CommentList({ items, tone }) {
  if (!items || !items.length) return <Empty />;
  return (
    <ul className="comments">
      {items.map((c, i) => (
        <li key={i}>
          <span>{c.comment}</span>
          <span className={`tag ${tone}`}>{pct(c.confidence * 100, 0)}</span>
        </li>
      ))}
    </ul>
  );
}

function WordList({ words, label }) {
  if (!words || !words.length) return <Empty />;
  return (
    <ul className="wordlist">
      {words.map((w) => (
        <li key={w.word}>
          <strong>{w.word}</strong>
          <span className="muted small">{w.share_in_class_pct}% of {w.posts} posts {label}</span>
        </li>
      ))}
    </ul>
  );
}

export default function Dashboard() {
  const state = useApi(() => api.dashboard());
  return (
    <>
      <PageHeader title="Executive Sentiment Dashboard" subtitle="Turning Facebook conversations into actionable sentiment insights." />
      <Async state={state}>
        {(d) => {
          const t = d.totals;
          return (
            <>
              <div className="grid kpis">
                <MetricCard label="Total posts" value={fmt(t.total)} sub={d.date_range ? `${d.date_range.min} to ${d.date_range.max}` : undefined} />
                <MetricCard tone="positive" label="Positive" value={fmt(t.counts.positive)} sub={pct(t.percentages.positive)} progress={t.percentages.positive} />
                <MetricCard tone="negative" label="Negative" value={fmt(t.counts.negative)} sub={pct(t.percentages.negative)} progress={t.percentages.negative} />
                <MetricCard tone="neutral" label="Neutral" value={fmt(t.counts.neutral)} sub={pct(t.percentages.neutral)} progress={t.percentages.neutral} />
                <MetricCard label="Overall sentiment" value={t.overall_sentiment} />
                <MetricCard label="Sentiment health" value={d.health.score === null ? "-" : `${d.health.score} / 100`} hint={d.health.formula} sub="see Methodology for formula" progress={d.health.score} />
                <MetricCard label="Customer voice index" value={d.voice_index.score === null ? "-" : d.voice_index.score} hint={d.voice_index.formula} sub="-100 to +100" />
                <MetricCard
                  label="Model macro-F1"
                  value={d.model ? d.model.f1_macro.toFixed(3) : "-"}
                  sub={d.model ? `accuracy ${d.model.accuracy.toFixed(3)} (${d.model.evaluated_on})` : "model report unavailable"}
                />
              </div>

              <div className="grid two top">
                <Card title="Sentiment distribution" subtitle="Labels from the dataset"><SentimentChart counts={t.counts} /></Card>
                <Card title="Most common words" subtitle="Posts containing each word (stop-words removed)"><KeywordChart words={d.common_words} /></Card>
              </div>

              <Card title="Insights" subtitle="Rule-based statements generated from the calculated statistics (not generative AI)" className="top">
                <InsightPanel insights={d.insights} />
              </Card>

              <div className="grid two top">
                <Card title="Most negative comments" subtitle="Highest model probability of negative, also labelled negative"><CommentList items={d.top_negative_comments} tone="negative" /></Card>
                <Card title="Most positive comments" subtitle="Highest model probability of positive, also labelled positive"><CommentList items={d.top_positive_comments} tone="positive" /></Card>
              </div>

              <div className="grid two top">
                <Card title="Top concern areas" subtitle="Words most strongly associated with negative posts"><WordList words={d.concern_areas} label="are negative" /></Card>
                <Card title="Positive drivers" subtitle="Words most strongly associated with positive posts"><WordList words={d.positive_drivers} label="are positive" /></Card>
              </div>
            </>
          );
        }}
      </Async>
    </>
  );
}
