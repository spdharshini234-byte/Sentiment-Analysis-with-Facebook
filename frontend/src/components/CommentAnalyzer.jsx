import { useState } from "react";
import { api } from "../services/api.js";
import { Card, ErrorBox } from "./State.jsx";
import { LABELS, SENTIMENT_COLORS, pct } from "../theme.js";

const EXAMPLES = [
  "Having trouble logging into my account, any fix?",
  "Thanks for resolving my issue fast!",
  "Matchday! Lions vs Tigers at 8pm. Predictions?",
];

export default function CommentAnalyzer() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function run(value) {
    const t = (value ?? text).trim();
    if (!t) {
      setError("Please type a comment first.");
      setResult(null);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setResult(await api.analyze(t));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid two">
      <Card title="Enter a Facebook-style comment" subtitle="The trained model scores it locally. No external AI service is used.">
        <textarea
          className="input"
          rows={5}
          maxLength={5000}
          value={text}
          placeholder="Type or paste a comment..."
          onChange={(e) => setText(e.target.value)}
        />
        <div className="row gap">
          <button className="btn primary" onClick={() => run()} disabled={busy}>{busy ? "Analyzing..." : "Analyze sentiment"}</button>
          <button className="btn" onClick={() => { setText(""); setResult(null); setError(null); }}>Clear</button>
        </div>
        <p className="muted small top">Try an example:</p>
        <div className="chips">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => { setText(ex); run(ex); }}>{ex}</button>
          ))}
        </div>
        {error && <ErrorBox message={error} />}
      </Card>

      <Card title="Result">
        {!result && <div className="empty">Results appear here after you analyze a comment.</div>}
        {result && (
          <div>
            <div className="verdict" style={{ borderColor: SENTIMENT_COLORS[result.sentiment] }}>
              <span className="pill" style={{ background: SENTIMENT_COLORS[result.sentiment] }}>{result.sentiment.toUpperCase()}</span>
              <span className="big">{pct(result.confidence * 100, 0)}</span>
              <span className="muted">confidence ({result.confidence_level})</span>
            </div>
            <p>{result.explanation}</p>
            {result.warning && <p className="warn">{result.warning}</p>}
            <h4>Probability of each sentiment</h4>
            {LABELS.map((k) => (
              <div key={k} className="prob">
                <span className="prob-name">{k}</span>
                <div className="bar"><div className="bar-fill" style={{ width: `${result.probabilities[k] * 100}%`, background: SENTIMENT_COLORS[k] }} /></div>
                <span className="prob-val">{pct(result.probabilities[k] * 100)}</span>
              </div>
            ))}
            <h4>Cleaned text</h4>
            <code className="code">{result.cleaned_text || "(empty after cleaning)"}</code>
            <h4>Words that influenced the prediction</h4>
            {result.important_words.length === 0 ? (
              <p className="muted small">No known words found in this comment.</p>
            ) : (
              <div className="chips">
                {result.important_words.map((w) => (
                  <span key={w.word} className={`chip static ${w.direction === "supports" ? "good" : "bad"}`}>
                    {w.word} <small>{w.direction === "supports" ? "supports" : "opposes"}</small>
                  </span>
                ))}
              </div>
            )}
            <p className="muted small top">Model: {result.model}</p>
          </div>
        )}
      </Card>
    </div>
  );
}
