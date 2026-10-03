import { Empty } from "./State.jsx";

const ICON = { warning: "!", positive: "+", info: "i" };

export default function InsightPanel({ insights }) {
  if (!insights || !insights.length) return <Empty message="No insights could be generated from the available data." />;
  return (
    <ul className="insights">
      {insights.map((i, idx) => (
        <li key={idx} className={`insight sev-${i.severity}`}>
          <span className="insight-icon">{ICON[i.severity] || "i"}</span>
          <span>{i.text}</span>
        </li>
      ))}
    </ul>
  );
}
