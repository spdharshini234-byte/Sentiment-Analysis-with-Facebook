export default function MetricCard({ label, value, sub, tone, progress, hint }) {
  return (
    <div className={`metric ${tone ? "tone-" + tone : ""}`} title={hint || ""}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub !== undefined && <div className="metric-sub">{sub}</div>}
      {progress !== undefined && progress !== null && (
        <div className="bar"><div className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
      )}
    </div>
  );
}
