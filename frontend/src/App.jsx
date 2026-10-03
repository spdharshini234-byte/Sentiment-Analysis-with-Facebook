import { useEffect, useState } from "react";
import Dashboard from "./pages/Dashboard.jsx";
import Pulse from "./pages/Pulse.jsx";
import Trend from "./pages/Trend.jsx";
import Analyzer from "./pages/Analyzer.jsx";
import Batch from "./pages/Batch.jsx";
import CustomerVoice from "./pages/CustomerVoice.jsx";
import Keywords from "./pages/Keywords.jsx";
import ModelPerformance from "./pages/ModelPerformance.jsx";
import Methodology from "./pages/Methodology.jsx";
import { api } from "./services/api.js";

const NAV = [
  ["overview", "Overview", Dashboard],
  ["pulse", "Sentiment Pulse", Pulse],
  ["trend", "Trend Monitor", Trend],
  ["analyzer", "Comment Analyzer", Analyzer],
  ["batch", "Batch Analyzer", Batch],
  ["voice", "Customer Voice", CustomerVoice],
  ["keywords", "Keyword Intelligence", Keywords],
  ["model", "Model Performance", ModelPerformance],
  ["method", "Project Methodology", Methodology],
];

const fromHash = () => {
  const id = window.location.hash.replace("#/", "");
  return NAV.some((n) => n[0] === id) ? id : "overview";
};

export default function App() {
  const [page, setPage] = useState(fromHash());
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [open, setOpen] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const onHash = () => setPage(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);
  useEffect(() => {
    api.health().then(setHealth).catch((e) => setHealth({ status: "offline", error: e.message }));
  }, []);

  const go = (id) => {
    window.location.hash = `#/${id}`;
    setOpen(false);
  };
  const Page = NAV.find((n) => n[0] === page)[2];

  return (
    <div className="shell">
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <div className="logo">f</div>
          <div>
            <div className="brand-name">Sentiment Analysis</div>
            <div className="brand-sub">with Facebook</div>
          </div>
        </div>
        <nav>
          {NAV.map(([id, label]) => (
            <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => go(id)}>{label}</button>
          ))}
        </nav>
        <div className="side-foot">
          <div className={`status ${health?.status || "loading"}`}>
            <span className="dot" />
            {health ? (health.status === "ok" ? "Backend and model ready" : health.status === "offline" ? "Backend offline" : "Backend degraded") : "Checking backend..."}
          </div>
          <button className="btn small-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </aside>
      <main className="main">
        <div className="topbar">
          <button className="btn menu" onClick={() => setOpen(!open)} aria-label="Menu">Menu</button>
          <span className="tagline">Turning Facebook conversations into actionable sentiment insights.</span>
        </div>
        {health && health.status !== "ok" && (
          <div className="error-box">
            <strong>{health.status === "offline" ? "The backend is not reachable." : "The backend is running but not fully ready."}</strong>
            <p>{health.error || health.model_error || "Run python scripts/train_model.py and python scripts/build_database.py, then restart the backend."}</p>
          </div>
        )}
        <div className="content"><Page /></div>
      </main>
    </div>
  );
}
