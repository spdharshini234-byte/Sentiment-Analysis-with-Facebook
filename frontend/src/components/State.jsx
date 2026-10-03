import { useEffect, useState } from "react";

export const NO_DATA = "No data available for this visualization.";

/** Runs an async loader and returns { data, loading, error, reload }. */
export function useApi(loader, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => alive && setState({ data, loading: false, error: null }))
      .catch((e) => alive && setState({ data: null, loading: false, error: e.message }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { ...state, reload: () => setTick((t) => t + 1) };
}

export function Empty({ message = NO_DATA }) {
  return <div className="empty">{message}</div>;
}

export function Loading() {
  return <div className="loading"><span className="spinner" /> Loading...</div>;
}

export function ErrorBox({ message, onRetry }) {
  return (
    <div className="error-box">
      <strong>Something went wrong</strong>
      <p>{message}</p>
      {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
    </div>
  );
}

/** Renders loading / error states and calls children(data) when data is ready. */
export function Async({ state, children }) {
  if (state.loading) return <Loading />;
  if (state.error) return <ErrorBox message={state.error} onRetry={state.reload} />;
  return children(state.data);
}

export function Card({ title, subtitle, children, className = "", right }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <header className="card-head">
          <div>
            <h3>{title}</h3>
            {subtitle && <p className="muted small">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle }) {
  return (
    <div className="page-head">
      <h1>{title}</h1>
      {subtitle && <p className="muted">{subtitle}</p>}
    </div>
  );
}
