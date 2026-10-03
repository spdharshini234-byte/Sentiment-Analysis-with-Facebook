export const SENTIMENT_COLORS = { positive: "#30a46c", neutral: "#8b95a7", negative: "#e5484d" };
export const LABELS = ["positive", "neutral", "negative"];
export const fmt = (n) => (n === null || n === undefined ? "-" : Number(n).toLocaleString());
export const pct = (n, d = 1) => (n === null || n === undefined ? "-" : `${Number(n).toFixed(d)}%`);
export const f3 = (n) => (n === null || n === undefined ? "-" : Number(n).toFixed(3));
