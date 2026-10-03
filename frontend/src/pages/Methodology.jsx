import { api } from "../services/api.js";
import { Async, Card, PageHeader, useApi } from "../components/State.jsx";
import { fmt, pct } from "../theme.js";

// Phase names and deliverables exactly as listed in the AI Project Lifecycle Guide (PDF).
const PHASES = [
  ["Problem Definition", "Problem Statement"],
  ["Requirement Analysis", "Requirement Specification (SRS)"],
  ["Data Acquisition", "Raw Dataset"],
  ["Data Preparation", "Prepared Dataset"],
  ["Exploratory Data Analysis (EDA)", "EDA Report"],
  ["Model Selection", "Selected Model"],
  ["Model Development", "Trained Model"],
  ["Model Evaluation", "Evaluation Report"],
  ["Solution Integration & Testing", "Working AI Solution"],
];

export default function Methodology() {
  const state = useApi(() => api.datasetInfo());
  return (
    <>
      <PageHeader title="Project Methodology" subtitle="The nine lifecycle phases from the AI Project Lifecycle Guide, plus the dataset facts they rely on." />
      <Card title="Lifecycle phases">
        <ol className="phases">
          {PHASES.map(([name, deliverable], i) => (
            <li key={name}>
              <span className="phase-no">{i + 1}</span>
              <div><strong>{name}</strong><div className="muted small">Deliverable: {deliverable} - see reports/phases/phase_{String(i + 1).padStart(2, "0")}.md</div></div>
            </li>
          ))}
        </ol>
      </Card>
      <Async state={state}>
        {(d) => {
          const lc = d.label_consistency;
          return (
            <>
              <div className="grid two top">
                <Card title="Dataset facts (computed)">
                  <ul className="plain">
                    <li>File: <strong>{d.source_file}</strong></li>
                    <li>Rows: <strong>{fmt(d.rows)}</strong>, unique texts: <strong>{fmt(d.unique_texts)}</strong>, duplicate texts: <strong>{fmt(d.duplicate_texts)}</strong></li>
                    <li>Missing values: <strong>{fmt(d.missing_values_total)}</strong>; empty texts: <strong>{fmt(d.empty_text)}</strong></li>
                    <li>Columns: {d.columns.map((c) => c.name).join(", ")}</li>
                    <li>Split: {d.split ? `train ${fmt(d.split.train.rows)}, validation ${fmt(d.split.validation.rows)}, test ${fmt(d.split.test.rows)} (${d.split.leakage_check})` : "-"}</li>
                  </ul>
                </Card>
                <Card title="Data-quality finding">
                  <p>
                    {fmt(lc.texts_with_conflicting_labels)} of {fmt(lc.unique_texts)} unique texts appear with more than one sentiment label
                    ({fmt(lc.rows_in_conflicting_texts)} rows). Even a perfect text-only model could not exceed about <strong>{lc.max_possible_text_only_accuracy_pct}%</strong> accuracy.
                  </p>
                  <p className="muted small">The original dataset was not modified; no rows were deleted or relabelled.</p>
                </Card>
              </div>
              <Card title="Formulas used on the dashboard" className="top">
                <ul className="plain">
                  <li><strong>Sentiment health</strong> = 100 x (positive + 0.5 x neutral) / total posts</li>
                  <li><strong>Customer voice index</strong> = 100 x (positive - negative) / (positive + negative), range -100 to +100</li>
                  <li><strong>Confidence</strong> = probability of the predicted class; high at 0.70 or more, medium from 0.50, low below 0.50</li>
                  <li><strong>Overall sentiment</strong> = the label with the most posts; MIXED when the top two are within 3 percentage points</li>
                </ul>
              </Card>
              <Card title="Limitations" className="top">
                <ul className="plain">
                  <li>Labels in the dataset are inconsistent for repeated texts, which caps accuracy.</li>
                  <li>The model uses TF-IDF word features, so it cannot understand sarcasm or context.</li>
                  <li>Texts are English; the language column is not reliable. {d.language_check?.note}</li>
                  <li>The dataset is a fixed file; this project does not collect live Facebook data.</li>
                </ul>
              </Card>
            </>
          );
        }}
      </Async>
    </>
  );
}
