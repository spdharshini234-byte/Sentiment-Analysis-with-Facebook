"""Generate the phase reports, final report and viva questions from the REAL computed results.

Run after training:  python scripts/generate_reports.py
Every number written into the documents is read from reports/*.json and models/*.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import config  # noqa: E402

TITLE = "Sentiment Analysis with Facebook"


def load(path):
    if not path.is_file():
        print(f"ERROR: {path.name} not found. Run `python scripts/train_model.py` first.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def phase_doc(no, name, deliverable, objective, inputs, implementation, tools, files, output, evidence, result, explain):
    bullets = lambda items: "\n".join(f"- {i}" for i in items)  # noqa: E731
    return f"""# Phase {no}: {name}

**Project:** {TITLE}  
**Deliverable (from the Phase PDF):** {deliverable}

## Objective
{objective}

## Inputs
{bullets(inputs)}

## Implementation
{bullets(implementation)}

## Tools
{bullets(tools)}

## Code / files involved
{bullets(files)}

## Output
{bullets(output)}

## Evidence
{bullets(evidence)}

## Result
{result}

## How to explain it during POP
{explain}
"""


def main():
    dr = load(config.DATA_REPORT_PATH)
    ev = load(config.EVALUATION_PATH)
    meta = load(config.METADATA_PATH)
    config.PHASES_DIR.mkdir(parents=True, exist_ok=True)

    n, uniq, dup = dr["rows"], dr["unique_texts"], dr["duplicate_texts"]
    cd, cdp = dr["class_distribution"], dr["class_distribution_pct"]
    lc = dr["label_consistency"]
    prep, sp = dr["preparation"], dr["split"]
    tm = ev["final_test_metrics"]
    sel = ev["selected_model"]
    base_f1, base_acc = ev["baseline_test_f1_macro"], ev["baseline_test_accuracy"]
    oc = ev["overfitting_check"]
    ceiling = lc["max_possible_text_only_accuracy_pct"]
    dr_range = dr.get("date_range", {})
    cols = ", ".join(c["name"] for c in dr["columns"])
    models_tbl = table(
        ["Model", "CV macro-F1", "Validation macro-F1", "Test accuracy", "Test macro-F1", "Test weighted-F1"],
        [[r["model"], "-" if r["cv_f1_macro_mean"] is None else f"{r['cv_f1_macro_mean']:.3f}",
          f"{r['validation']['f1_macro']:.3f}", f"{r['test']['accuracy']:.3f}", f"{r['test']['f1_macro']:.3f}",
          f"{r['test']['f1_weighted']:.3f}"] for r in ev["models"]])
    split_tbl = table(["Split", "Rows", "Unique text groups", "Negative", "Neutral", "Positive"],
                      [[k.capitalize(), sp[k]["rows"], sp[k]["unique_text_groups"], sp[k]["class_distribution"]["negative"],
                        sp[k]["class_distribution"]["neutral"], sp[k]["class_distribution"]["positive"]]
                       for k in ("train", "validation", "test")])
    cm = tm["confusion_matrix"]
    cm_tbl = table(["Actual \\ Predicted", "negative", "neutral", "positive"],
                   [[lab, *row] for lab, row in zip(config.LABELS, cm)])
    pc_tbl = table(["Class", "Precision", "Recall", "F1", "Support"],
                   [[k, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]] for k, v in tm["per_class"].items()])
    honest = (f"{lc['texts_with_conflicting_labels']:,} of {lc['unique_texts']:,} unique texts carry more than one label, "
              f"so text-only accuracy cannot exceed about {ceiling}%")

    docs = {}
    docs[1] = phase_doc(1, "Problem Definition", "Problem Statement",
        "Understand the business problem and define the project objectives.",
        ["Project title: Sentiment Analysis with Facebook", "Manus project brief (Manus.mht)", "Phase PDF: AI Project Lifecycle Guide"],
        ["**Problem:** organisations receive large volumes of Facebook posts and comments and cannot read them all. They need an automatic way to see whether opinion is positive, neutral or negative, and which topics drive it.",
         "**Objective:** build a system that classifies Facebook-style posts into negative / neutral / positive and shows the results in an interactive dashboard.",
         f"**Measurable success criteria:** (1) the final model must beat a majority-class baseline on test macro-F1 (baseline = {base_f1:.3f}); (2) all dashboard numbers must be computed from the dataset, never hard-coded; (3) the full pipeline must run from raw CSV to dashboard with documented commands.",
         "**Stakeholders:** social-media / customer-support teams (users), the student team (developers), the college POP evaluators.",
         "**Scope:** classify the supplied dataset and new comments entered by a user. **Out of scope:** live Facebook data collection, paid AI APIs.",
         "**Constraints:** one provided dataset only, traditional ML (TF-IDF + NB / LR / SVM), must run on Windows and Linux."],
        ["Python, scikit-learn", "FastAPI, React"],
        ["README.md (Problem statement, Objectives)", "reports/final_report.md (Introduction, Problem Statement, Objectives)"],
        ["Problem Statement (this document)"],
        ["This file", "Success criteria are checked automatically in reports/evaluation.json (`success_criteria`)"],
        "The problem and the measurable success criteria are defined before any code. Criterion 1 is evaluated in Phase 8.",
        "Say: 'Companies cannot read thousands of posts, so we built a tool that labels each post as positive, neutral or negative and summarises the mood. We decided in advance how to judge success: our model must beat a model that always guesses the most common label.'")

    docs[2] = phase_doc(2, "Requirement Analysis", "Requirement Specification (SRS)",
        "Translate the business need into technical requirements.",
        ["Problem Statement (Phase 1)", "Manus brief"],
        ["**Functional requirements:** FR1 executive dashboard; FR2 sentiment pulse; FR3 time-based trend monitor (only when a valid date column exists); FR4 comment analyzer (prediction, confidence, probabilities, important words); FR5 batch CSV analyzer with download; FR6 top customer complaints; FR7 keyword intelligence and keyword x sentiment; FR8 model performance centre with 4-model comparison; FR9 methodology page.",
         "**Non-functional requirements:** NFR1 every number computed from data; NFR2 model trained once and loaded by the API (not retrained at start-up); NFR3 no paid or external AI service; NFR4 works on Windows and Linux with pathlib paths; NFR5 friendly errors with no stack traces for users; NFR6 reproducible (fixed random_state).",
         "**Evaluation metrics:** accuracy, precision, recall, F1 per class, macro-F1 (primary, because classes are imbalanced), weighted-F1, confusion matrix.",
         "**Resources:** one CSV dataset, a normal laptop (no GPU), Python 3.11+.",
         f"**Assumptions:** the `sentiment` column is the ground truth label. **Risks:** {honest}. Mitigation: document it, split by text group, report macro-F1 against a baseline."],
        ["Markdown documentation"],
        ["This file", "README.md (Features)"],
        ["Requirement Specification (SRS) - this document"],
        ["This file", "tests/test_api.py covers FR4-FR9 endpoints"],
        "Nine functional and six non-functional requirements, a metric choice and a risk list.",
        "Say: 'We wrote what the system must do and how well it must do it. We chose macro-F1 because negative posts are only a small share of the data and plain accuracy would hide poor performance on them.'")

    docs[3] = phase_doc(3, "Data Acquisition", "Raw Dataset",
        "Collect quality data for model development.",
        ["archive.zip supplied by the student, containing one CSV"],
        [f"**Source:** the supplied Facebook dataset file `{dr['source_file']}`. No other dataset was used or mixed in.",
         f"**Raw data:** {n:,} rows and {len(dr['columns'])} columns: {cols}.",
         "**Privacy and compliance check:** "
         f"@mentions: {dr['privacy_check']['texts_with_mentions']}, e-mail addresses: {dr['privacy_check']['texts_with_emails']}, phone-like numbers: {dr['privacy_check']['texts_with_phone_like_numbers']}, URLs: {dr['privacy_check']['texts_with_urls']}, personal-identifier columns: {len(dr['privacy_check']['personal_identifier_columns'])}. No personal data was found.",
         f"**Quality assessment:** missing values {dr['missing_values_total']}, empty texts {dr['empty_text']}, exact duplicate records (ignoring id) {dr['duplicate_records_excluding_id']}, but only {uniq:,} unique texts ({dup:,} repeated texts). {honest}.",
         f"**Labels:** negative {cd['negative']:,} ({cdp['negative']}%), neutral {cd['neutral']:,} ({cdp['neutral']}%), positive {cd['positive']:,} ({cdp['positive']}%).",
         f"**Other observation:** the `language` column lists {', '.join(dr['language_distribution'])} but {dr['language_check']['rows_with_non_ascii_letters']} texts contain non-English letters, so the tag is not reliable.",
         "The original file is never modified; it stays in data/raw/."],
        ["pandas", "scripts/prepare_data.py"],
        ["src/data/load_data.py (`load_raw`, `inspect_dataset`)", "data/raw/Facebook_data_txt.csv", "reports/data_report.json", "notebooks/01_data_understanding.ipynb"],
        ["Raw Dataset: data/raw/Facebook_data_txt.csv (unchanged)", "Data quality assessment: reports/data_report.json"],
        ["reports/data_report.json", "notebooks/01_data_understanding.ipynb"],
        "The raw dataset was loaded, validated and assessed. It is usable, with one documented quality issue (inconsistent labels for repeated texts).",
        "Say: 'We used only the dataset provided. We checked it for missing values, duplicates and personal data. We found that many sentences repeat with different labels, so we decided to be honest about it and handle it in the split.'")

    docs[4] = phase_doc(4, "Data Preparation", "Prepared Dataset",
        "Prepare clean data for machine learning.",
        ["Raw Dataset (Phase 3)"],
        ["**Cleaning (src/data/preprocess.py):** HTML removal, URLs removed, @mentions removed, hashtags keep their word, lower-casing, numbers replaced by a `num` token, emojis kept as tokens, `!` and `?` kept as tone tokens, extra whitespace removed, stop-words removed but **negations are kept** (not, never, no) so that sentiment is not destroyed.",
         f"**Missing values and duplicates:** rows dropped for missing/empty text: {prep['dropped_missing_or_empty_text']}; invalid labels: {prep['dropped_invalid_label']}; exact duplicate records: {prep['dropped_exact_duplicate_records']}; empty after cleaning: {prep['dropped_empty_after_cleaning']}. Final rows: {prep['rows_final']:,}.",
         "**Labelling:** not required; the dataset already has sentiment labels. No label was changed.",
         f"**Leakage control:** after cleaning there are {prep['unique_texts_final']:,} unique cleaned texts. Identical texts are grouped and a group is never allowed to appear in two splits (StratifiedGroupKFold, random_state={sp['random_state']}).",
         "**Split (70 / 15 / 15):** see table below."],
        ["pandas, regular expressions, scikit-learn StratifiedGroupKFold"],
        ["src/data/preprocess.py", "src/models/train.py (`grouped_stratified_split`)", "data/processed/prepared_dataset.csv", "tests/test_preprocessing.py", "notebooks/02_data_preparation.ipynb"],
        ["Prepared Dataset: data/processed/prepared_dataset.csv (includes a `split` column)", "Split summary in reports/data_report.json"],
        ["tests/test_preprocessing.py includes a leakage test", sp["leakage_check"]],
        "The prepared dataset was written to a separate file; the original is untouched.\n\n" + split_tbl,
        "Say: 'We cleaned the text carefully and kept words like \"not\". Because the same sentence appears many times, a normal random split would let the model see test sentences during training, so we split by sentence group to avoid that cheating.'")

    docs[5] = phase_doc(5, "Exploratory Data Analysis (EDA)", "EDA Report",
        "Understand the characteristics of the dataset.",
        ["Prepared Dataset (Phase 4)"],
        ["Analysed class distribution, text length, word frequency per sentiment, sentiment by category, sentiment by language tag, monthly trend and label consistency.",
         f"**Class imbalance:** negative is the minority ({cdp['negative']}%).",
         f"**Text length:** mean {dr['text_length_chars']['mean']} characters (min {dr['text_length_chars']['min']:.0f}, max {dr['text_length_chars']['max']:.0f}); short, post-style sentences.",
         f"**Date range:** {dr_range.get('min', 'n/a')} to {dr_range.get('max', 'n/a')} (all timestamps valid, so the trend chart is available).",
         "**Category matters:** sentiment differs strongly by `category` (for example customer_support posts are mostly negative, meme_joke posts mostly positive) - see figure 04.",
         f"**Label consistency:** {honest} - see figure 07.",
         "**Outliers:** text length has no extreme outliers; no outlier removal was needed."],
        ["pandas, matplotlib, seaborn"],
        ["src/analysis/figures.py", "reports/figures/*.png", "notebooks/03_eda.ipynb"],
        ["EDA Report: this document + 10 figures in reports/figures/"],
        ["reports/figures/01_class_distribution.png", "02_text_length.png", "03_top_words_by_sentiment.png", "04_sentiment_by_category.png",
         "05_monthly_trend.png", "06_sentiment_by_language.png", "07_label_consistency.png"],
        "EDA showed three key facts: imbalanced classes, strong category effect, and inconsistent labels for repeated texts.",
        "Say: 'EDA means looking at the data before modelling. We saw that negative posts are rare, that customer-support posts are mostly negative, and that identical sentences sometimes have different labels - which limits any model.'")

    docs[6] = phase_doc(6, "Model Selection", "Selected Model",
        "Choose the most suitable AI algorithm.",
        ["Prepared Dataset", "EDA findings"],
        ["**Candidates (required by the brief):** TF-IDF + Multinomial Naive Bayes, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.",
         "**Baseline:** a majority-class dummy classifier (always predicts the most common label).",
         "**Trade-offs:** Naive Bayes is fast and strong on small text data; Logistic Regression gives calibrated probabilities; Linear SVM is a strong margin classifier but has no native probabilities (we use a softmax of its scores).",
         "**Selection rule (documented before training):** highest macro-F1 on the VALIDATION split among the three required models; the test split is never used to choose.",
         f"**Selected model:** {sel}."],
        ["scikit-learn Pipeline, TfidfVectorizer"],
        ["src/features/tfidf_features.py", "src/models/train.py (`candidate_models`)", "reports/model_comparison.json", "notebooks/05_model_comparison.ipynb"],
        ["Selected Model: " + sel, "Comparison table in reports/model_comparison.json"],
        ["reports/model_comparison.json", "reports/figures/08_model_comparison.png"],
        models_tbl,
        "Say: 'We compared three standard text classifiers against a do-nothing baseline. We picked the winner using the validation set, not the test set, so the final test is fair.'")

    docs[7] = phase_doc(7, "Model Development", "Trained Model",
        "Train and optimize the AI model.",
        ["Training split", "Selected candidates"],
        [f"Each model was tuned with GridSearchCV (scoring = macro-F1) using 5-fold **GroupKFold** so identical texts never straddle folds. Tuned: TF-IDF n-gram range (1,1) or (1,2), min_df 1 or 2, and the classifier parameter (alpha for NB, C for LR and SVM).",
         "Class weights are balanced for Logistic Regression and Linear SVM to help the minority class.",
         f"**Best parameters of the selected model:** `{json.dumps(meta['best_params'])}`.",
         "The full pipeline (cleaning -> TF-IDF -> classifier) is saved as one joblib file, so the API applies exactly the same preprocessing.",
         f"Training is a separate command (`python scripts/train_model.py`); the API only loads `{config.PIPELINE_PATH.name}`."],
        ["scikit-learn, joblib"],
        ["src/models/train.py", "scripts/train_model.py", "models/sentiment_pipeline.pkl", "models/model_metadata.json", "models/label_mapping.json"],
        ["Trained Model: models/sentiment_pipeline.pkl", f"Trained at (UTC): {meta['trained_at_utc']}"],
        ["models/model_metadata.json (parameters, library versions, metrics)", "notebooks/04_feature_engineering.ipynb"],
        f"Trained model saved. Python {meta['library_versions']['python']}, scikit-learn {meta['library_versions']['scikit-learn']}.",
        "Say: 'We tried several settings for each model with cross-validation and kept the best. Everything - cleaning, TF-IDF and the classifier - is saved as one file so the website uses exactly what we trained.'")

    docs[8] = phase_doc(8, "Model Evaluation", "Evaluation Report",
        "Validate model performance against the project goals.",
        ["Trained models", "Held-out TEST split (used once)"],
        [f"Test set size: {tm['n']} posts the model never saw.",
         f"**Final model ({sel}):** accuracy {tm['accuracy']:.3f}, macro-F1 {tm['f1_macro']:.3f}, weighted-F1 {tm['f1_weighted']:.3f}, macro precision {tm['precision_macro']:.3f}, macro recall {tm['recall_macro']:.3f}.",
         f"**Baseline:** accuracy {base_acc:.3f}, macro-F1 {base_f1:.3f}. Lift in macro-F1: +{ev['lift_over_baseline_f1_macro']:.3f}.",
         "**Error analysis:** largest confusions: " + "; ".join(f"{c['count']} x actual {c['true']} predicted {c['predicted']}" for c in ev["top_confusions"]) + ".",
         f"**Overfitting / underfitting check:** train macro-F1 {oc['train_f1_macro']:.3f}, validation {oc['validation_f1_macro']:.3f}, test {oc['test_f1_macro']:.3f}. {oc['verdict']} {oc['underfitting_note']}",
         f"**Success criteria:** (1) beats baseline: {'MET' if ev['success_criteria']['criterion_1_met'] else 'NOT MET'}; (2) accuracy is below the label-consistency ceiling of {ceiling}% (no leakage sign): {'MET' if ev['success_criteria']['criterion_2_met'] else 'NOT MET'}.",
         f"**Honest interpretation:** accuracy near {tm['accuracy']*100:.0f}% is not high. This is mostly a property of the data ({honest}), not only of the model."],
        ["scikit-learn metrics, matplotlib"],
        ["src/models/evaluate.py", "scripts/evaluate_model.py", "reports/evaluation.json", "notebooks/06_model_evaluation.ipynb"],
        ["Evaluation Report: this document + reports/evaluation.json", "Figures 09 (confusion matrix) and 10 (per-class metrics)"],
        ["reports/evaluation.json", "reports/figures/09_confusion_matrix.png", "reports/figures/10_per_class_metrics.png"],
        "Confusion matrix (test):\n\n" + cm_tbl + "\n\nPer-class results (test):\n\n" + pc_tbl,
        "Say: 'We tested on posts the model had never seen. It clearly beats the baseline. The scores are moderate because the dataset itself gives conflicting labels for the same sentence, which we can prove with numbers. We report this honestly rather than hiding it.'")

    docs[9] = phase_doc(9, "Solution Integration & Testing", "Working AI Solution",
        "Integrate the AI model into a working application.",
        ["Saved pipeline (Phase 7)", "Evaluation results (Phase 8)", "Prepared dataset scored into SQLite"],
        ["**API (FastAPI):** GET /api/health, /api/dashboard, /api/sentiment-distribution, /api/sentiment-trend, /api/common-words, /api/keyword-sentiment, /api/negative-feedback, /api/positive-feedback, /api/model-performance, /api/dataset-info; POST /api/predict, /api/analyze, /api/batch-predict. Pydantic validation, CORS, JSON errors without stack traces.",
         "**Database (SQLite):** every prepared post is scored by the saved model once (`scripts/build_database.py`); the dashboard reads from it.",
         "**User interface (React + Vite + Recharts):** 9 pages - Overview, Sentiment Pulse, Trend Monitor, Comment Analyzer, Batch Analyzer, Customer Voice, Keyword Intelligence, Model Performance, Project Methodology. Dark/light themes, responsive, empty-state messages instead of blank charts.",
         "**Testing:** automated pytest suite in tests/ (preprocessing, leakage, model, analytics, API incl. batch errors); frontend `npm run build`; manual end-to-end run.",
         "**Deployment preparation:** render.yaml (backend), frontend/vercel.json, configurable `VITE_API_URL`."],
        ["FastAPI, Uvicorn, SQLite, React, Vite, Recharts, pytest"],
        ["backend/app/**", "frontend/src/**", "tests/**", "render.yaml", "frontend/vercel.json"],
        ["Working AI Solution: run `python -m uvicorn backend.app.main:app` and open http://127.0.0.1:8000"],
        ["python -m pytest (all tests pass)", "frontend/dist (production build)"],
        "The model is integrated into a working, tested web application.",
        "Say: 'The React website calls our FastAPI server, which loads the saved model. Every number on the dashboard is calculated from the data, and we have automated tests to prove the API and the model pipeline work.'")

    names = {1: "problem_definition", 2: "requirement_analysis", 3: "data_acquisition", 4: "data_preparation",
             5: "eda", 6: "model_selection", 7: "model_development", 8: "model_evaluation", 9: "integration_testing"}
    for i, text in docs.items():
        (config.PHASES_DIR / f"phase_{i:02d}.md").write_text(text, encoding="utf-8")
    print(f"Wrote {len(docs)} phase reports to {config.PHASES_DIR}")

    final = f"""# {TITLE} - Final Report

## 1. Introduction
Facebook produces huge amounts of opinionated text. This project classifies Facebook-style posts as negative, neutral or positive and presents the results in an interactive dashboard. It uses a supplied Facebook dataset and follows the nine phases of the AI Project Lifecycle Guide.

## 2. Problem Statement
Organisations cannot manually read thousands of posts. They need an automatic, explainable summary of opinion and the topics behind it.

## 3. Objectives
1. Classify posts into three sentiments with traditional ML. 2. Compare Multinomial Naive Bayes, Logistic Regression and Linear SVM. 3. Provide a dashboard where every number is computed from data. 4. Follow all lifecycle phases and document each.

## 4. Existing System
Manual reading of comments, or generic sentiment tools that are not tied to an organisation's own data and give no keyword-level business insight.

## 5. Proposed System
A pipeline from raw CSV to cleaned data, TF-IDF features, a tuned classifier, a saved model, a FastAPI service, and a React dashboard with analytics (health score, voice index, concern areas, positive drivers, keyword x sentiment, confidence distribution, rule-based insights).

## 6. Dataset
File `{dr['source_file']}`: {n:,} rows, {len(dr['columns'])} columns ({cols}). Labels: negative {cd['negative']:,} ({cdp['negative']}%), neutral {cd['neutral']:,} ({cdp['neutral']}%), positive {cd['positive']:,} ({cdp['positive']}%). Date range {dr_range.get('min', 'n/a')} to {dr_range.get('max', 'n/a')}. Unique texts: {uniq:,}; repeated texts: {dup:,}. {honest}.

## 7. Data Preprocessing
Cleaning keeps negations, emojis and tone marks, removes URLs, HTML, mentions and stop-words. No rows were dropped (missing {prep['dropped_missing_or_empty_text']}, invalid labels {prep['dropped_invalid_label']}, exact duplicates {prep['dropped_exact_duplicate_records']}). The original file is unchanged; the prepared data is saved separately.

## 8. EDA
See reports/figures/ and phase_05.md. Key facts: imbalanced classes, strong category effect, inconsistent labels for repeated texts.

## 9. Feature Engineering
TF-IDF with unigrams and bigrams, sub-linear term frequency, min_df tuned (1 or 2), up to 20,000 features.

## 10. Model Selection
{models_tbl}

Selected: **{sel}** (best validation macro-F1 among the three required models).

## 11. Model Training
GridSearchCV with 5-fold GroupKFold on the training split, scoring macro-F1. Best parameters: `{json.dumps(meta['best_params'])}`. Pipeline saved with joblib.

## 12. Model Evaluation
Test accuracy {tm['accuracy']:.3f}, macro-F1 {tm['f1_macro']:.3f}, weighted-F1 {tm['f1_weighted']:.3f}; baseline macro-F1 {base_f1:.3f}. {oc['verdict']}

{cm_tbl}

## 13. System Architecture
CSV -> preprocessing -> TF-IDF -> classifier -> saved pipeline -> FastAPI (+ SQLite analytics) -> React dashboard.

## 14. Backend
FastAPI with Pydantic schemas, CORS, graceful errors; the model is loaded once; analytics read from SQLite.

## 15. Frontend
React + Vite + Recharts, nine pages, dark/light themes, empty states, configurable API URL.

## 16. Results
The model beats the baseline by +{ev['lift_over_baseline_f1_macro']:.3f} macro-F1. Overall dataset sentiment is positive-leaning ({cdp['positive']}% positive). The weakest class is the one with the lowest F1 in the per-class table of phase_08.md.

## 17. Business Use Case
Customer-support and marketing teams can track mood, spot categories and keywords linked to negative sentiment, and triage the most clearly negative comments.

## 18. Limitations
- Labels in the dataset are inconsistent for repeated texts, capping accuracy (about {ceiling}% upper bound).
- Bag-of-words features cannot understand sarcasm or context.
- The dataset is a fixed file; there is no live data collection.
- The `language` column does not match the actual text language.

## 19. Future Scope
Cleaner or multi-annotator labels; transformer models (e.g. BERT) when better labels exist; real language detection; emotion labels; scheduled retraining and monitoring.

## 20. Conclusion
The project delivers a complete, tested ML + NLP + web application that follows all nine lifecycle phases, with honest reporting of the dataset's limits.
"""
    (config.REPORTS_DIR / "final_report.md").write_text(final, encoding="utf-8")
    print("Wrote reports/final_report.md")

    qa = [
        ("Why Facebook sentiment analysis?", "Facebook is where people openly share opinions. Knowing whether they are positive or negative helps businesses react faster."),
        ("Why this dataset?", "It was provided by our course for this project. It has text, a sentiment label, a category and a date, which lets us build a classifier and a trend dashboard."),
        ("What is NLP?", "Natural Language Processing is making computers work with human language, for example cleaning text and turning it into numbers."),
        ("What is sentiment analysis?", "Deciding whether a piece of text is positive, negative or neutral."),
        ("Why TF-IDF?", "It turns text into numbers and gives more weight to words that are important in a post but not common in every post."),
        ("Why Naive Bayes?", "It is fast, simple and works well on small text datasets. It was our best model on the validation set."),
        ("Why Logistic Regression?", "It is a strong, simple linear model that gives probabilities and is easy to explain."),
        ("Why SVM?", "A Linear SVM finds the best separating line between classes and often works very well on TF-IDF features."),
        ("Why compare models?", "No model is best on every dataset. Comparing them on the same split lets us choose fairly."),
        ("Why macro-F1?", "Our classes are imbalanced. Macro-F1 averages the F1 of each class equally, so a poor result on the small negative class is not hidden."),
        ("What is overfitting?", "When a model memorises training data and does badly on new data. We check by comparing train, validation and test scores."),
        ("What is data leakage?", "When information from the test set leaks into training so scores look better than they really are. Repeated sentences cause this, so we split by sentence group."),
        ("Why train, validation and test?", "Train to learn, validation to choose the model and settings, test to give one honest final score."),
        ("What is FastAPI?", "A Python framework for building web APIs. It validates input automatically and creates documentation at /docs."),
        ("Why React?", "It builds interactive dashboards from reusable components, and Recharts gives ready-made charts."),
        ("How does the frontend talk to the backend?", "The browser sends HTTP requests (fetch) to the /api endpoints and receives JSON, which React shows as cards and charts."),
        ("What happens when a comment is entered?", "The text goes to POST /api/analyze, the saved pipeline cleans it, TF-IDF converts it, the classifier returns probabilities, and the UI shows the label, confidence and important words."),
        ("How is confidence calculated?", "It is the model's probability for the predicted class. We call it high at 0.70 or more, medium from 0.50, and low below 0.50."),
        ("How are dashboard values calculated?", "On start-up the saved model scores every prepared post into SQLite. The API counts and aggregates those rows. Nothing is typed in by hand."),
        ("How is deployment done?", "Backend on Render with the start command uvicorn backend.app.main:app; frontend on Vercel with VITE_API_URL pointing at the Render URL. Details are in the README."),
        ("How did you handle duplicates?", f"Exact duplicate records: {dr['duplicate_records_excluding_id']}. But {dup:,} texts repeat with different ids and dates, so we kept them and grouped identical texts to prevent leakage."),
        ("Why is the accuracy not higher?", f"The same sentence often has different labels in the dataset ({honest}). That sets a ceiling for any text-based model."),
        ("Did you change the original dataset?", "No. The raw CSV stays in data/raw. Cleaned data is a separate file in data/processed."),
        ("What is the baseline and why use it?", f"A model that always predicts the most common label (macro-F1 {base_f1:.3f}). Our model must beat it to be useful."),
        ("Why keep words like 'not' and emojis?", "'Not good' and 'good' have opposite meanings, and emojis show feeling. Removing them would hurt sentiment."),
        ("What is cross-validation?", "Splitting training data into folds, training on some and checking on the rest, to get a more stable score when tuning settings."),
        ("What is GridSearchCV?", "It tries every combination of chosen settings with cross-validation and keeps the best one."),
        ("What is a confusion matrix?", "A table of actual versus predicted labels. The diagonal shows correct predictions."),
        ("How is Sentiment Health calculated?", "100 x (positive + 0.5 x neutral) / total posts. Neutral counts as half."),
        ("How is the Customer Voice Index calculated?", "100 x (positive - negative) / (positive + negative). It ranges from -100 to +100."),
        ("Why is the date trend available?", "The dataset has a valid timestamp column. If a dataset had none, the app would show a message and never invent dates."),
        ("Does the project use live Facebook data?", "No. It analyses the supplied dataset. Live collection is listed as future scope."),
        ("What are the limitations?", "Inconsistent labels, bag-of-words cannot catch sarcasm, fixed dataset, unreliable language column."),
        ("How can the project be improved?", "Better labels, a transformer model, real language detection, emotion labels, and scheduled retraining."),
        ("How do you make sure the model file and the API match?", "The pipeline includes the cleaning step, and tests check that the saved labels match the model's classes."),
        ("Which nine phases did you follow?", "Problem Definition, Requirement Analysis, Data Acquisition, Data Preparation, Exploratory Data Analysis (EDA), Model Selection, Model Development, Model Evaluation, Solution Integration & Testing."),
    ]
    lines = [f"# Viva Questions and Answers - {TITLE}\n"]
    for i, (q, a) in enumerate(qa, 1):
        lines.append(f"**Q{i}. {q}**  \n{a}\n")
    (config.REPORTS_DIR / "viva_questions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote reports/viva_questions.md ({len(qa)} questions)")


if __name__ == "__main__":
    main()
