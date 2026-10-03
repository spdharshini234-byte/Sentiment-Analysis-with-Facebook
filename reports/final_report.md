# Sentiment Analysis with Facebook - Final Report

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
File `Facebook_data_txt.csv`: 5,000 rows, 11 columns (post_id, timestamp, text, category, sentiment, language, has_link, likes, comments, shares, is_spam). Labels: negative 834 (16.68%), neutral 1,913 (38.26%), positive 2,253 (45.06%). Date range 2023-01-01 15:55:00 to 2025-08-08 18:48:00. Unique texts: 1,639; repeated texts: 3,361. 659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08%.

## 7. Data Preprocessing
Cleaning keeps negations, emojis and tone marks, removes URLs, HTML, mentions and stop-words. No rows were dropped (missing 0, invalid labels 0, exact duplicates 0). The original file is unchanged; the prepared data is saved separately.

## 8. EDA
See reports/figures/ and phase_05.md. Key facts: imbalanced classes, strong category effect, inconsistent labels for repeated texts.

## 9. Feature Engineering
TF-IDF with unigrams and bigrams, sub-linear term frequency, min_df tuned (1 or 2), up to 20,000 features.

## 10. Model Selection
| Model | CV macro-F1 | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|
| Baseline (majority class) | - | 0.207 | 0.450 | 0.207 | 0.279 |
| Multinomial Naive Bayes | 0.496 | 0.530 | 0.541 | 0.513 | 0.528 |
| Logistic Regression | 0.479 | 0.499 | 0.504 | 0.474 | 0.488 |
| Linear SVM | 0.492 | 0.498 | 0.531 | 0.505 | 0.520 |

Selected: **Multinomial Naive Bayes** (best validation macro-F1 among the three required models).

## 11. Model Training
GridSearchCV with 5-fold GroupKFold on the training split, scoring macro-F1. Best parameters: `{"clf__alpha": 0.1, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 1]}`. Pipeline saved with joblib.

## 12. Model Evaluation
Test accuracy 0.541, macro-F1 0.513, weighted-F1 0.528; baseline macro-F1 0.207. Small gap: no strong sign of overfitting.

| Actual \ Predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 50 | 23 | 46 |
| neutral | 29 | 104 | 142 |
| positive | 12 | 77 | 233 |

## 13. System Architecture
CSV -> preprocessing -> TF-IDF -> classifier -> saved pipeline -> FastAPI (+ SQLite analytics) -> React dashboard.

## 14. Backend
FastAPI with Pydantic schemas, CORS, graceful errors; the model is loaded once; analytics read from SQLite.

## 15. Frontend
React + Vite + Recharts, nine pages, dark/light themes, empty states, configurable API URL.

## 16. Results
The model beats the baseline by +0.306 macro-F1. Overall dataset sentiment is positive-leaning (45.06% positive). The weakest class is the one with the lowest F1 in the per-class table of phase_08.md.

## 17. Business Use Case
Customer-support and marketing teams can track mood, spot categories and keywords linked to negative sentiment, and triage the most clearly negative comments.

## 18. Limitations
- Labels in the dataset are inconsistent for repeated texts, capping accuracy (about 70.08% upper bound).
- Bag-of-words features cannot understand sarcasm or context.
- The dataset is a fixed file; there is no live data collection.
- The `language` column does not match the actual text language.

## 19. Future Scope
Cleaner or multi-annotator labels; transformer models (e.g. BERT) when better labels exist; real language detection; emotion labels; scheduled retraining and monitoring.

## 20. Conclusion
The project delivers a complete, tested ML + NLP + web application that follows all nine lifecycle phases, with honest reporting of the dataset's limits.
