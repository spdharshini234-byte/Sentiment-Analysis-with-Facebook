# Sentiment Analysis with Facebook

Facebook post sentiment analysis (negative / neutral / positive) with **Machine Learning + NLP**, a **FastAPI** backend and an interactive **React** dashboard. The project follows the nine phases of the *AI Project Lifecycle Guide* (Problem Definition to Solution Integration & Testing) and uses only the Facebook dataset supplied with the project.

---

## 1. Project description
### Problem statement
Organisations receive far more Facebook posts and comments than anyone can read. They need an automatic, explainable way to see the overall mood and which topics and words drive negative or positive opinion.

### Objective
Classify Facebook-style posts into three sentiments using traditional ML (TF-IDF with Multinomial Naive Bayes, Logistic Regression and Linear SVM), choose the best model with a documented metric (macro-F1), and present the results in a dashboard in which **every number is computed from the data** - nothing is hard-coded.

> This project analyses a supplied Facebook dataset. It does **not** collect live Facebook data.

## 2. Features
| Page | What it does |
|---|---|
| Overview | KPI cards (total, positive/negative/neutral counts and %, overall sentiment, sentiment health, customer voice index, model macro-F1), distribution chart, common words, rule-based insights, most negative/positive comments, top concern areas, positive drivers |
| Sentiment Pulse | Count/percentage cards, donut chart, confidence distribution (high/medium/low), sentiment mix by category, labels vs. model predictions |
| Trend Monitor | Sentiment over time with day / week / month grouping and a date-range filter (shows a clear message instead of inventing dates if no date column exists) |
| Comment Analyzer | Type a comment: sentiment, confidence, probability of each class, cleaned text, influential words |
| Batch Analyzer | Upload a CSV, auto-detect the text column, get predictions and download the results as CSV |
| Customer Voice | Top customer complaints (negative comments sorted by model confidence) |
| Keyword Intelligence | Most common words, TF-IDF keywords, sentiment split for each keyword |
| Model Performance | Baseline + 3 models compared, accuracy / precision / recall / F1 / macro-F1 / weighted-F1, confusion matrix, error analysis, overfitting check |
| Project Methodology | The nine lifecycle phases, dataset facts, formulas, limitations |

Extra: dark / light theme, responsive layout, empty-state messages instead of blank charts, friendly error messages (no stack traces for users).

## 3. Dataset information
| Item | Value |
|---|---|
| File | `data/raw/Facebook_data_txt.csv` (from the supplied `archive.zip`) - **never modified** |
| Rows / columns | 5,000 rows, 11 columns |
| Columns | post_id, timestamp, text, category, sentiment, language, has_link, likes, comments, shares, is_spam |
| Labels | negative 834 (16.68%), neutral 1,913 (38.26%), positive 2,253 (45.06%) |
| Date range | 2023-01-01 15:55:00 to 2025-08-08 18:48:00 |
| Missing values | 0 |
| Unique texts | 1,639 (so 3,361 rows repeat an earlier text) |

**Known data-quality issue (reported honestly):** 659 of 1,639 unique texts appear with more than one sentiment label. Even a perfect text-only model could not exceed about **70.08%** accuracy. For this reason the dataset is split by *text group* (identical texts never appear in both training and test), and results are compared with a majority-class baseline. The `language` column is not reliable: no text contains non-English letters.

Cleaning is written to a separate file (`data/processed/prepared_dataset.csv`); the raw file stays identical.

## 4. Technologies used
- **Language:** Python 3.11 or newer (3.12 tested), JavaScript (React)
- **ML / NLP:** scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM, GridSearchCV), pandas, numpy, scipy, joblib
- **Charts for reports:** matplotlib, seaborn
- **Backend:** FastAPI, Uvicorn, Pydantic, SQLite (built into Python)
- **Frontend:** React 18, Vite, Recharts (plain CSS)
- **Testing:** pytest, FastAPI TestClient

## 5. Required Python version and packages
- **Python 3.11 or newer** (the pinned scikit-learn / pandas / numpy need 3.11+). Check with `python --version`.
- Packages are listed in `requirements.txt`: scikit-learn 1.8.0, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, joblib 1.5.3, matplotlib 3.10.8, seaborn 0.13.2, fastapi 0.142.2, uvicorn 0.54.0, pydantic 2.13.5, python-multipart 0.0.32.
- **Node.js is NOT needed to run the app.** The production build of the React UI is already included in `frontend/dist` and is served by the backend. Node.js 18+ is only needed if you want to change the frontend.

## 6. Installation and exact commands (Windows)
Open the project folder in **PowerShell** (or Command Prompt).

**Easiest way**
```powershell
.\setup.bat
.\run.bat
```
Then open **http://127.0.0.1:8000** in your browser.

**Manual way (same result)**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\train_model.py
.\.venv\Scripts\python.exe scripts\build_database.py
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
If PowerShell blocks scripts, you do not need to "activate" the environment: the commands above call `.venv\Scripts\python.exe` directly.

The trained model (`models/`) and database (`data/processed/sentiment.db`) are already included, so training is optional. Run the training command again whenever you want to reproduce the results.

### Linux / macOS / GitHub Codespaces
```bash
./setup.sh
./run.sh        # then open http://127.0.0.1:8000
```

## 7. Everything else you may want to run
| Task | Command (Windows, from the project folder) |
|---|---|
| Train the model (single command) | `.\.venv\Scripts\python.exe scripts\train_model.py` |
| Rebuild the analytics database | `.\.venv\Scripts\python.exe scripts\build_database.py` |
| Regenerate phase reports, final report and viva questions | `.\.venv\Scripts\python.exe scripts\generate_reports.py` |
| Print the evaluation summary | `.\.venv\Scripts\python.exe scripts\evaluate_model.py` |
| Run the automated tests | `.\.venv\Scripts\python.exe -m pip install pytest httpx` then `.\.venv\Scripts\python.exe -m pytest` |
| Frontend development (needs Node 18+) | `cd frontend`, `npm install`, `npm run dev` (UI on http://localhost:5173, backend must run on port 8000) |
| Rebuild the frontend | `cd frontend`, `npm install`, `npm run build` |
| Interactive API docs | http://127.0.0.1:8000/docs |

Training takes about 20-60 seconds on a normal laptop.

## 8. Expected result
After `run.bat` you should see `Uvicorn running on http://127.0.0.1:8000`. In the browser:
- The sidebar shows **"Backend and model ready"** (green dot).
- **Overview** shows **5,000** total posts, **2,253** positive (45.06%), **1,913** neutral (38.26%) and **834** negative (16.68%), overall sentiment POSITIVE. These come from the data.
- **Model Performance** shows the selected model **Multinomial Naive Bayes** with test accuracy **0.541** and macro-F1 **0.513**, against a baseline macro-F1 of 0.207.
- In **Comment Analyzer**, entering *"Having trouble logging into my account, any fix?"* returns **Negative** with high confidence and an explanation sentence.
- `python -m pytest` ends with all tests passing.

## 9. Results (from the shipped training run)
| Model | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Baseline (majority class) | 0.207 | 0.450 | 0.207 |
| Multinomial Naive Bayes | 0.530 | 0.541 | 0.513 |
| Logistic Regression | 0.499 | 0.504 | 0.474 |
| Linear SVM | 0.498 | 0.531 | 0.505 |

Selected by highest **validation** macro-F1 among the three required models: **Multinomial Naive Bayes**. Small gap: no strong sign of overfitting. Moderate scores are explained mainly by the inconsistent labels described above (see `reports/phases/phase_08.md`).

## 10. Project phases (from the Phase PDF)
| # | Phase | Deliverable | Report |
|---|---|---|---|
| 1 | Problem Definition | Problem Statement | `reports/phases/phase_01.md` |
| 2 | Requirement Analysis | Requirement Specification (SRS) | `reports/phases/phase_02.md` |
| 3 | Data Acquisition | Raw Dataset | `reports/phases/phase_03.md` |
| 4 | Data Preparation | Prepared Dataset | `reports/phases/phase_04.md` |
| 5 | Exploratory Data Analysis (EDA) | EDA Report | `reports/phases/phase_05.md` |
| 6 | Model Selection | Selected Model | `reports/phases/phase_06.md` |
| 7 | Model Development | Trained Model | `reports/phases/phase_07.md` |
| 8 | Model Evaluation | Evaluation Report | `reports/phases/phase_08.md` |
| 9 | Solution Integration & Testing | Working AI Solution | `reports/phases/phase_09.md` |

Also: `reports/final_report.md` (POP report), `reports/viva_questions.md` (36 questions and answers), `reports/figures/` (10 EDA/evaluation charts), `notebooks/` (7 executed notebooks).

## 11. Architecture and ML workflow
```
Facebook_data_txt.csv (unchanged)
   -> load + schema validation          src/data/load_data.py
   -> cleaning + leakage-free split     src/data/preprocess.py, src/models/train.py
   -> TF-IDF (word 1-2 grams)           src/features/tfidf_features.py
   -> NB / Logistic Regression / SVM    GridSearchCV + GroupKFold, macro-F1
   -> select on validation macro-F1     models/sentiment_pipeline.pkl (cleaning + TF-IDF + classifier)
   -> score all posts into SQLite       scripts/build_database.py
   -> FastAPI (loads the saved model)   backend/app/
   -> React dashboard                   frontend/
```
Training is a separate command; the API never retrains on start-up.

## 12. Project folder structure
```
sentiment-analysis-with-facebook/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_analysis.py
│   │   │   ├── routes_batch.py
│   │   │   ├── routes_dashboard.py
│   │   │   └── routes_prediction.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   └── init_db.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── prediction.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── analytics_service.py
│   │   │   ├── insight_service.py
│   │   │   ├── keyword_service.py
│   │   │   └── prediction_service.py
│   │   ├── __init__.py
│   │   └── main.py
│   ├── __init__.py
│   └── requirements.txt
├── data/
│   ├── processed/
│   │   ├── prepared_dataset.csv
│   │   ├── README.md
│   │   └── sentiment.db
│   └── raw/
│       ├── Facebook_data_txt.csv
│       └── README.md
├── docs/
│   └── screenshots/
│       └── README.md
├── frontend/
│   ├── dist/
│   │   ├── assets/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── BatchAnalyzer.jsx
│   │   │   ├── CommentAnalyzer.jsx
│   │   │   ├── InsightPanel.jsx
│   │   │   ├── KeywordChart.jsx
│   │   │   ├── MetricCard.jsx
│   │   │   ├── SentimentChart.jsx
│   │   │   ├── State.jsx
│   │   │   └── TrendChart.jsx
│   │   ├── pages/
│   │   │   ├── Analyzer.jsx
│   │   │   ├── Batch.jsx
│   │   │   ├── CustomerVoice.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Keywords.jsx
│   │   │   ├── Methodology.jsx
│   │   │   ├── ModelPerformance.jsx
│   │   │   ├── Pulse.jsx
│   │   │   └── Trend.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   ├── main.jsx
│   │   └── theme.js
│   ├── .env.example
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── vercel.json
│   └── vite.config.js
├── models/
│   ├── label_mapping.json
│   ├── model_metadata.json
│   └── sentiment_pipeline.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_final_analysis.ipynb
├── reports/
│   ├── figures/
│   │   ├── 01_class_distribution.png
│   │   ├── 02_text_length.png
│   │   ├── 03_top_words_by_sentiment.png
│   │   ├── 04_sentiment_by_category.png
│   │   ├── 05_monthly_trend.png
│   │   ├── 06_sentiment_by_language.png
│   │   ├── 07_label_consistency.png
│   │   ├── 08_model_comparison.png
│   │   ├── 09_confusion_matrix.png
│   │   └── 10_per_class_metrics.png
│   ├── phases/
│   │   ├── phase_01.md
│   │   ├── phase_02.md
│   │   ├── phase_03.md
│   │   ├── phase_04.md
│   │   ├── phase_05.md
│   │   ├── phase_06.md
│   │   ├── phase_07.md
│   │   ├── phase_08.md
│   │   └── phase_09.md
│   ├── data_report.json
│   ├── evaluation.json
│   ├── final_report.md
│   ├── model_comparison.json
│   └── viva_questions.md
├── scripts/
│   ├── build_database.py
│   ├── evaluate_model.py
│   ├── generate_reports.py
│   ├── make_notebooks.py
│   ├── prepare_data.py
│   └── train_model.py
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── figures.py
│   │   ├── insights.py
│   │   ├── keywords.py
│   │   └── sentiment.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── tfidf_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── __init__.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_analytics.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── .gitignore
├── LICENSE
├── render.yaml
├── requirements-dev.txt
├── requirements.txt
├── run.bat
├── run.sh
├── setup.bat
└── setup.sh
```

## 13. API documentation
Base URL `http://127.0.0.1:8000`. All responses are JSON; errors look like `{"error": "message"}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model / database status |
| POST | `/api/predict` | `{"text": "..."}` -> sentiment, confidence, probabilities |
| POST | `/api/analyze` | Same plus influential words and confidence level |
| POST | `/api/batch-predict` | multipart CSV upload (`file`) -> predictions for every row |
| GET | `/api/dashboard` | All numbers for the Overview page |
| GET | `/api/sentiment-distribution` | Counts, %, health score, voice index, by category, confidence |
| GET | `/api/sentiment-trend?granularity=month&start=YYYY-MM-DD&end=YYYY-MM-DD` | Time series |
| GET | `/api/common-words?limit=15&sentiment=negative` | Frequent words + TF-IDF keywords |
| GET | `/api/keyword-sentiment` | Sentiment split per keyword |
| GET | `/api/negative-feedback`, `/api/positive-feedback` | Top comments |
| GET | `/api/model-performance` | Model comparison and metrics |
| GET | `/api/dataset-info` | Dataset report |

Formulas: **Sentiment health** = 100 x (positive + 0.5 x neutral) / total. **Customer voice index** = 100 x (positive - negative) / (positive + negative). **Confidence** = probability of the predicted class (high >= 0.70, medium >= 0.50, low < 0.50).

## 14. Deployment (optional)
### GitHub
```bash
git init
git add .
git commit -m "Sentiment Analysis with Facebook"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
### Backend on Render
1. Render dashboard -> **New** -> **Blueprint** (uses `render.yaml`) or **Web Service** from your repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
4. Environment variables: `PYTHON_VERSION=3.12.3`, optional `CORS_ORIGINS=https://your-frontend.vercel.app`.
5. Check `https://<your-service>.onrender.com/api/health`.

### Frontend on Vercel
1. Vercel -> **Add New Project** -> import the repo, set **Root Directory** to `frontend`.
2. Build command `npm run build`, output directory `dist`.
3. Environment variable **`VITE_API_URL`** = your Render URL (no trailing slash).

If you deploy only on Render, the backend already serves the UI, so Vercel is optional.

## 15. Screenshots
Save screenshots of the nine pages in `docs/screenshots/` for your presentation.

## 16. Limitations
- Labels are inconsistent for repeated texts, which limits accuracy to about 70.08% for any text-only model.
- TF-IDF cannot understand sarcasm or context.
- The dataset is a fixed file (no live data); the `language` column is unreliable; there is no emotion or platform field.

## 17. Future enhancements
Cleaner or multi-annotator labels, transformer models, real language detection, emotion labels, scheduled retraining.

## 18. Troubleshooting (Windows)
| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python 3.11+ from python.org and tick "Add python.exe to PATH", then reopen the terminal |
| "Model could not be loaded" / version warning | Run `.\.venv\Scripts\python.exe scripts\train_model.py` then `scripts\build_database.py` |
| Port 8000 already used | Add `--port 8001` to the uvicorn command and open http://127.0.0.1:8001 |
| Blank page | Make sure `frontend/dist` exists (it is included); otherwise run `npm install` and `npm run build` in `frontend` |

## 19. Viva
See `reports/viva_questions.md`.
