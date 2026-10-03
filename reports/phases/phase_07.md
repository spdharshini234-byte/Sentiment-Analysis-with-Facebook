# Phase 7: Model Development

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Trained Model

## Objective
Train and optimize the AI model.

## Inputs
- Training split
- Selected candidates

## Implementation
- Each model was tuned with GridSearchCV (scoring = macro-F1) using 5-fold **GroupKFold** so identical texts never straddle folds. Tuned: TF-IDF n-gram range (1,1) or (1,2), min_df 1 or 2, and the classifier parameter (alpha for NB, C for LR and SVM).
- Class weights are balanced for Logistic Regression and Linear SVM to help the minority class.
- **Best parameters of the selected model:** `{"clf__alpha": 0.1, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 1]}`.
- The full pipeline (cleaning -> TF-IDF -> classifier) is saved as one joblib file, so the API applies exactly the same preprocessing.
- Training is a separate command (`python scripts/train_model.py`); the API only loads `sentiment_pipeline.pkl`.

## Tools
- scikit-learn, joblib

## Code / files involved
- src/models/train.py
- scripts/train_model.py
- models/sentiment_pipeline.pkl
- models/model_metadata.json
- models/label_mapping.json

## Output
- Trained Model: models/sentiment_pipeline.pkl
- Trained at (UTC): 2026-10-03T10:56:48+00:00

## Evidence
- models/model_metadata.json (parameters, library versions, metrics)
- notebooks/04_feature_engineering.ipynb

## Result
Trained model saved. Python 3.12.3, scikit-learn 1.8.0.

## How to explain it during POP
Say: 'We tried several settings for each model with cross-validation and kept the best. Everything - cleaning, TF-IDF and the classifier - is saved as one file so the website uses exactly what we trained.'
