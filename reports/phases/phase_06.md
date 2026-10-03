# Phase 6: Model Selection

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Selected Model

## Objective
Choose the most suitable AI algorithm.

## Inputs
- Prepared Dataset
- EDA findings

## Implementation
- **Candidates (required by the brief):** TF-IDF + Multinomial Naive Bayes, TF-IDF + Logistic Regression, TF-IDF + Linear SVM.
- **Baseline:** a majority-class dummy classifier (always predicts the most common label).
- **Trade-offs:** Naive Bayes is fast and strong on small text data; Logistic Regression gives calibrated probabilities; Linear SVM is a strong margin classifier but has no native probabilities (we use a softmax of its scores).
- **Selection rule (documented before training):** highest macro-F1 on the VALIDATION split among the three required models; the test split is never used to choose.
- **Selected model:** Multinomial Naive Bayes.

## Tools
- scikit-learn Pipeline, TfidfVectorizer

## Code / files involved
- src/features/tfidf_features.py
- src/models/train.py (`candidate_models`)
- reports/model_comparison.json
- notebooks/05_model_comparison.ipynb

## Output
- Selected Model: Multinomial Naive Bayes
- Comparison table in reports/model_comparison.json

## Evidence
- reports/model_comparison.json
- reports/figures/08_model_comparison.png

## Result
| Model | CV macro-F1 | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|
| Baseline (majority class) | - | 0.207 | 0.450 | 0.207 | 0.279 |
| Multinomial Naive Bayes | 0.496 | 0.530 | 0.541 | 0.513 | 0.528 |
| Logistic Regression | 0.479 | 0.499 | 0.504 | 0.474 | 0.488 |
| Linear SVM | 0.492 | 0.498 | 0.531 | 0.505 | 0.520 |

## How to explain it during POP
Say: 'We compared three standard text classifiers against a do-nothing baseline. We picked the winner using the validation set, not the test set, so the final test is fair.'
