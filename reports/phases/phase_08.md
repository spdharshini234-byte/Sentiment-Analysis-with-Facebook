# Phase 8: Model Evaluation

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Evaluation Report

## Objective
Validate model performance against the project goals.

## Inputs
- Trained models
- Held-out TEST split (used once)

## Implementation
- Test set size: 716 posts the model never saw.
- **Final model (Multinomial Naive Bayes):** accuracy 0.541, macro-F1 0.513, weighted-F1 0.528, macro precision 0.538, macro recall 0.507.
- **Baseline:** accuracy 0.450, macro-F1 0.207. Lift in macro-F1: +0.306.
- **Error analysis:** largest confusions: 142 x actual neutral predicted positive; 77 x actual positive predicted neutral; 46 x actual negative predicted positive.
- **Overfitting / underfitting check:** train macro-F1 0.525, validation 0.530, test 0.513. Small gap: no strong sign of overfitting. Even train macro-F1 is low, so most remaining error comes from noisy/inconsistent labels rather than model capacity.
- **Success criteria:** (1) beats baseline: MET; (2) accuracy is below the label-consistency ceiling of 70.08% (no leakage sign): MET.
- **Honest interpretation:** accuracy near 54% is not high. This is mostly a property of the data (659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08%), not only of the model.

## Tools
- scikit-learn metrics, matplotlib

## Code / files involved
- src/models/evaluate.py
- scripts/evaluate_model.py
- reports/evaluation.json
- notebooks/06_model_evaluation.ipynb

## Output
- Evaluation Report: this document + reports/evaluation.json
- Figures 09 (confusion matrix) and 10 (per-class metrics)

## Evidence
- reports/evaluation.json
- reports/figures/09_confusion_matrix.png
- reports/figures/10_per_class_metrics.png

## Result
Confusion matrix (test):

| Actual \ Predicted | negative | neutral | positive |
|---|---|---|---|
| negative | 50 | 23 | 46 |
| neutral | 29 | 104 | 142 |
| positive | 12 | 77 | 233 |

Per-class results (test):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.549 | 0.420 | 0.476 | 119 |
| neutral | 0.510 | 0.378 | 0.434 | 275 |
| positive | 0.553 | 0.724 | 0.627 | 322 |

## How to explain it during POP
Say: 'We tested on posts the model had never seen. It clearly beats the baseline. The scores are moderate because the dataset itself gives conflicting labels for the same sentence, which we can prove with numbers. We report this honestly rather than hiding it.'
