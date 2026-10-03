# Phase 5: Exploratory Data Analysis (EDA)

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** EDA Report

## Objective
Understand the characteristics of the dataset.

## Inputs
- Prepared Dataset (Phase 4)

## Implementation
- Analysed class distribution, text length, word frequency per sentiment, sentiment by category, sentiment by language tag, monthly trend and label consistency.
- **Class imbalance:** negative is the minority (16.68%).
- **Text length:** mean 44.99 characters (min 22, max 86); short, post-style sentences.
- **Date range:** 2023-01-01 15:55:00 to 2025-08-08 18:48:00 (all timestamps valid, so the trend chart is available).
- **Category matters:** sentiment differs strongly by `category` (for example customer_support posts are mostly negative, meme_joke posts mostly positive) - see figure 04.
- **Label consistency:** 659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08% - see figure 07.
- **Outliers:** text length has no extreme outliers; no outlier removal was needed.

## Tools
- pandas, matplotlib, seaborn

## Code / files involved
- src/analysis/figures.py
- reports/figures/*.png
- notebooks/03_eda.ipynb

## Output
- EDA Report: this document + 10 figures in reports/figures/

## Evidence
- reports/figures/01_class_distribution.png
- 02_text_length.png
- 03_top_words_by_sentiment.png
- 04_sentiment_by_category.png
- 05_monthly_trend.png
- 06_sentiment_by_language.png
- 07_label_consistency.png

## Result
EDA showed three key facts: imbalanced classes, strong category effect, and inconsistent labels for repeated texts.

## How to explain it during POP
Say: 'EDA means looking at the data before modelling. We saw that negative posts are rare, that customer-support posts are mostly negative, and that identical sentences sometimes have different labels - which limits any model.'
