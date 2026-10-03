# Phase 3: Data Acquisition

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Raw Dataset

## Objective
Collect quality data for model development.

## Inputs
- archive.zip supplied by the student, containing one CSV

## Implementation
- **Source:** the supplied Facebook dataset file `Facebook_data_txt.csv`. No other dataset was used or mixed in.
- **Raw data:** 5,000 rows and 11 columns: post_id, timestamp, text, category, sentiment, language, has_link, likes, comments, shares, is_spam.
- **Privacy and compliance check:** @mentions: 0, e-mail addresses: 0, phone-like numbers: 0, URLs: 0, personal-identifier columns: 0. No personal data was found.
- **Quality assessment:** missing values 0, empty texts 0, exact duplicate records (ignoring id) 0, but only 1,639 unique texts (3,361 repeated texts). 659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08%.
- **Labels:** negative 834 (16.68%), neutral 1,913 (38.26%), positive 2,253 (45.06%).
- **Other observation:** the `language` column lists en, fr, de, es, hi but 0 texts contain non-English letters, so the tag is not reliable.
- The original file is never modified; it stays in data/raw/.

## Tools
- pandas
- scripts/prepare_data.py

## Code / files involved
- src/data/load_data.py (`load_raw`, `inspect_dataset`)
- data/raw/Facebook_data_txt.csv
- reports/data_report.json
- notebooks/01_data_understanding.ipynb

## Output
- Raw Dataset: data/raw/Facebook_data_txt.csv (unchanged)
- Data quality assessment: reports/data_report.json

## Evidence
- reports/data_report.json
- notebooks/01_data_understanding.ipynb

## Result
The raw dataset was loaded, validated and assessed. It is usable, with one documented quality issue (inconsistent labels for repeated texts).

## How to explain it during POP
Say: 'We used only the dataset provided. We checked it for missing values, duplicates and personal data. We found that many sentences repeat with different labels, so we decided to be honest about it and handle it in the split.'
