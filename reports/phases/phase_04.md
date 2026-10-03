# Phase 4: Data Preparation

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Prepared Dataset

## Objective
Prepare clean data for machine learning.

## Inputs
- Raw Dataset (Phase 3)

## Implementation
- **Cleaning (src/data/preprocess.py):** HTML removal, URLs removed, @mentions removed, hashtags keep their word, lower-casing, numbers replaced by a `num` token, emojis kept as tokens, `!` and `?` kept as tone tokens, extra whitespace removed, stop-words removed but **negations are kept** (not, never, no) so that sentiment is not destroyed.
- **Missing values and duplicates:** rows dropped for missing/empty text: 0; invalid labels: 0; exact duplicate records: 0; empty after cleaning: 0. Final rows: 5,000.
- **Labelling:** not required; the dataset already has sentiment labels. No label was changed.
- **Leakage control:** after cleaning there are 1,345 unique cleaned texts. Identical texts are grouped and a group is never allowed to appear in two splits (StratifiedGroupKFold, random_state=42).
- **Split (70 / 15 / 15):** see table below.

## Tools
- pandas, regular expressions, scikit-learn StratifiedGroupKFold

## Code / files involved
- src/data/preprocess.py
- src/models/train.py (`grouped_stratified_split`)
- data/processed/prepared_dataset.csv
- tests/test_preprocessing.py
- notebooks/02_data_preparation.ipynb

## Output
- Prepared Dataset: data/processed/prepared_dataset.csv (includes a `split` column)
- Split summary in reports/data_report.json

## Evidence
- tests/test_preprocessing.py includes a leakage test
- passed (no text group appears in two splits)

## Result
The prepared dataset was written to a separate file; the original is untouched.

| Split | Rows | Unique text groups | Negative | Neutral | Positive |
|---|---|---|---|---|---|
| Train | 3567 | 962 | 596 | 1363 | 1608 |
| Validation | 717 | 188 | 119 | 275 | 323 |
| Test | 716 | 195 | 119 | 275 | 322 |

## How to explain it during POP
Say: 'We cleaned the text carefully and kept words like "not". Because the same sentence appears many times, a normal random split would let the model see test sentences during training, so we split by sentence group to avoid that cheating.'
