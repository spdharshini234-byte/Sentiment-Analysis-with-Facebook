# Phase 2: Requirement Analysis

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Requirement Specification (SRS)

## Objective
Translate the business need into technical requirements.

## Inputs
- Problem Statement (Phase 1)
- Manus brief

## Implementation
- **Functional requirements:** FR1 executive dashboard; FR2 sentiment pulse; FR3 time-based trend monitor (only when a valid date column exists); FR4 comment analyzer (prediction, confidence, probabilities, important words); FR5 batch CSV analyzer with download; FR6 top customer complaints; FR7 keyword intelligence and keyword x sentiment; FR8 model performance centre with 4-model comparison; FR9 methodology page.
- **Non-functional requirements:** NFR1 every number computed from data; NFR2 model trained once and loaded by the API (not retrained at start-up); NFR3 no paid or external AI service; NFR4 works on Windows and Linux with pathlib paths; NFR5 friendly errors with no stack traces for users; NFR6 reproducible (fixed random_state).
- **Evaluation metrics:** accuracy, precision, recall, F1 per class, macro-F1 (primary, because classes are imbalanced), weighted-F1, confusion matrix.
- **Resources:** one CSV dataset, a normal laptop (no GPU), Python 3.11+.
- **Assumptions:** the `sentiment` column is the ground truth label. **Risks:** 659 of 1,639 unique texts carry more than one label, so text-only accuracy cannot exceed about 70.08%. Mitigation: document it, split by text group, report macro-F1 against a baseline.

## Tools
- Markdown documentation

## Code / files involved
- This file
- README.md (Features)

## Output
- Requirement Specification (SRS) - this document

## Evidence
- This file
- tests/test_api.py covers FR4-FR9 endpoints

## Result
Nine functional and six non-functional requirements, a metric choice and a risk list.

## How to explain it during POP
Say: 'We wrote what the system must do and how well it must do it. We chose macro-F1 because negative posts are only a small share of the data and plain accuracy would hide poor performance on them.'
