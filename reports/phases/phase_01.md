# Phase 1: Problem Definition

**Project:** Sentiment Analysis with Facebook  
**Deliverable (from the Phase PDF):** Problem Statement

## Objective
Understand the business problem and define the project objectives.

## Inputs
- Project title: Sentiment Analysis with Facebook
- Manus project brief (Manus.mht)
- Phase PDF: AI Project Lifecycle Guide

## Implementation
- **Problem:** organisations receive large volumes of Facebook posts and comments and cannot read them all. They need an automatic way to see whether opinion is positive, neutral or negative, and which topics drive it.
- **Objective:** build a system that classifies Facebook-style posts into negative / neutral / positive and shows the results in an interactive dashboard.
- **Measurable success criteria:** (1) the final model must beat a majority-class baseline on test macro-F1 (baseline = 0.207); (2) all dashboard numbers must be computed from the dataset, never hard-coded; (3) the full pipeline must run from raw CSV to dashboard with documented commands.
- **Stakeholders:** social-media / customer-support teams (users), the student team (developers), the college POP evaluators.
- **Scope:** classify the supplied dataset and new comments entered by a user. **Out of scope:** live Facebook data collection, paid AI APIs.
- **Constraints:** one provided dataset only, traditional ML (TF-IDF + NB / LR / SVM), must run on Windows and Linux.

## Tools
- Python, scikit-learn
- FastAPI, React

## Code / files involved
- README.md (Problem statement, Objectives)
- reports/final_report.md (Introduction, Problem Statement, Objectives)

## Output
- Problem Statement (this document)

## Evidence
- This file
- Success criteria are checked automatically in reports/evaluation.json (`success_criteria`)

## Result
The problem and the measurable success criteria are defined before any code. Criterion 1 is evaluated in Phase 8.

## How to explain it during POP
Say: 'Companies cannot read thousands of posts, so we built a tool that labels each post as positive, neutral or negative and summarises the mood. We decided in advance how to judge success: our model must beat a model that always guesses the most common label.'
