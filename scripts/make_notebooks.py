"""Create (and optionally run) the seven analysis notebooks:  python scripts/make_notebooks.py [--run]"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = '''import sys, json
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
from src import config
from src.data.load_data import load_raw, inspect_dataset
from src.data.preprocess import prepare_dataframe, clean_text'''

NOTEBOOKS = {
"01_data_understanding": ("Phase 3 - Data Acquisition: understanding the raw dataset", [
 ("md", "Load the **unchanged** raw CSV and check shape, columns, missing values, duplicates, labels and label consistency."),
 ("code", SETUP),
 ("code", "raw = load_raw()\nprint(raw.shape)\nraw.head()"),
 ("code", "rep = inspect_dataset(raw, config.find_dataset())\nprint('rows:', rep['rows'], '| unique texts:', rep['unique_texts'], '| duplicate texts:', rep['duplicate_texts'])\nprint('missing values:', rep['missing_values_total'], '| empty texts:', rep['empty_text'])\npd.DataFrame(rep['columns'])"),
 ("code", "print('class distribution:', rep['class_distribution'])\nprint('privacy check:', rep['privacy_check'])"),
 ("code", "print(json.dumps(rep['label_consistency'], indent=2))"),
 ("md", "**Finding:** many identical texts carry different labels, which limits the best possible accuracy of any text-only model."),
]),
"02_data_preparation": ("Phase 4 - Data Preparation", [
 ("md", "Clean the text, check duplicates, and create the leakage-free train / validation / test split."),
 ("code", SETUP + "\nfrom src.models.train import grouped_stratified_split"),
 ("code", "raw = load_raw()\ndf, prep = prepare_dataframe(raw)\nprep"),
 ("code", "for t in raw['text'].head(5):\n    print(repr(t), '->', repr(clean_text(t)))"),
 ("code", "tr, va, te = grouped_stratified_split(df)\ng = df['text_group'].values\nprint('overlap train/test groups:', len(set(g[tr]) & set(g[te])))\nprint('overlap train/val groups :', len(set(g[tr]) & set(g[va])))\nfor name, idx in [('train', tr), ('validation', va), ('test', te)]:\n    print(name, len(idx), df.iloc[idx]['sentiment'].value_counts().to_dict())"),
]),
"03_eda": ("Phase 5 - Exploratory Data Analysis", [
 ("md", "Distributions, text length, sentiment by category, trend and label consistency."),
 ("code", SETUP + "\nprep_df = pd.read_csv(config.DATA_PROCESSED / 'prepared_dataset.csv', parse_dates=['timestamp'])"),
 ("code", "prep_df['sentiment'].value_counts().reindex(config.LABELS).plot(kind='bar', color=['#e5484d','#8b95a7','#30a46c'], title='Class distribution')\nplt.show()"),
 ("code", "prep_df['n_chars'] = prep_df['text'].str.len()\nsns.boxplot(data=prep_df, x='sentiment', y='n_chars', order=config.LABELS)\nplt.title('Text length by sentiment'); plt.show()"),
 ("code", "ct = pd.crosstab(prep_df['category'], prep_df['sentiment'], normalize='index')[config.LABELS] * 100\nct.sort_values('negative').plot(kind='barh', stacked=True, color=['#e5484d','#8b95a7','#30a46c'], figsize=(8,5))\nplt.title('Sentiment mix by category (%)'); plt.show()"),
 ("code", "m = prep_df.assign(month=prep_df['timestamp'].dt.to_period('M').dt.start_time).pivot_table(index='month', columns='sentiment', values='text', aggfunc='count', fill_value=0)[config.LABELS]\nm.plot(figsize=(9,3.5), color=['#e5484d','#8b95a7','#30a46c'], title='Monthly sentiment trend'); plt.show()"),
 ("code", "from src.analysis.keywords import common_words\nfor lab in config.LABELS:\n    print(lab, [w['word'] for w in common_words(prep_df, 8, lab)])"),
]),
"04_feature_engineering": ("Phase 7 - Feature engineering with TF-IDF", [
 ("md", "Build TF-IDF features on the training split only and inspect the vocabulary."),
 ("code", SETUP + "\nfrom src.models.train import grouped_stratified_split\nfrom src.features.tfidf_features import build_vectorizer"),
 ("code", "df, _ = prepare_dataframe(load_raw())\ntr, va, te = grouped_stratified_split(df)\nvec = build_vectorizer()\nX = vec.fit_transform(df.iloc[tr]['clean_text'])\nprint('train matrix:', X.shape, '| non-zero:', X.nnz)"),
 ("code", "names = vec.get_feature_names_out()\nscores = np.asarray(X.mean(axis=0)).ravel()\npd.DataFrame({'feature': names, 'mean_tfidf': scores}).sort_values('mean_tfidf', ascending=False).head(15)"),
]),
"05_model_comparison": ("Phase 6 - Model Selection", [
 ("md", "Read the saved comparison of the baseline and three candidate models (training is done by `python scripts/train_model.py`)."),
 ("code", SETUP),
 ("code", "ev = json.loads(config.EVALUATION_PATH.read_text(encoding='utf-8'))\nrows = [{'model': r['model'], 'cv_f1_macro': r['cv_f1_macro_mean'], 'val_f1_macro': r['validation']['f1_macro'], 'test_accuracy': r['test']['accuracy'], 'test_f1_macro': r['test']['f1_macro']} for r in ev['models']]\ndf = pd.DataFrame(rows); print('Selected:', ev['selected_model'], '-', ev['selection_rule']); df.round(3)"),
 ("code", "df.set_index('model')[['val_f1_macro','test_f1_macro']].plot(kind='bar', figsize=(8,4), rot=15, title='Macro-F1 by model'); plt.show()"),
]),
"06_model_evaluation": ("Phase 8 - Model Evaluation", [
 ("md", "Evaluate the saved pipeline on the held-out test split."),
 ("code", SETUP + "\nfrom src.models.predict import SentimentPredictor\nfrom src.models.evaluate import compute_metrics\nfrom sklearn.metrics import classification_report"),
 ("code", "df = pd.read_csv(config.DATA_PROCESSED / 'prepared_dataset.csv')\ntest = df[df['split'] == 'test']\npred = SentimentPredictor()\nyhat = [p['sentiment'] for p in pred.predict_many(test['text'].tolist())]\nprint(classification_report(test['sentiment'], yhat, digits=3))"),
 ("code", "m = compute_metrics(test['sentiment'], yhat)\nsns.heatmap(m['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=config.LABELS, yticklabels=config.LABELS)\nplt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion matrix (test)'); plt.show()"),
 ("code", "ev = json.loads(config.EVALUATION_PATH.read_text(encoding='utf-8'))\nprint(ev['overfitting_check']['verdict'])\nprint(ev['success_criteria'])"),
]),
"07_final_analysis": ("Final analysis - dashboard numbers and insights", [
 ("md", "Compute the same analytics the dashboard shows, directly from the database."),
 ("code", SETUP + "\nfrom backend.app.database.database import load_posts_df\nfrom src.analysis import sentiment as sa, keywords as kw\nfrom src.analysis.insights import generate_insights"),
 ("code", "posts = load_posts_df()\nd = sa.distribution(posts)\nprint(d)\nprint('health:', sa.health_score(d['counts'], d['total']), '| voice index:', sa.voice_index(d['counts']))"),
 ("code", "pd.DataFrame(kw.keyword_sentiment(posts, top_n=10, min_count=30))[['keyword','count','positive_pct','neutral_pct','negative_pct']]"),
 ("code", "ev = json.loads(config.EVALUATION_PATH.read_text(encoding='utf-8'))\nrep = json.loads(config.DATA_REPORT_PATH.read_text(encoding='utf-8'))\nfor i in generate_insights(posts, ev, rep):\n    print('-', i['text'])"),
]),
}

def build():
    NB.mkdir(exist_ok=True)
    for name, (title, cells) in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\nProject: **Sentiment Analysis with Facebook**"))
        for kind, src in cells:
            nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbf.write(nb, NB / f"{name}.ipynb")
    print(f"Created {len(NOTEBOOKS)} notebooks")

def run():
    from nbclient import NotebookClient
    for path in sorted(NB.glob("*.ipynb")):
        nb = nbf.read(path, as_version=4)
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, path)
        print("executed", path.name)

if __name__ == "__main__":
    build()
    if "--run" in sys.argv:
        run()
