# Raw data

`Facebook_data_txt.csv` is the dataset supplied with the project (from `archive.zip`). **It is never modified.**
All cleaning is written to a separate file: `data/processed/prepared_dataset.csv`.

Columns (11): `post_id, timestamp, text, category, sentiment, language, has_link, likes, comments, shares, is_spam`.

## If the file is missing
1. Extract `archive.zip` (the dataset ZIP you were given).
2. Copy `Facebook_data_txt.csv` into this folder (`data/raw/`).
3. Run `python scripts/train_model.py`.

You can also keep the file elsewhere and set the environment variable `FSI_DATA_PATH` to its full path.
No Kaggle credentials or API tokens are needed or stored in this project.
