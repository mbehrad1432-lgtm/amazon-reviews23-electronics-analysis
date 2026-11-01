"""Preliminary text-only baseline with product-disjoint split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


def score(y, pred):
    return {'accuracy': round(float(accuracy_score(y, pred)), 4),
            'macro_f1': round(float(f1_score(y, pred, average='macro', zero_division=0)), 4)}


def main():
    path = Path('data/raw/reviews_electronics.jsonl')
    if not path.exists():
        raise FileNotFoundError('Run python -m src.download first')
    df = pd.read_json(path, lines=True)
    needed = {'text', 'rating', 'parent_asin'}
    if not needed.issubset(df):
        raise ValueError(f'Missing columns: {needed - set(df)}')
    df = df.dropna(subset=['rating', 'parent_asin']).copy()
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip().ne('')]
    df = df[df['rating'].isin([1, 2, 3, 4, 5])]
    if len(df) < 50 or df['parent_asin'].nunique() < 5:
        raise ValueError('Insufficient pilot data or distinct products')
    # Remove duplicate text-product-rating tuples before splitting.
    df = df.drop_duplicates(subset=['parent_asin', 'text', 'rating'])
    groups = df['parent_asin'].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, hold_idx = next(splitter.split(df, df['rating'], groups))
    train = df.iloc[train_idx]
    hold = df.iloc[hold_idx]
    # Holdout is provisional validation only; no final test is claimed.
    if train['rating'].nunique() < 2:
        raise ValueError('Training split has fewer than 2 classes')
    vec = TfidfVectorizer(max_features=20000, min_df=2, ngram_range=(1, 2), stop_words='english')
    X_train = vec.fit_transform(train['text'])
    X_hold = vec.transform(hold['text'])
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, train['rating'])
    model = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_train, train['rating'])
    result = {'note': 'Exploratory group-disjoint 70/30 train/validation; NOT final test.',
              'train_rows': len(train), 'validation_rows': len(hold),
              'train_products': int(train['parent_asin'].nunique()),
              'validation_products': int(hold['parent_asin'].nunique()),
              'dummy': score(hold['rating'], dummy.predict(X_hold)),
              'tfidf_logreg': score(hold['rating'], model.predict(X_hold))}
    out = Path('reports/phase1/baseline_metrics.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
