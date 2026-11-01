"""Create reproducible product-disjoint train/validation/test splits.
Run from the repository root: python -m src.create_phase1_split
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
INPUT = Path('data/raw/reviews_electronics.jsonl')
OUT = Path('data/processed')
REPORT = Path('reports/phase1')


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing {INPUT}; run the downloader first.')
    df = pd.read_json(INPUT, lines=True)
    needed = {'parent_asin', 'rating'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Missing required columns: {needed - set(df.columns)}')
    if df['parent_asin'].isna().any():
        raise ValueError('Missing parent_asin values; audit and resolve before splitting.')
    if df['parent_asin'].nunique() < 10:
        raise ValueError('Too few unique products for reliable group splits.')
    df = df.reset_index(drop=True)
    idx = df.index.to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    train_i, rest_i = next(splitter.split(idx, groups=df['parent_asin']))
    rest = df.iloc[rest_i]
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED + 1)
    val_relative, test_relative = next(second.split(rest, groups=rest['parent_asin']))
    partitions = {'train': train_i, 'validation': rest_i[val_relative], 'test': rest_i[test_relative]}
    groups = {key: set(df.iloc[ix]['parent_asin']) for key, ix in partitions.items()}
    assert all(not groups[a].intersection(groups[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {}
    for label, indices in partitions.items():
        sub = df.iloc[indices].copy()
        # Row index references the original JSONL line. No text or private IDs in split manifest.
        pd.DataFrame({'row_index': indices, 'parent_asin': sub['parent_asin'].to_numpy(),
                      'rating': sub['rating'].to_numpy()}).to_csv(OUT / f'{label}_manifest.csv', index=False)
        result[label] = {'rows': len(sub), 'products': sub['parent_asin'].nunique(),
                         'ratings': {str(k): int(v) for k, v in sub['rating'].value_counts().sort_index().items()}}
    user_overlap = {}
    if 'user_id' in df.columns:
        users = {k: set(df.iloc[v]['user_id'].dropna()) for k, v in partitions.items()}
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            user_overlap[f'{a}_{b}'] = len(users[a].intersection(users[b]))
    result['method'] = 'GroupShuffleSplit, product-disjoint by parent_asin, approx 70/15/15'
    result['seed'] = SEED
    result['user_overlap_count'] = user_overlap
    result['warnings'] = ['Ratios are approximate because products are grouped.',
                          'This first-N pilot is not representative.',
                          'Keep test untouched for final evaluation. Do not tune on test.',
                          'Fit vectorizers and preprocessors only on train.']
    (REPORT / 'split_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
