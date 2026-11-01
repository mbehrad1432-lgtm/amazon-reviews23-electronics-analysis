"""Basic schema, missingness, quality and modality coverage audit."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('data/raw')
OUT = Path('reports/phase1')
FIG = Path('figures')


def load(name):
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run python -m src.download first.')
    return pd.read_json(path, lines=True)


def report_frame(df):
    return {'rows': int(len(df)), 'columns': list(df.columns),
            'missing_pct': {str(k): round(float(v), 2) for k, v in (df.isna().mean() * 100).items()},
            'exact_duplicates': int(df.astype(str).duplicated().sum())}


def main():
    reviews = load('reviews_electronics.jsonl')
    meta = load('metadata_electronics.jsonl')
    if 'rating' not in reviews.columns:
        raise ValueError('rating missing: inspect upstream schema')
    result = {'reviews': report_frame(reviews), 'metadata': report_frame(meta),
              'warnings': ['First-N pilot is not representative.',
                           'Do not use target-derived product average ratings as features.']}
    result['rating_distribution'] = {str(k): int(v) for k, v in reviews['rating'].value_counts(dropna=False).sort_index().items()}
    if 'text' in reviews:
        texts = reviews['text'].fillna('').astype(str)
        result['text'] = {'empty_pct': round(float(texts.str.strip().eq('').mean() * 100), 2),
                          'length_chars': {str(k): float(v) for k, v in texts.str.len().describe().items()}}
    for col in ['parent_asin', 'user_id', 'timestamp', 'verified_purchase', 'helpful_vote', 'images']:
        if col in reviews:
            result.setdefault('review_modalities', {})[col] = round(float(reviews[col].notna().mean() * 100), 2)
    if 'parent_asin' in reviews and 'parent_asin' in meta:
        keys = set(meta['parent_asin'].dropna().astype(str))
        result['metadata_join_coverage_pct'] = round(float(reviews['parent_asin'].astype(str).isin(keys).mean() * 100), 2)
        result['unique_products_reviews'] = int(reviews['parent_asin'].nunique())
        result['unique_products_metadata'] = int(meta['parent_asin'].nunique())
    if 'user_id' in reviews:
        result['unique_users'] = int(reviews['user_id'].nunique())
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4))
    reviews['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set(xlabel='Star rating', ylabel='Reviews', title='Electronics pilot rating distribution')
    fig.tight_layout()
    fig.savefig(FIG / 'rating_distribution.png', dpi=160)
    plt.close(fig)
    summary = ['# Phase 1 pilot audit', '', f"Review rows: {len(reviews):,}", f"Metadata rows: {len(meta):,}",
               f"Metadata join coverage: {result.get('metadata_join_coverage_pct', 'N/A')}%", '',
               'See `audit.json` for column-level missingness and distributions.',
               'Warning: sequential pilot; do not generalize descriptive statistics to full corpus.']
    (OUT / 'summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('\n'.join(summary))

if __name__ == '__main__':
    main()
