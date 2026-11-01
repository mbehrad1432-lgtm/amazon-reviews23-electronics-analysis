"""Phase 1 exploratory data analysis for Amazon Reviews'23 / Electronics.

Run from project root: python -m src.eda
Writes figures and tables under reports/phase1/ and figures/phase1/.
No training and no mutation of input data.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path):
    if not path.is_file():
        raise SystemExit(f'Missing input: {path}. Run download/match_metadata first.')
    records = []
    with path.open('r', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON in {path}:{n}: {exc}') from exc
    return pd.DataFrame.from_records(records)


def nonempty(value):
    if value is None:
        return False
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return False


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', type=Path, default=Path('data/raw/reviews_electronics.jsonl'))
    ap.add_argument('--metadata', type=Path, default=Path('data/raw/metadata_electronics.jsonl'))
    ap.add_argument('--output', type=Path, default=Path('reports/phase1'))
    ap.add_argument('--figures', type=Path, default=Path('figures/phase1'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    reviews, meta = read_jsonl(args.reviews), read_jsonl(args.metadata)
    if reviews.empty:
        raise SystemExit('Reviews file is empty.')
    for col in ('rating', 'parent_asin', 'text'):
        if col not in reviews:
            raise SystemExit(f'Reviews missing required column: {col}')
    if 'parent_asin' not in meta:
        raise SystemExit('Metadata missing required column: parent_asin')

    reviews['rating_num'] = pd.to_numeric(reviews['rating'], errors='coerce')
    reviews['text_len'] = reviews['text'].fillna('').astype(str).str.len()
    reviews['word_count'] = reviews['text'].fillna('').astype(str).str.split().str.len()
    reviews['has_review_image'] = reviews['images'].map(nonempty) if 'images' in reviews else False
    reviews['product_key'] = reviews['parent_asin'].astype('string')
    meta['product_key'] = meta['parent_asin'].astype('string')
    unique_meta = meta.drop_duplicates(subset=['product_key'], keep='first').copy()
    metadata_ids = set(unique_meta['product_key'].dropna())
    reviews['has_metadata'] = reviews['product_key'].isin(metadata_ids)
    merged = reviews.merge(unique_meta.drop(columns=['parent_asin'], errors='ignore'), on='product_key', how='left', suffixes=('_review', '_product'), validate='many_to_one')
    # Preserve missingness semantics for nested/list metadata fields.
    price_field = 'price' if 'price' in merged else None
    merged['has_price'] = merged[price_field].map(nonempty) if price_field else False
    merged['has_product_image'] = merged['images_product'].map(nonempty) if 'images_product' in merged else False
    merged['has_review_text'] = merged['text'].map(nonempty)
    merged['has_review_title'] = merged['title_review'].map(nonempty) if 'title_review' in merged else (merged['title'].map(nonempty) if 'title' in merged else False)
    merged['has_timestamp'] = merged['timestamp'].map(nonempty) if 'timestamp' in merged else False
    merged['has_user_id'] = merged['user_id'].map(nonempty) if 'user_id' in merged else False

    dist = reviews['rating_num'].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(7, 4))
    dist.plot(kind='bar')
    plt.title('Review rating distribution (first-N pilot)')
    plt.xlabel('Rating'); plt.ylabel('Reviews')
    save_plot(args.figures / '01_rating_distribution.png')

    plt.figure(figsize=(7, 4))
    plt.hist(reviews['text_len'], bins=35)
    plt.title('Review text length (characters)')
    plt.xlabel('Characters'); plt.ylabel('Reviews')
    save_plot(args.figures / '02_text_length.png')

    rating_lengths = reviews.loc[reviews['rating_num'].between(1, 5)]
    rating_groups = [rating_lengths.loc[rating_lengths['rating_num'].eq(k), 'text_len'].to_numpy() for k in range(1, 6)]
    plt.figure(figsize=(7, 4))
    plt.boxplot(rating_groups, tick_labels=[str(i) for i in range(1, 6)], showfliers=False)
    plt.title('Review length by rating (outliers hidden in plot)')
    plt.xlabel('Rating'); plt.ylabel('Characters')
    save_plot(args.figures / '03_length_by_rating.png')

    top = reviews['product_key'].value_counts().head(15)
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(top)), top.to_numpy())
    plt.xticks(range(len(top)), [str(x)[-6:] for x in top.index], rotation=45, ha='right')
    plt.title('Most reviewed products (last 6 ASIN characters)')
    plt.xlabel('Product ID suffix'); plt.ylabel('Reviews')
    save_plot(args.figures / '04_top_products.png')

    plt.figure(figsize=(6, 4))
    merged['has_metadata'].value_counts().reindex([True, False], fill_value=0).rename(index={True:'Matched',False:'Unmatched'}).plot(kind='bar')
    plt.title('Review-to-metadata match coverage')
    plt.ylabel('Reviews'); plt.xticks(rotation=0)
    save_plot(args.figures / '05_metadata_coverage.png')

    rating_by_match = pd.crosstab(merged['rating_num'], merged['has_metadata'], normalize='columns').reindex(index=[1,2,3,4,5], fill_value=0).fillna(0)
    rating_by_match.columns = ['Matched' if c else 'Unmatched' for c in rating_by_match.columns]
    rating_by_match.plot(kind='bar', figsize=(7, 4))
    plt.title('Rating mix: matched vs unmatched metadata')
    plt.xlabel('Rating'); plt.ylabel('Within-group fraction')
    save_plot(args.figures / '06_rating_by_metadata_coverage.png')

    if 'timestamp' in merged:
        dates = pd.to_datetime(pd.to_numeric(merged['timestamp'], errors='coerce'), unit='ms', errors='coerce', utc=True)
        monthly = dates.dt.to_period('M').value_counts().sort_index()
        if monthly.size:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(range(len(monthly)), monthly.to_numpy())
            ticks = np.linspace(0, len(monthly)-1, min(8,len(monthly)), dtype=int)
            ax.set_xticks(ticks, [str(monthly.index[i]) for i in ticks], rotation=45)
            ax.set_title('Review timestamps by month (UTC)')
            ax.set_ylabel('Reviews')
            save_plot(args.figures / '07_reviews_by_month.png')

    quality = pd.DataFrame({'field': ['review_text','review_title','review_image','timestamp','user_id','matched_metadata','product_price','product_image'],
      'present_count': [int(merged[k].sum()) for k in ['has_review_text','has_review_title','has_review_image','has_timestamp','has_user_id','has_metadata','has_price','has_product_image']]})
    quality['present_pct'] = (100 * quality['present_count'] / len(merged)).round(2)
    quality.to_csv(args.output / 'modality_coverage.csv', index=False)
    by_rating = reviews.groupby('rating_num', dropna=False).agg(reviews=('text_len','size'), median_chars=('text_len','median'), mean_chars=('text_len','mean'), median_words=('word_count','median')).reset_index()
    by_rating.to_csv(args.output / 'rating_text_summary.csv', index=False)
    rating_by_match.to_csv(args.output / 'rating_by_metadata.csv')
    overview = {
        'pilot_sampling': 'first-N reviews, NOT random or representative',
        'review_count': int(len(reviews)),
        'metadata_rows': int(len(meta)),
        'unique_review_products': int(reviews['product_key'].nunique()),
        'matched_reviews': int(merged['has_metadata'].sum()),
        'matched_review_pct': round(100*merged['has_metadata'].mean(), 2),
        'matched_unique_products': int(reviews.loc[reviews['has_metadata'], 'product_key'].nunique()),
        'unique_users': int(reviews['user_id'].nunique()) if 'user_id' in reviews else None,
        'exact_duplicate_reviews': int(reviews.drop(columns=['rating_num','text_len','word_count','has_review_image','product_key','has_metadata']).astype(str).duplicated().sum()),
        'empty_text_count': int((~reviews['text'].map(nonempty)).sum()),
        'review_length_char_median': float(reviews['text_len'].median()),
        'rating_counts': {str(k):int(v) for k,v in dist.items()},
        'cautions': ['Do not generalize a sequential pilot to the full dataset.', 'A non-null images field can still contain an empty list.', 'Product aggregate average_rating and rating_number may leak rating information.', 'Price missingness among matched products may differ from unmatched products.', 'No statistical significance or final evaluation inferred from EDA.']
    }
    (args.output / 'eda_summary.json').write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding='utf-8')
    print('EDA completed; source data unchanged.')
    print(f"Reviews: {len(reviews):,}; matched reviews: {overview['matched_reviews']:,} ({overview['matched_review_pct']}%)")
    print(f'Figures: {args.figures.resolve()}')
    print(f'Tables and summary: {args.output.resolve()}')


if __name__ == '__main__':
    main()
