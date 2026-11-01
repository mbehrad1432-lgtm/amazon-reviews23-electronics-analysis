"""Phase 1 modality, missingness, and leakage audit for Amazon Reviews'23 Electronics.

Run from repository root: python -m src.modality_leakage_audit
Read-only for raw data; writes reports/phase1 and figures/phase1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_jsonl(path):
    if not path.is_file():
        raise SystemExit(f'Missing {path}')
    rows = []
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f'Invalid JSON at {path}:{i}: {exc}') from exc
    return pd.DataFrame(rows)


def present(x):
    if x is None or isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return False
    if isinstance(x, str):
        return bool(x.strip())
    if isinstance(x, (list, dict, tuple)):
        return bool(x)
    return True


def availability(df, name):
    return df[name].map(present).astype(bool) if name in df else pd.Series(False, index=df.index, dtype=bool)


def price_ok(value):
    if not present(value):
        return False
    if isinstance(value, str):
        value = value.replace('$','').replace(',','').strip()
    try:
        n = float(value)
        return np.isfinite(n) and n > 0
    except (TypeError, ValueError):
        return False


def rate(count, n):
    return round(100 * int(count) / n, 2) if n else 0.0


def chart(df, path):
    columns = ['Review text','Review images','Product metadata','Product price','Product images']
    vals = [df[k].mean() * 100 for k in ['has_text','has_review_images','has_metadata','has_price','has_product_images']]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(columns[::-1], vals[::-1])
    ax.set_xlim(0, 105)
    ax.set_xlabel('Reviews with available modality (%)')
    ax.set_title('Availability of input modalities (pilot sample)')
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2, f'{b.get_width():.1f}%', va='center')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reviews', type=Path, default=Path('data/raw/reviews_electronics.jsonl'))
    p.add_argument('--metadata', type=Path, default=Path('data/raw/metadata_electronics.jsonl'))
    p.add_argument('--output', type=Path, default=Path('reports/phase1'))
    p.add_argument('--figures', type=Path, default=Path('figures/phase1'))
    a = p.parse_args()
    review = load_jsonl(a.reviews)
    meta = load_jsonl(a.metadata)
    if review.empty:
        raise SystemExit('No reviews found')
    if 'parent_asin' not in review or 'parent_asin' not in meta:
        raise SystemExit('parent_asin missing in reviews or metadata')
    a.output.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    n = len(review)
    review['key'] = review['parent_asin'].astype('string').str.strip()
    meta['key'] = meta['parent_asin'].astype('string').str.strip()
    meta = meta.dropna(subset=['key']).drop_duplicates('key', keep='first')
    meta_cols = [k for k in ('key','price','images','title','description','main_category','average_rating','rating_number') if k in meta]
    merged = review.merge(meta[meta_cols], how='left', on='key', suffixes=('_review','_product'), validate='many_to_one', indicator=True)
    merged['has_metadata'] = merged['_merge'].eq('both')
    merged['has_text'] = availability(merged, 'text')
    merged['has_review_images'] = availability(merged, 'images_review' if 'images_review' in merged else 'images')
    merged['has_product_images'] = merged['has_metadata'] & availability(merged, 'images_product' if 'images_product' in merged else '__missing__')
    merged['has_price'] = merged['has_metadata'] & (merged['price'].map(price_ok) if 'price' in merged else False)
    merged['has_product_description'] = merged['has_metadata'] & availability(merged, 'description')
    merged['has_product_category'] = merged['has_metadata'] & availability(merged, 'main_category')
    merged['has_timestamp'] = availability(merged,'timestamp')
    merged['has_user_id'] = availability(merged,'user_id')
    merged['has_rating'] = availability(merged,'rating')
    merged['len_chars'] = merged['text'].fillna('').astype(str).str.len() if 'text' in merged else 0
    merged['rating_num'] = pd.to_numeric(merged['rating'], errors='coerce') if 'rating' in merged else np.nan
    merged['timestamp_date'] = pd.to_datetime(merged['timestamp'], unit='ms', errors='coerce', utc=True) if 'timestamp' in merged else pd.NaT
    merged['year'] = merged['timestamp_date'].dt.year
    groups = merged['has_metadata'].map({True:'matched',False:'unmatched'})

    modality_cols = ['has_rating','has_text','has_review_images','has_metadata','has_price','has_product_images','has_product_description','has_product_category','has_timestamp','has_user_id']
    modality = pd.DataFrame({'modality':modality_cols,'count':[int(merged[k].sum()) for k in modality_cols]})
    modality['percentage_reviews'] = (modality['count'] / n * 100).round(2)
    modality.to_csv(a.output/'modality_detailed.csv', index=False)

    summary = merged.groupby(groups, observed=True).agg(n=('key','size'), mean_rating=('rating_num','mean'),median_length=('len_chars','median'),mean_length=('len_chars','mean'),five_star_pct=('rating_num',lambda s:round(100 * s.eq(5).mean(),2)),review_image_pct=('has_review_images',lambda s:round(100*s.mean(),2)),year_median=('year','median')).reset_index().rename(columns={'has_metadata':'metadata_group'})
    summary.to_csv(a.output/'matched_vs_unmatched.csv', index=False)
    ratingmix = pd.crosstab(groups, merged['rating_num'], normalize='index').mul(100).round(2)
    ratingmix.to_csv(a.output/'rating_mix_by_match.csv')
    year_table = pd.crosstab(merged['year'], groups)
    year_table.to_csv(a.output/'year_by_metadata_match.csv')

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ['matched','unmatched']:
        if name in ratingmix.index:
            ax.plot(ratingmix.columns.astype(str), ratingmix.loc[name].values, marker='o', label=name)
    ax.set(xlabel='Rating',ylabel='Within-group share (%)',title='Rating distribution by metadata availability')
    ax.legend()
    fig.tight_layout(); fig.savefig(a.figures/'08_rating_match_diagnostic.png', dpi=160); plt.close(fig)
    chart(merged, a.figures/'09_modality_availability.png')

    # Risk and split diagnostics. Does NOT generate a final split or train a model.
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    cut = max(1,min(n-1,int(round(n*.7)))) if n > 1 else 1
    tr, va = merged.iloc[perm[:cut]], merged.iloc[perm[cut:]]
    product_overlap = len(set(tr['key'].dropna()) & set(va['key'].dropna()))
    user_overlap = None
    if 'user_id' in merged:
        user_overlap = len(set(tr['user_id'].dropna()) & set(va['user_id'].dropna()))
    excluded = ['rating (target itself)','average_rating (aggregated label information)','rating_number (aggregate based on rating events)','helpful_vote (post-publication interaction, timing-dependent)']
    stats = {
        'sampling_warning':'First-N sequential pilot, not representative.',
        'reviews':n,'unique_products_in_reviews':int(merged['key'].nunique()),
        'metadata_unique_products':int(meta['key'].nunique()),
        'matched_reviews':int(merged['has_metadata'].sum()),
        'matched_review_pct':rate(merged['has_metadata'].sum(),n),
        'matched_products':int(merged.loc[merged.has_metadata,'key'].nunique()),
        'modality_percentages':{k:rate(merged[k].sum(),n) for k in modality_cols},
        'valid_price_among_matched_pct':rate(merged['has_price'].sum(),int(merged['has_metadata'].sum())),
        'diagnostic_random_row_split_70_30':{'seed':42,'product_id_overlap_count':product_overlap,'user_id_overlap_count':user_overlap,'note':'Diagnostic only; NOT the final evaluation split.'},
        'leakage_exclusion_candidates':excluded,
        'notes':['Missing price/images differ from null versus empty containers.','Aggregate product ratings can encode the target.','Product-disjoint train/validation/test is recommended for generalization to unseen products; inspect user overlap separately.','A group split may have unstable class proportions with small first-N pilot.','Metadata matching is incomplete and not random; compare subgroups before multimodal claims.']
    }
    (a.output/'modality_leakage_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Reviews: {n:,} | joined metadata: {stats["matched_review_pct"]}% | matched products: {stats["matched_products"]}')
    print('Actual image availability:',stats['modality_percentages']['has_review_images'],'% review; ',stats['modality_percentages']['has_product_images'],'% product')
    print('Valid price among matched reviews:',stats['valid_price_among_matched_pct'],'%')
    print('Diagnostic random-split overlap:', product_overlap,'product IDs,', user_overlap,'user IDs')
    print(f'Outputs: {a.output}/modality_leakage_summary.json and CSVs; {a.figures}/08... and 09...')


if __name__ == '__main__':
    main()
