"""Check Phase 1 split manifests against original JSONL review order.
Run from project root: python src/verify_phase1_split.py
Never inspect test prediction performance here.
"""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path.cwd()
RAW = ROOT / 'data/raw/reviews_electronics.jsonl'
SPLITS = ROOT / 'data/processed'
OUT = ROOT / 'reports/phase1/split_verification.json'


def main():
    if not RAW.exists():
        raise FileNotFoundError(f'Missing {RAW}')
    source = pd.read_json(RAW, lines=True).reset_index(drop=True)
    required = {'parent_asin', 'rating'}
    if not required.issubset(source):
        raise ValueError(f'Missing source columns: {required - set(source.columns)}')
    manifests = {}
    for name in ('train', 'validation', 'test'):
        path = SPLITS / f'{name}_manifest.csv'
        df = pd.read_csv(path, dtype={'parent_asin': 'string'})
        expected = {'row_index', 'parent_asin', 'rating'}
        if not expected.issubset(df):
            raise ValueError(f'{path}: missing columns {expected - set(df.columns)}')
        if df['row_index'].isna().any() or df['row_index'].duplicated().any():
            raise ValueError(f'{path}: null or duplicated row index')
        if not df['row_index'].between(0, len(source)-1).all():
            raise ValueError(f'{path}: row_index outside source range')
        original = source.iloc[df['row_index'].to_numpy()]
        if not original['parent_asin'].astype('string').reset_index(drop=True).equals(df['parent_asin'].reset_index(drop=True)):
            raise ValueError(f'{path}: parent_asin differs from original source rows')
        if not pd.to_numeric(original['rating'].reset_index(drop=True)).equals(pd.to_numeric(df['rating'])):
            raise ValueError(f'{path}: rating differs from original source rows')
        manifests[name] = df
    names = list(manifests)
    rows = {n: set(manifests[n]['row_index'].astype(int)) for n in names}
    products = {n: set(manifests[n]['parent_asin'].dropna()) for n in names}
    pairs = [(names[i], names[j]) for i in range(3) for j in range(i+1,3)]
    row_overlap = {f'{a}_{b}': len(rows[a] & rows[b]) for a,b in pairs}
    product_overlap = {f'{a}_{b}': len(products[a] & products[b]) for a,b in pairs}
    complete = set.union(*rows.values()) == set(range(len(source)))
    results = {'source_reviews':len(source), 'split_rows': {n:len(manifests[n]) for n in names},
               'row_overlap':row_overlap, 'product_overlap':product_overlap,
               'every_source_review_assigned_once':complete and all(v == 0 for v in row_overlap.values()),
               'product_disjoint':all(v == 0 for v in product_overlap.values())}
    results['passed'] = results['every_source_review_assigned_once'] and results['product_disjoint']
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results,indent=2))
    if not results['passed']:
        sys.exit('FAIL: split manifests must be reviewed before Phase 1 submission')

if __name__=='__main__':
    main()
