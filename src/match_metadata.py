"""Find product metadata matching existing Electronics reviews (Amazon Reviews'23).

Run from the project root: python -m src.match_metadata --max-scan 200000
Requires: requests. Keeps a backup of original metadata file.
"""
import argparse
import gzip
import json
from pathlib import Path
import requests

META_URL = ('https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
            'raw/meta_categories/meta_Electronics.jsonl.gz')
REVIEWS_PATH = Path('data/raw/reviews_electronics.jsonl')
METADATA_PATH = Path('data/raw/metadata_electronics.jsonl')


def wanted_ids(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f'Missing reviews: {path}. Run download.py first.')
    ids = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                value = json.loads(line).get('parent_asin')
                if value:
                    ids.add(str(value))
    if not ids:
        raise SystemExit('No parent_asin IDs found in reviews.')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-scan', type=int, default=200000,
                        help='Maximum metadata records to inspect, not download count')
    parser.add_argument('--metadata-url', default=META_URL)
    args = parser.parse_args()
    if args.max_scan < 1:
        parser.error('--max-scan must be >= 1')

    wanted = wanted_ids(REVIEWS_PATH)
    tmp = METADATA_PATH.with_suffix('.jsonl.partial')
    found = set()
    scanned = 0
    try:
        with requests.get(args.metadata_url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw, mode='rb') as source, tmp.open('w', encoding='utf-8') as output:
                for line in source:
                    if not line.strip():
                        continue
                    scanned += 1
                    obj = json.loads(line)
                    product_id = obj.get('parent_asin')
                    if product_id is not None:
                        product_id = str(product_id)
                    if product_id in wanted and product_id not in found:
                        output.write(json.dumps(obj, ensure_ascii=False) + '\n')
                        found.add(product_id)
                    if scanned >= args.max_scan or len(found) == len(wanted):
                        break
        if not found:
            print('No matching products in scanned range; original metadata unchanged.')
            return
        backup = METADATA_PATH.with_name('metadata_electronics.first_n_backup.jsonl')
        if METADATA_PATH.exists() and not backup.exists():
            METADATA_PATH.replace(backup)
            print(f'Previous metadata backed up to {backup}')
        tmp.replace(METADATA_PATH)
        print(f'Scanned {scanned:,} metadata records')
        print(f'Matched {len(found):,} of {len(wanted):,} unique review product IDs')
        print(f'Product coverage: {100 * len(found) / len(wanted):.2f}%')
        print(f'Output: {METADATA_PATH}')
        print('NOTE: limited first-N review sample; not representative.')
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == '__main__':
    main()
