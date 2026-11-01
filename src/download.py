"""Stream bounded pilot subsets from Amazon Reviews'23 gzip JSONL archives."""
import argparse
import gzip
import json
from pathlib import Path
import requests

REVIEWS = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"

META = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"


def stream_gzip_jsonl(url: str, output: Path, limit: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.partial')
    count = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            with gzip.GzipFile(fileobj=response.raw, mode='rb') as gz, tmp.open('w', encoding='utf-8') as out:
                for raw in gz:
                    if not raw.strip():
                        continue
                    obj = json.loads(raw)
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    count += 1
                    if count >= limit:
                        break
        tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-reviews', type=int, default=50000)
    ap.add_argument('--max-metadata', type=int, default=50000)
    ap.add_argument('--reviews-url', default=REVIEWS)
    ap.add_argument('--metadata-url', default=META)
    args = ap.parse_args()
    if args.max_reviews < 1 or args.max_metadata < 1:
        ap.error('limits must be positive')
    for label, url, filename, limit in [
        ('reviews', args.reviews_url, 'reviews_electronics.jsonl', args.max_reviews),
        ('metadata', args.metadata_url, 'metadata_electronics.jsonl', args.max_metadata),
    ]:
        n = stream_gzip_jsonl(url, Path('data/raw') / filename, limit)
        print(f'{label}: saved {n:,} rows to data/raw/{filename}')
    print('NOTE: first-N pilot only; not a random or representative sample.')

if __name__ == '__main__':
    main()
