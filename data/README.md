# Data provenance
Dataset: Amazon Reviews’23, McAuley Lab / UCSD, https://amazon-reviews-2023.github.io/ . Category: Electronics. Source records are third-party user reviews and product metadata, observational rather than controlled experimental data.

`src.download` reads gzip JSONL from the upstream source and saves first N entries to `data/raw/`. This is a convenience pilot; first-N ordering may bias the sample. For final experiments use a documented randomized/stratified sample and record sampling seed, inclusion rules and source snapshot.

Expected review keys may include rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase. Metadata may include parent_asin, title, average_rating, rating_number, features, description, price, images, categories. Verify actual columns with audit. Product aggregate ratings are potentially leakage-prone and should not be used naively.

Do not commit raw records or personal identifiers to Git.
