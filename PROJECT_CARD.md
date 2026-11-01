# PROJECT CARD — Phase 1 (Draft)

> Track B — Amazon Reviews’23 / Electronics. Hypotheses and scope are provisional until the audit is run.

1. **Track and subset:** Track B; Electronics reviews and product metadata; pilot first 50,000 review records and 50,000 metadata records (sequential pilot, not a representative random sample).
2. **Working title:** Predicting Amazon Electronics Review Ratings from Text and Product Metadata.
3. **Real-world decision:** Estimate a review's likely star rating from the text and available product context; evaluate whether product context adds value.
4. **Unit of analysis / prediction:** One review; predict its 1–5 star rating.
5. **Population:** Amazon Electronics reviews in the released Amazon Reviews’23 corpus; pilot may not represent the whole category.
6. **Primary research question:** Does adding product metadata to review text improve 1–5 rating prediction on held-out reviews/products?
7. **Target:** `rating` (five-class ordinal label); never use rating or derivatives as input.
8. **Baselines:** majority-class DummyClassifier; text-only TF-IDF + logistic regression; metadata-only model; combined model in later phases.
9. **Metrics:** macro-F1 (primary), accuracy, confusion matrix; subgroup performance by review length and product frequency.
10. **Train/validation/test:** Provisional 70/15/15 with product-group disjointness by `parent_asin` for generalization to unseen products; audit feasibility and class coverage before freezing split. Do not tune on test.
11. **Leakage risks:** rating as feature; review text copying the numeric rating; product average rating or rating histogram computed using evaluation reviews; post-review helpful votes; duplicate reviews; user/product overlap; metadata derived from later time periods.
12. **Unsupervised component:** TF-IDF/SVD review-text clustering with cluster interpretation and stability checks in later phases.
13. **Track-specific advanced component:** Compare metadata-only, text-only, and combined modalities; optional sentence embeddings and ablation analysis. Audit availability of images but do not assume images are downloaded.
14. **Ethics, privacy, bias:** Public review text may contain personal information; avoid publishing raw text/user IDs; language and selection bias; extreme-class imbalance; evaluate subgroup differences.

## Three candidate questions
- Q1: How well does review text alone predict star ratings?
- Q2 (selected): Does product metadata improve over review text alone?
- Q3: How does performance change by review length and product popularity?

## Freeze after pilot audit
Confirm sampling strategy, coverage of metadata join, target balance, and exact split; record final decisions here.
