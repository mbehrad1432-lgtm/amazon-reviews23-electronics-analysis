# PROJECT CARD — Phase 1 (Updated from actual pilot results)

**Project:** Amazon Reviews’23 — Electronics  
**Track:** B (text + structured product metadata)  
**Status:** Phase 1 pilot audited; research design provisionally fixed for the pilot, **not** a final representative dataset or final benchmark.  
**Seed for current group split:** `42`

## Required 14 fields

1. **Track and selected subset.** Track B, `Electronics` review JSONL and product metadata. The **current** review dataset contains the **first 1,000 records** of the Electronics review archive (sequential first-N, not random or representative). Metadata was subsequently searched for matching `parent_asin` values; current linked metadata has **275 unique products**. A sampling redesign is needed before claims about the whole Electronics category.

2. **Working title.** *Does Product Metadata Improve Star-Rating Prediction from Amazon Electronics Review Text?*

3. **Real-world decision supported.** Assess whether review text and product context can provide useful automatic indications of the associated 1–5 star rating, and specifically whether product information improves on a text-only method. This is an **analytical comparison**, not a claim that ratings can be inferred reliably in deployment.

4. **Unit of analysis and prediction.** One review (one row), with a product link through `parent_asin`; prediction is one 1–5 star label for that review. Multiple reviews may come from one user or concern one product.

5. **Target population and scope.** Electronics reviews within Amazon Reviews’23. Observed statistics and model results currently describe the sequential **1,000-review pilot only**. The pilot contains **959 unique reviewed products** and **195 unique users**.

6. **Primary research question.** **Q2 (selected):** On held-out products, does combining **eligible, non-leaking product metadata** with review text improve five-class star-rating prediction compared with a text-only model? For a fair modality comparison, train/evaluate text-only, metadata-only and combined methods on the **same matched subset**, and also report text-only performance on the wider review sample separately.

7. **Prediction target.** `rating`, with the five possible classes 1, 2, 3, 4 and 5. Pilot class counts: **1: 90; 2: 35; 3: 65; 4: 145; 5: 665**. Do not use `rating` or target-derived aggregates as inputs. The ordinal nature of labels may be considered in later error analysis.

8. **Initial baseline(s).** Majority-class `DummyClassifier` and text-only TF-IDF + Logistic Regression have been run on an **earlier exploratory 702/298 group-separated train/validation split**, not the subsequently created 70/15/15 split. Exploratory results: Dummy accuracy **0.6745**, macro-F1 **0.1611**; TF-IDF + Logistic Regression accuracy **0.6477**, macro-F1 **0.3936**. These are **pilot diagnostic values**, not final or directly comparable to future evaluation on the newer manifest. A structured metadata-only baseline and a combined baseline remain **planned, not completed**.

9. **Evaluation metrics.** **Macro-F1 (primary)** because of class imbalance; accuracy and class-wise precision/recall/F1 and confusion matrix as secondary diagnostics. For later evaluation examine performance by review length, product-frequency grouping and metadata availability. Compare modalities using identical split membership and matched rows.

10. **Training / validation / test scheme.** Current manifest uses `GroupShuffleSplit` on `parent_asin`, seed **42**, approximately 70/15/15 by review count: **train 702 reviews / 671 products; validation 150 reviews / 144 products; test 148 reviews / 144 products**. Product-disjointness is the intended design; independently verify the three manifest intersections are empty before treating this as locked. Users **are not disjoint** (observed overlap counts: train–validation **53**, train–test **49**, validation–test **30**). Never tune, choose features, or inspect model performance repeatedly on test. **Do not report test performance in Phase 1.** If sampling is materially revised, regenerate/version the manifests and document the change before model comparison.

11. **Leakage risks and controls.** Exclude `rating` (target), product `average_rating` and `rating_number` (label-derived aggregates and potential temporal leakage), and `helpful_vote` (post-publication interaction) from initial predictive features. Check for explicit ratings mentioned in review text, duplicate or near-duplicate reviews, text preprocessing fitted outside train, user overlap across product-disjoint groups, and time-dependent product data. Fit TF-IDF, imputers and other learned preprocessing on **train only**. A diagnostic random-row 70/30 split showed **16 shared product IDs** and **69 shared user IDs**, motivating a product-group split; it is not our evaluation split.

12. **Proposed unsupervised component.** In a later phase, build TF-IDF / truncated SVD representations of review text and cluster reviews, interpret clusters using terms/examples, and test cluster stability. This is **planned**, not implemented in Phase 1.

13. **Track-specific advanced / multimodal component.** Compare **text-only**, **metadata-only** and **text + metadata** predictions with appropriate ablations; sentence embeddings are optional later. Actual pilot availability across all 1,000 reviews: text **100%**, rating **100%**, product metadata **29.7%** (297 matched reviews), valid product price **21.2%**, product images **29.7%**, product description **12.7%**, product category **29.5%**, and actual review images **2%**. Review images are too sparse to assume an image-based review model; product-image *field presence* does not imply image bytes have been downloaded. Among matched reviews, **71.38%** have valid price. Compare availability/missingness patterns before making multimodal claims.

14. **Ethical, privacy and bias risks.** Public reviews may contain identifiable details; avoid republishing user IDs or raw personal text. The data may exhibit language, product, user, temporal and positive-rating selection biases. Metadata matching is incomplete and potentially selective: matched reviews **297 (29.7%)**, unmatched **703 (70.3%)**; do not assume missing completely at random. The sequential first-N extraction is not representative of the source population. Note fairness and error differences across practical subgroups when sufficient data exist.

## Three candidate research questions and rationale

- **Q1 — text:** How effectively does review text alone classify 1–5-star ratings? Feasible on the full 1,000-review pilot.
- **Q2 — selected, multimodal:** Does non-leaking product metadata add predictive value beyond review text on **the same eligible reviews and held-out products**? Selected because it tests the distinctive Track B multimodal requirement. Current metadata availability limits sample size and the interpretation of this comparison.
- **Q3 — subgroup robustness:** Does predictive performance differ by review length or number of reviews per product? Useful as a secondary diagnostic, but many products have very few reviews in the current pilot.

## Phase 1 empirical audit snapshot

| Item | Observed value |
|---|---:|
| Reviews | 1,000 |
| Unique reviewed products | 959 |
| Unique users | 195 |
| Exact duplicate review rows | 0 |
| Empty review texts | 0 |
| Median review length | 166 characters |
| Five-star reviews | 665 (66.5%) |
| Matched reviews / linked metadata | 297 (29.7%) |
| Matched unique products / metadata records | 275 |
| Reviews without linked metadata | 703 |
| Reviews with actual attached review images | 2.0% |
| Reviews with valid product prices | 21.2% |

**EDA interpretation:** Rating classes are markedly imbalanced (especially the dominant 5-star class); text lengths are right-skewed. Boxplots suggest differing review-length distributions by rating, but no statistical significance is claimed. Most products have only a few reviews in this pilot. Matched and unmatched reviews show somewhat different rating mixes, so complete-case multimodal results must not be generalized automatically. The time trend is descriptive of the sequential pilot, **not evidence of growth in the underlying platform**.

## Remaining Phase 1 checks before marking complete

- [ ] Verify train/validation/test product-ID intersections directly from the saved manifests; retain split files and document the review-row indexing convention.
- [ ] Document provenance, source URLs, archive version/date if known, and exact sequential sampling and metadata-matching procedures in `data/README.md`.
- [ ] Check invalid values, unusual characters and text extremes; analyze whether missing product fields vary by rating/year/length; document treatment decisions.
- [ ] Consider obtaining a broader, reproducible sample with better matched metadata for the eventual core study; if done, **version and rerun** audit, EDA and group splits.
- [ ] Prepare `reports/phase1.pdf` with evidence-based findings, relevant figures, limitations, candidate questions and split/leakage rationale.
- [ ] Update main `README.md` and `AI_USAGE.md`, rerun commands from a clean environment, commit deliverables and create the `phase-1` Git tag only after review.

**Integrity note:** Current text-only diagnostic results use an earlier validation split; no final-model or held-out-test result has been produced. A GPU is available but is not needed for the present auditing/EDA work.
