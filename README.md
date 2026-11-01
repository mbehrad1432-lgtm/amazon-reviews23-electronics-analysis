# Amazon Reviews’23 — Electronics (Track B)

Phase 1 starter repository. GPU is available for future embedding/deep-learning experiments; initial data audit and TF-IDF baseline run on CPU.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python -m src.download --max-reviews 50000 --max-metadata 50000
python -m src.audit
python -m src.baseline
```

Download streams from the UCSD Amazon Reviews’23 category archives; it stores bounded JSONL samples rather than full category archives. Sampling is **first N records** and therefore exploratory, not representative. Change sample sizes using CLI arguments. If URLs change, pass `--reviews-url` / `--metadata-url` or update configuration.

Audit outputs: `reports/phase1/audit.json`, `reports/phase1/summary.md`, `figures/rating_distribution.png`. Baseline outputs: `reports/phase1/baseline_metrics.json`. The baseline uses group-aware holdout by product, with training-only TF-IDF fitting. It is preliminary, not a final test result.

## Dataset source
Amazon Reviews’23 (Hou et al.), UCSD McAuley Lab: https://amazon-reviews-2023.github.io/ . See `data/README.md` for provenance and sampling caveats.

## Phase 1 completion
Audit actual data; confirm dataset subset and question; update PROJECT_CARD; run EDA and baseline; prepare `reports/phase1.pdf`; update README and AI_USAGE; commit and tag `phase-1` only after validation.
