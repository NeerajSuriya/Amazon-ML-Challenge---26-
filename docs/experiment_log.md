# Entity-resolution experiment log

## 2026-09-25 — feature/evaluation stage baseline

- **Data:** `processed/sample/train_source1_sample.tsv`,
  `train_source2_sample.tsv`, `train_source3_sample.tsv`, and
  `dataset/sample/train_ground_truth_sample.tsv`.
- **Ground truth:** 3,000 S1 entities and 3,000 rows; 180 S1 entities have
  zero true matches; 10,444 positive links consist of 4,989 S2 links and
  5,455 S3 links. Links are retained as zero-to-many sets.
- **Validation split:** deterministic S1-level split with
  `validation_fraction=0.2`, `seed=42`.
  - Train: 2,400 S1 entities, 134 empty-truth entities, 8,358 positive
    links (3,973 S2 / 4,385 S3).
  - Validation: 600 S1 entities, 46 empty-truth entities, 2,086 positive
    links (1,016 S2 / 1,070 S3).
  - S1 overlap: 0.
- **Features:** exact agreement, RapidFuzz string ratios, token Jaccard and
  overlap, character n-gram Jaccard, name/address TF-IDF cosine, country
  agreement, and text-length ratios. JSON list columns are parsed with
  `json.loads()` at feature-generation time.
- **Candidate generation:** intentionally external; no blocking or candidate
  generation is implemented here.
- **Model/threshold:** none yet. This entry establishes the data contract and
  evaluator baseline only.
- **Verification:** `myenv/bin/python -m pytest -q` and a full-sample smoke
  test for candidate transformation, labeling, and S1 split integrity.
