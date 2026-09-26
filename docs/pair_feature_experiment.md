# Pair-feature audit and controlled experiment

## Scope and contracts

This experiment covers the part of the team pipeline after candidate generation:

```text
blocked candidate pairs -> pair features -> matcher -> development threshold
-> entity-level macro F0.5
```

Preprocessing, normalization, `src/blocking.py`, blocking defaults, zero-to-many
matching, and S2/S3 source identity were not changed. Candidates are supplied by
the default-union blocker; the experiment never constructs a Cartesian product.
The current matcher remains balanced `LogisticRegression`. A small matcher
protocol/adapter is available for Neeraj's future model without changing the
evaluator's candidate-level or entity-level semantics.

All splits are by S1 entity. The outer split creates training and held-out
validation S1s. The inner split creates model-fit and development S1s. The
threshold is selected only from the development table. The final model is fit on
all outer-training S1s and evaluated once on held-out validation S1s. Validation
labels are post-hoc reporting only.

## Current feature inventory

The default generator still emits the original 20 columns, in this order:

- **Name:** `name_exact`, `name_ratio`, `name_token_set_ratio`,
  `name_token_sort_ratio`, `name_jaccard`, `name_overlap`,
  `name_char_ngram_jaccard`, `name_tfidf_cosine`.
- **Address:** `address_exact`, `address_ratio`, `address_token_set_ratio`,
  `address_token_sort_ratio`, `address_jaccard`, `address_overlap`,
  `address_char_ngram_jaccard`, `address_tfidf_cosine`.
- **Country:** `country_exact`, `country_nonempty_both`.
- **Length:** `name_length_ratio`, `address_length_ratio`.

Experimental columns are opt-in (`PairFeatureGenerator(include_experimental=True)`):

- **NAME:** partial ratio and directional token containment (3 columns).
- **ADDRESS:** partial ratio, directional containment, numeric-token Jaccard
  and numeric-token overlap (5 columns).
- **COUNTRY:** either-missing and both-missing indicators (2 columns).
- **CROSS_FIELD:** name/address ratio product, TF-IDF product, minimum ratio,
  and both-fields-high indicator (4 columns).

Thus the complete opt-in matrix has 34 columns. The baseline path remains 20
columns and its existing feature values are unchanged.

## Baseline audit

The audit was run on the default-union candidates:

- candidates / feature rows: **161,162**
- baseline feature matrix: **161,162 x 20**
- development rows / positive links: **26,203 / 1,654**
- held-out validation rows / positive links: **30,603 / 2,072**
- development-selected threshold: **0.90**
- validation labels used for selection: **no**

### Feature diagnostics

The machine-readable diagnostics are in `artifacts/pair_feature_audit/`.
Notable development-partition observations:

- `country_exact` and `country_nonempty_both` are constant at 1.0 on this
  partition (AUC 0.5, no separation). They are retained for contract
  compatibility, but the country-missingness experiment did not change the
  result on this data.
- The strongest univariate features are address-based:
  `address_overlap` AUC 0.9635, `address_jaccard` 0.9622,
  `address_tfidf_cosine` 0.9612, `address_char_ngram_jaccard` 0.9596,
  and `address_token_set_ratio` 0.9595.
- Name TF-IDF and character n-gram Jaccard are useful but weaker (AUC 0.9108
  and 0.9049). `address_exact` is rare and weak (AUC 0.5345).
- Length ratios are weaker, especially address length ratio (AUC 0.6452).
- Coefficients show strong multicollinearity: for example, address TF-IDF
  (+11.22), name TF-IDF (+10.78), address Jaccard (+6.43), and address
  character Jaccard (+6.11) dominate absolute magnitude, while several
  correlated fuzzy features have negative coefficients. This is a diagnostic
  signal, not grounds for deleting features without a separate experiment.

### Development and held-out results

At the threshold selected from development (0.90):

| Stage | Precision | Recall | Macro F0.5 | TP | FP | model FN | blocked-out FN | total FN | Predicted links |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 0.976984 | 0.967026 | 0.970494 | 1,613 | 38 | 41 | 14 | 55 | 1,651 |
| Held-out validation | 0.983824 | 0.962128 | 0.978807 | 2,007 | 33 | 65 | 14 | 79 | 2,040 |

The validation threshold table is emitted in
`artifacts/pair_feature_audit/threshold_validation.tsv`. For reference, the
held-out metric at threshold 0.95 is macro F0.5 0.980691, but it is not the
selected threshold because selection is development-only.

## Error analysis

The audit emits:

- `validation_model_false_negatives.tsv`: all 65 validation true pairs that
  survived blocking but scored below 0.90, with all baseline features and
  normalized source values.
- `validation_false_positives_high_confidence.tsv`: the top 25 false positives
  by score.
- `validation_false_positives_near_threshold.tsv`: false positives within 0.05
  of the selected threshold.
- `validation_error_summary.tsv`: counts by source, score band, missingness,
  and exact-agreement flags.

The 65 model false negatives are not blocking misses: they are available
candidate pairs that the matcher did not select. Examples show abbreviated or
misspelled names, transliteration, reordered address components, and records
with a missing candidate address. The examples include strong evidence in one
field but insufficient combined score to reach 0.90.

The 33 false positives are often driven by generic/shared address tokens or
locations combined with country agreement, while names are weak or unrelated.
There are also repeated S1 patterns where several candidates share the same
address wording. In the combined error table, 45 rows are S2 candidates and 53
are S3 candidates; 33 rows have a missing candidate address. These are
observational categories, not labels used for feature selection.

## Evidence-backed feature groups

The additions were deliberately limited to patterns visible in the processed
fields and error tables:

| Group | Added signals | Motivation | Expected benefit |
|---|---|---|---|
| NAME | partial ratio; S1-to-candidate and candidate-to-S1 token containment | abbreviations, suffixes, partial names, and asymmetric token sets | recall, with possible precision cost |
| ADDRESS | partial ratio; directional containment; numeric-token Jaccard/overlap | reordered/abbreviated addresses and house/postal-number evidence | recall and precision against generic text |
| COUNTRY | either-missing and both-missing indicators | distinguish agreement from missingness without using truth | precision in mixed-missing cases |
| CROSS_FIELD | ratio/TF-IDF products, minimum similarity, both-high indicator | separate two-field agreement from one-field-only agreement | precision and threshold discrimination |

No feature reads ground truth, candidate labels, or validation outcomes.

## Controlled ablation on fixed candidates

Every set below used the same 161,162 default-union candidates, nested split,
balanced logistic family, threshold grid, and entity-level metric. The single
seed-42 results are:

| Feature set | Count | Dev threshold | Validation precision | Validation recall | Validation macro F0.5 | TP | FP | model FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 0.90 | 0.983824 | 0.962128 | 0.978807 | 2,007 | 33 | 65 |
| baseline + NAME | 23 | 0.90 | 0.984298 | 0.961649 | 0.978967 | 2,006 | 32 | 66 |
| baseline + ADDRESS | 25 | 0.95 | 0.989152 | 0.961649 | 0.984120 | 2,006 | 22 | 66 |
| baseline + COUNTRY | 22 | 0.90 | 0.983824 | 0.962128 | 0.978807 | 2,007 | 33 | 65 |
| baseline + CROSS_FIELD | 24 | 0.90 | 0.985330 | 0.965964 | 0.983126 | 2,015 | 30 | 57 |
| baseline + NAME + ADDRESS + CROSS_FIELD | 32 | 0.95 | 0.990128 | 0.961649 | 0.984874 | 2,006 | 20 | 66 |

The single split suggests that address and cross-field additions are promising,
but it is not sufficient to declare a final feature set.

## Five-seed robustness experiment

The focused robustness run used outer S1 seeds **42, 123, 2024, 3407, and
7777**, with inner development seeds `seed + 1`. Every configuration used the
same fixed default-union candidates: 161,162 rows, blocking recall 0.995021
(S2 0.993385, S3 0.996517). The selected threshold was always chosen from the
inner development report for that seed/configuration. Validation labels were
used only after selection for held-out reporting.

### Aggregate validation results

Standard deviations are population standard deviations across the five
validation splits. Candidate count is constant because candidates are fixed.

| Feature set | Features | Macro F0.5 mean +/- std | Min--max | Precision mean | Recall mean | Threshold mean | Candidate count mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 0.978307 +/- 0.003644 | 0.971772--0.982877 | 0.984141 | 0.960046 | 0.930 | 161,162 |
| baseline + ADDRESS | 25 | **0.981543 +/- 0.003542** | 0.975303--0.984942 | 0.985547 | 0.966605 | 0.930 | 161,162 |
| baseline + CROSS_FIELD | 24 | 0.979442 +/- 0.004843 | 0.970109--0.983126 | 0.982900 | 0.965557 | 0.910 | 161,162 |
| baseline + ADDRESS + CROSS_FIELD | 29 | **0.981731 +/- 0.003409** | 0.975533--0.984808 | 0.985877 | 0.968202 | 0.920 | 161,162 |
| baseline + NAME + ADDRESS + CROSS_FIELD | 32 | 0.981938 +/- 0.003505 | 0.975735--0.985219 | 0.984966 | **0.970029** | 0.910 | 161,162 |

The complete NAME+ADDRESS+CROSS_FIELD set has the highest mean macro F0.5 and
recall by a small margin, but it is not consistently better than baseline. The
ADDRESS+CROSS_FIELD set is the strongest configuration with a positive paired
improvement on every seed while avoiding the larger variability of the full
combination.

### Per-seed macro F0.5

| Seed | Baseline | + ADDRESS | + CROSS_FIELD | + ADDRESS + CROSS_FIELD | + NAME + ADDRESS + CROSS_FIELD |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.978807 | 0.984120 | 0.983126 | 0.984799 | 0.984874 |
| 123 | 0.979894 | 0.983342 | 0.981722 | 0.982399 | 0.985219 |
| 2024 | 0.982877 | 0.984942 | 0.982821 | 0.984808 | 0.980597 |
| 3407 | 0.978184 | 0.980007 | 0.979434 | 0.981113 | 0.983264 |
| 7777 | 0.971772 | 0.975303 | 0.970109 | 0.975533 | 0.975735 |

### Paired differences against baseline

Each value is the same-seed feature-set macro F0.5 minus the baseline macro
F0.5. This avoids comparing different validation populations.

| Feature set | Seed differences | Mean improvement | Std. dev. | Min--max | Positive seeds |
|---|---|---:|---:|---:|---:|
| baseline + ADDRESS | +0.005314, +0.003449, +0.002065, +0.001823, +0.003531 | **+0.003236** | 0.001398 | +0.001823--+0.005314 | 5/5 |
| baseline + CROSS_FIELD | +0.004319, +0.001828, -0.000057, +0.001250, -0.001662 | +0.001136 | 0.002229 | -0.001662--+0.004319 | 3/5 |
| baseline + ADDRESS + CROSS_FIELD | +0.005992, +0.002506, +0.001931, +0.002929, +0.003762 | **+0.003424** | 0.001583 | +0.001931--+0.005992 | 5/5 |
| baseline + NAME + ADDRESS + CROSS_FIELD | +0.006067, +0.005325, -0.002281, +0.005080, +0.003963 | +0.003631 | 0.003390 | -0.002281--+0.006067 | 4/5 |

### Incremental CROSS_FIELD value on ADDRESS

Comparing `baseline + ADDRESS + CROSS_FIELD` directly against `baseline +
ADDRESS` gives per-seed differences of `+0.000679, -0.000943, -0.000134,
+0.001106, +0.000231`: mean **+0.000188**, standard deviation 0.000703,
range -0.000943 to +0.001106, positive on 3/5 seeds. CROSS_FIELD adds a small
average improvement on top of ADDRESS, but the improvement is not consistent
across all splits.

### Exploratory conclusion and production boundary

ADDRESS improvement over baseline is consistent in this five-seed experiment:
all five paired differences are positive, with mean +0.003236 macro F0.5. This
is stronger evidence than the earlier three-seed result, but it remains a sample
experiment and does not authorize changing production defaults. ADDRESS+CROSS_FIELD
has a slightly larger and also consistent paired gain, while the full NAME
combination has the highest mean but regresses on seed 2024. No production
configuration or default matcher was changed automatically.

## Post-hoc error analysis

`artifacts/pair_feature_robustness/error_analysis.tsv` contains the error
artifact for `baseline_plus_address_cross_field` on validation seed 42, the
most consistently supported configuration by paired differences. It contains
false positives, model false negatives among generated candidates, and blocked
true links, with source IDs, S2/S3 source tags, normalized name/address/country
values, selected feature values, model probability, threshold, and error type.
The artifact is post-hoc and did not influence threshold or feature selection.

Recurring patterns remain generic/shared address locations combined with weak
names among false positives, and abbreviated/transliterated/reordered or
partially missing fields among false negatives. Blocked links are kept distinct
from model false negatives; no individual examples were used to modify blocking
rules.

## Matcher interface

`src/matching.py` defines `Matcher.fit(features, feature_names)` and
`Matcher.predict_proba(features, feature_names)`. `LogisticMatcher` is an
evaluation/reference matcher only; it is **not** the team's production
matching model. It exists so that blocking and pair-feature experiments can be
evaluated before Neeraj's production matcher is available.

`LogisticMatcher` preserves the current class balancing, solver, iteration
limit, random seed, and empty/single-class fallbacks. `fit_matcher()` and
`score_matcher()` in the evaluation script accept an injected matcher factory,
preserve candidate row order, and pass only selected feature columns. No
one-to-one assignment is introduced. Neeraj's future matcher should implement
or adapt to the same narrow contract:

```python
matcher.fit(features, feature_names)
matcher.predict_proba(features, feature_names)
```

It can then be benchmarked through the same candidate-level threshold and
entity-level macro-F0.5 evaluation without changing blocking, pair features,
or metric semantics.

## Reproducibility

```bash
myenv/bin/python scripts/audit_pair_features.py
myenv/bin/python scripts/evaluate_pair_feature_ablations.py
myenv/bin/python scripts/evaluate_pair_feature_ablations.py --seeds 42,123,2024
myenv/bin/python -m pytest -q
git diff --check
```

Artifacts are written under `artifacts/pair_feature_audit/` and
`artifacts/pair_feature_ablations/`. Runtime for the six-set single-seed
ablation was approximately 3.6--3.8 seconds in this environment; this is
reported per row in `per_seed.tsv`.

## Limitations and next experiment

This remains a sample-dataset, five-split robustness experiment. The feature
sets were predeclared from development-side diagnostics, but the aggregate
validation table is evidence for follow-up rather than a final model-selection
claim. The next experiment should test the address and cross-field groups on
full training data and additional independent S1 seeds, then evaluate Neeraj's
replaceable matcher under the same fixed candidates and development-only
threshold protocol. Any production choice should inspect the remaining
high-confidence false positives for generic-location collisions and verify that
numeric address signals do not overfit locality artifacts. In particular, the
small incremental CROSS_FIELD gain over ADDRESS is not stable enough to treat
as a standalone production recommendation.
