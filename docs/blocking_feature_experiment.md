# Blocking-to-feature matching experiment

## 1. Objective

This experiment measures the end-to-end effect of candidate reduction:

```text
blocking → candidate pairs → pair features → matching baseline → entity macro F0.5
```

The question is how much candidate volume can be removed before downstream matching performance materially deteriorates. The existing default blocking configuration was not changed. Ground truth was used only for candidate recall, labels, threshold development, and evaluation; it was not used to construct or select production candidates.

## 2. Dataset and split

The experiment uses the processed sample data:

- Source 1: 3,000 records
- Source 2: 4,989 records
- Source 3: 5,455 records
- Ground-truth S1 rows: 3,000
- Ground-truth links: 10,444
  - S2 links: 4,989
  - S3 links: 5,455
- Zero-match S1 entities: 180

The outer split uses the existing deterministic `split_ground_truth_by_s1` function with `validation_fraction=0.2` and `seed=42`:

| Split | S1 entities | Positive links | Zero-match S1s |
|---|---:|---:|---:|
| Outer training | 2,400 | 8,358 | 134 |
| Held-out validation | 600 | 2,086 | 46 |

Threshold development uses only the outer-training partition. It is split again with `validation_fraction=0.2` and `seed=43`:

| Split | S1 entities | Positive links | Zero-match S1s |
|---|---:|---:|---:|
| Model fit | 1,920 | 6,690 | 111 |
| Threshold development | 480 | 1,668 | 23 |

The final model is fit on all 2,400 outer-training S1 entities after the threshold is selected on the inner development S1s. The 600 outer-validation S1 entities are not used for fitting or threshold selection.

There is one deterministic validation split and no independent test set, so these results are a baseline comparison rather than an unbiased estimate of generalization to a separate test population.

## 3. Blocking configurations

The evaluator compares these source-only configurations:

- `strict_overlap_2`: exact rules plus two-token name/address overlap rules and the combined token rule.
- `name_cap_25_address_cap_25`: default rules with both token frequency caps set to 25.
- `address_cap_25`: default rules with the address-token cap set to 25.
- `name_cap_25`: default rules with the name-token cap set to 25.
- `default_union`: the unchanged default configuration with both token caps set to 50.

No top-N truncation, S1-specific retention, candidate ranking, or ground-truth-derived blocking rule was used.

## 4. Candidate counts and blocking recall

| Configuration | Candidates | Positive candidate pairs | Negative candidate pairs | Avg/S1 | Median | P95 | P99 | Max | S1 >100 | Blocking recall | S2 recall | S3 recall | Missed links |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `strict_overlap_2` | 16,035 | 10,153 | 5,882 | 5.345 | 4 | 16 | 30 | 41 | 0 | 97.2137% | 96.9333% | 97.4702% | 291 |
| `name_cap_25_address_cap_25` | 70,524 | 10,319 | 60,205 | 23.508 | 22 | 49 | 64 | 110 | 1 | 98.8031% | 98.7372% | 98.8634% | 125 |
| `address_cap_25` | 109,052 | 10,354 | 98,698 | 36.351 | 34 | 76 | 91.01 | 133 | 12 | 99.1383% | 98.9176% | 99.3401% | 90 |
| `name_cap_25` | 122,759 | 10,361 | 112,398 | 40.920 | 36 | 94 | 124.01 | 170 | 119 | 99.2053% | 99.1581% | 99.2484% | 83 |
| `default_union` | 161,162 | 10,392 | 150,770 | 53.721 | 50 | 111 | 142 | 198 | 252 | 99.5021% | 99.3385% | 99.6517% | 52 |

The candidate distribution is not uniform across S1 entities. In the default union, 252 of 3,000 S1 entities have more than 100 candidates, but those entities account for 30,356 of 161,162 candidates, or 18.84% of all candidates. The reduced configurations substantially shorten this tail:

| Configuration | S1s >100 | Candidates from S1s >100 | Share of candidates |
|---|---:|---:|---:|
| `strict_overlap_2` | 0 | 0 | 0.00% |
| `name_cap_25_address_cap_25` | 1 | 110 | 0.16% |
| `address_cap_25` | 12 | 1,325 | 1.22% |
| `name_cap_25` | 119 | 13,935 | 11.35% |
| `default_union` | 252 | 30,356 | 18.84% |

This confirms that the candidate-count penalty is materially affected by a relatively small high-volume S1 tail, especially under the default and name-only cap configurations.

## 5. Feature generation

`PairFeatureGenerator` was fit on the processed entity records to create its existing TF-IDF views. Its `transform` method received only the candidate pairs emitted by the blocker. No full Source-1 × (Source-2 + Source-3) table was created.

Every configuration produced 20 pair-feature columns:

- name exact, fuzzy, token, character n-gram, and TF-IDF features
- address exact, fuzzy, token, character n-gram, and TF-IDF features
- country agreement features
- name and address length ratios

The feature matrix row count matched the generated candidate count for every configuration:

| Configuration | Feature matrix shape |
|---|---:|
| `strict_overlap_2` | `(16,035, 20)` |
| `name_cap_25_address_cap_25` | `(70,524, 20)` |
| `address_cap_25` | `(109,052, 20)` |
| `name_cap_25` | `(122,759, 20)` |
| `default_union` | `(161,162, 20)` |

Candidate labels were attached after blocking using the existing zero-to-many ground-truth contract. Multiple S2 and S3 links were retained independently.

## 6. Matching model

The baseline is a transparent `sklearn.linear_model.LogisticRegression` classifier:

- `class_weight="balanced"`
- solver: `liblinear`
- `max_iter=1000`
- `random_state=42`
- model inputs: only the 20 existing pair-feature columns

Candidate IDs, Source-1 IDs, source labels, and blocking provenance were excluded from model inputs. The model was fitted only on training S1 entities. Class weighting compensates for the negative-heavy candidate sets without sampling or using ground truth to change the production candidate universe.

## 7. Threshold procedure

Scores are positive-class probabilities. The evaluator tests thresholds from 0.00 through 1.00 in increments of 0.05.

For each configuration:

1. Fit a provisional model on the inner model-fit S1s.
2. Select the threshold with the highest inner-development entity-level macro F0.5.
3. Break tied macro F0.5 values in favor of the higher threshold.
4. Refit the classifier on all outer-training S1s.
5. Apply the fixed threshold to the untouched outer-validation S1s.

The complete development and validation threshold tables are emitted by `scripts/evaluate_blocking_features.py`, or can be saved with `--threshold-output`. Threshold selection never reads outer-validation labels.

Precision and recall below are aggregate link-level metrics over unique `(source1_entity_id, candidate_entity_id)` pairs. `macro_f0_5` is the existing entity-level metric from `src/metrics.py`.

## 8. Held-out results at the development-selected threshold

| Configuration | Threshold | Dev macro F0.5 | Validation precision | Validation recall | Validation macro F0.5 | Predicted links | TP | FP | Model FN among candidates | Blocked-out validation links | Total FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `strict_overlap_2` | 0.55 | 0.976687 | 0.993464 | 0.947267 | 0.980219 | 1,989 | 1,976 | 13 | 56 | 54 | 110 |
| `name_cap_25_address_cap_25` | 0.75 | 0.967617 | 0.982524 | 0.970278 | 0.980421 | 2,060 | 2,024 | 36 | 38 | 24 | 62 |
| `address_cap_25` | 0.85 | 0.971366 | 0.982868 | 0.962608 | 0.979116 | 2,043 | 2,008 | 35 | 58 | 20 | 78 |
| `name_cap_25` | 0.85 | 0.969514 | 0.982550 | 0.971716 | 0.980919 | 2,063 | 2,027 | 36 | 41 | 18 | 59 |
| `default_union` | 0.90 | 0.970494 | 0.983824 | 0.962128 | 0.978807 | 2,040 | 2,007 | 33 | 65 | 14 | 79 |

The validation set contains 2,086 true links. For example, `default_union` makes 2,072 of those links available to matching, so 14 are lost before scoring. The selected threshold predicts 2,040 links, including 2,007 true links and 33 false positives. Its 65 model false negatives among available candidates plus 14 blocked-out links produce 79 total false negatives.

The distinction between blocking loss and model loss is important:

```text
validation total FN = blocked-out validation links
                       + model FN among available candidates
```

This experiment reports both terms rather than treating blocking recall as final matching recall.

## 9. Interpretation

On this deterministic sample split:

- `strict_overlap_2` reduces candidates by approximately 90% relative to the default and removes the high-volume tail entirely, but loses 291 links at blocking time across the full sample. Its held-out macro F0.5 is 0.980219.
- `name_cap_25_address_cap_25` reduces candidates by approximately 56% and retains a 98.8031% blocking recall. Its held-out macro F0.5 is 0.980421.
- `address_cap_25` reduces candidates by approximately 32% and retains 99.1383% blocking recall. Its held-out macro F0.5 is 0.979116.
- `name_cap_25` reduces candidates by approximately 24% and retains 99.2053% blocking recall. Its held-out macro F0.5 is 0.980919.
- `default_union` has the highest blocking recall and largest candidate tail, but its held-out macro F0.5 is 0.978807 on this split.

These results do not establish a universally optimal configuration. The reduced configurations have fewer candidate pairs and smaller tails, while the default preserves more links before matching. The differences in held-out macro F0.5 are small enough that the challenge's exact candidate-count penalty, repeated splits, and full-data behavior remain important.

## 10. Leakage and correctness checks

- Blocking indexes use only processed S1/S2/S3 records.
- Ground truth is not used in candidate construction, ranking, top-N selection, or per-S1 capping.
- The classifier sees only candidate rows generated by the blocker.
- Validation S1 entities are excluded from model fitting and threshold selection.
- Multiple true links are represented as sets, not collapsed to one match.
- S1 entities with no true links and S1 entities with no candidates remain in macro F0.5 evaluation.
- S2 and S3 candidates are evaluated together and retain their source-prefixed IDs.
- No changes were made to preprocessing, blocking, pair features, or metrics.

## 11. Runtime and limitations

On the sample data, the evaluator's per-configuration runtimes were approximately:

| Configuration | Runtime (seconds) |
|---|---:|
| `strict_overlap_2` | 7.10 |
| `name_cap_25_address_cap_25` | 22.34 |
| `address_cap_25` | 32.89 |
| `name_cap_25` | 36.76 |
| `default_union` | 47.35 |

The complete five-configuration run took approximately 148 seconds and reached approximately 779 MB peak RSS in the local environment. These are sample measurements, not a large-data benchmark. Pair-feature generation is currently row-oriented and may require additional engineering for full challenge-scale files.

Limitations include:

- one deterministic outer validation split
- no independent test set
- threshold grid resolution of 0.05
- a simple logistic baseline rather than a tuned matching model
- no challenge-provided candidate penalty formula available in this repository
- no claim that sample ranking will exactly match the external challenge ranking

## 12. Next experiment

Keep the default blocking configuration unchanged until the downstream evaluation is repeated on additional S1-level splits or full challenge data. The most informative next comparison is among `strict_overlap_2`, `name_cap_25_address_cap_25`, `name_cap_25`, and `default_union` using:

1. the exact challenge candidate-count penalty,
2. repeated deterministic S1 validation splits,
3. a fixed matching-model protocol, and
4. the same separation of blocked-out links from model false negatives.

The current results suggest that `name_cap_25_address_cap_25` and `name_cap_25` are plausible efficiency candidates, but the evidence is not sufficient to replace the default production configuration.
