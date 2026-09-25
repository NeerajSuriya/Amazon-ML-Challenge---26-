# Blocking experiment log

## 2026-09-25 — sample blocking baseline

### Data and configuration

- S1: 3,000 records; S2: 4,989; S3: 5,455.
- Ground-truth links: 10,444 (4,989 S2 and 5,455 S3).
- Country compatibility: reject only when both normalized countries are
  non-empty and unequal; missing country remains eligible.
- Empty normalized names/addresses are never indexed.
- Name-token frequency limit: 50 candidate records.
- Address-token frequency limit: 50 candidate records.
- Exact-key bucket limit: 500 candidate records.
- Candidate identity: `(source1_entity_id, candidate_entity_id)`.
- Duplicate pairs are unioned and retain sorted `blocking_rules` provenance.

### Rule comparison

| Configuration | Candidates | Avg/S1 | Median/S1 | Max/S1 | Recall | S2 recall | S3 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `country_name_exact` | 2,453 | 0.82 | 1 | 5 | 23.19% | 23.15% | 23.23% |
| `country_address_exact` | 811 | 0.27 | 0 | 4 | 7.77% | 11.55% | 4.31% |
| `name_token` | 70,989 | 23.66 | 22 | 93 | 83.66% | 81.16% | 85.94% |
| `address_token` | 98,717 | 32.91 | 27 | 170 | 94.37% | 94.45% | 94.30% |
| `name_address_token` | 8,549 | 2.85 | 3 | 12 | 78.52% | 76.27% | 80.59% |
| `union_default` | 161,162 | 53.72 | 50 | 198 | 99.50% | 99.34% | 99.65% |

The full Cartesian comparison space is 31,332,000 pairs. The default union
produces 161,162 unique pairs, approximately 0.51% of that space, while
retrieving 10,392 of 10,444 true links.

### Notes

- Exact-name and exact-address rules alone are insufficient for recall.
- Address tokens contribute more recall than name tokens on this sample, but
  both are needed for complementary coverage.
- The default union is intentionally recall-oriented. The 52 unretrieved true
  links should be investigated before lowering token-frequency limits or adding
  more permissive rules.
- Blocking produces plausible candidates only; downstream pair features and
  matching logic remain responsible for final decisions.

## 2026-09-25 — submission-oriented per-S1 analysis

The updated challenge requirement scores `candidate_pairs.tsv` as well as
matching output and prefers a smaller candidate set per Source-1 entity. The
default union was evaluated without using ground truth to construct candidates.
Ground truth was used only for recall measurements.

### Default per-S1 distribution

| Population | S1 count | Mean | Median | P75 | P90 | P95 | P99 | Max | 0 candidates | 1 candidate | <=5 | <=10 | <=20 | <=50 | <=100 | >100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All S1 | 3,000 | 53.72 | 50 | 73.00 | 97.00 | 111.00 | 142.00 | 198 | 4 | 4 | 49 | 152 | 435 | 1,529 | 2,748 | 252 |
| S1 with truth | 2,820 | 54.22 | 50 | 73.00 | 98.00 | 111.00 | 142.00 | 198 | 0 | 3 | 40 | 132 | 395 | 1,422 | 2,577 | 243 |
| S1 with zero truth | 180 | 45.83 | 40 | 64.25 | 84.10 | 97.35 | 138.05 | 164 | 4 | 1 | 9 | 20 | 40 | 107 | 171 | 9 |

The default remains 161,162 candidates, with mean 53.72, median 50, P95
111, P99 142, maximum 198, and 99.5021% blocking recall. Four S1 entities
have no candidates; all four are zero-truth entities.

### Rule contribution analysis

`unique_candidate_count` means a pair emitted by that rule and no other default
rule. `overlap_candidate_count` means the rule's pair is also emitted by at
least one other rule. The unique true-link column is evaluation-only.

| Rule | Candidates | Unique candidates | Overlap candidates | True links | Unique true links | Mean/S1 | P95 | P99 | Max | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `country_name_exact` | 2,453 | 5 | 2,448 | 2,422 | 0 | 0.82 | 3 | 4 | 5 | 23.1903% |
| `country_address_exact` | 811 | 0 | 811 | 811 | 0 | 0.27 | 1 | 2 | 4 | 7.7652% |
| `name_token` | 70,989 | 62,266 | 8,723 | 8,737 | 388 | 23.66 | 57 | 72 | 93 | 83.6557% |
| `address_token` | 98,717 | 90,004 | 8,713 | 9,856 | 1,491 | 32.91 | 85 | 116.01 | 170 | 94.3700% |
| `name_address_token` | 8,549 | 0 | 8,549 | 8,201 | 0 | 2.85 | 6 | 7 | 12 | 78.5236% |

The exact rules add only five unique candidate identities in the union and no
unique true links on this sample. `address_token` contributes the largest
unique candidate volume and the largest unique true-link set. `name_token` is
also substantial and recovers 388 true links not recovered by the other
individual rules. `name_address_token` is entirely redundant as a candidate
identity rule under the current default union, although it remains useful as a
separately observable provenance/evidence rule.

### Controlled candidate-reduction configurations

These configurations were selected from source-only keys and limits before
reading their ground-truth results. No arbitrary top-N truncation was used.

| Configuration | Candidates | Avg/S1 | Median | P95 | P99 | Max | Recall | S2 recall | S3 recall | Missed links |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_only` | 3,140 | 1.05 | 1 | 3 | 4 | 6 | 29.7683% | 32.2109% | 27.5344% | 7,335 |
| `no_name_token` | 98,896 | 32.97 | 27 | 67 | 85 | 170 | 95.7871% | 95.7707% | 95.8020% | 440 |
| `no_address_token` | 71,158 | 23.72 | 22 | 50 | 57 | 93 | 85.2260% | 83.5638% | 86.7461% | 1,543 |
| `name_cap_10` | 103,804 | 34.60 | 29 | 69 | 87.05 | 119 | 98.5446% | 98.5167% | 98.5701% | 152 |
| `address_cap_10` | 82,469 | 27.49 | 25 | 54 | 62 | 96 | 98.2861% | 98.0357% | 98.5151% | 179 |
| `strict_overlap_3` | 10,655 | 3.55 | 3 | 7 | 13.01 | 25 | 92.1103% | 91.7819% | 92.4106% | 824 |
| `strict_overlap_2` | 16,035 | 5.35 | 4 | 16 | 30 | 41 | 97.2137% | 96.9333% | 97.4702% | 291 |
| `name_cap_25_address_cap_10` | 43,855 | 14.62 | 12 | 34 | 46 | 59 | 97.7116% | 97.7150% | 97.7085% | 239 |
| `name_cap_10_address_cap_25` | 51,518 | 17.17 | 14 | 41 | 57 | 108 | 98.0180% | 97.9755% | 98.0568% | 207 |
| `name_cap_25_address_cap_25` | 70,524 | 23.51 | 22 | 49 | 64 | 110 | 98.8031% | 98.7372% | 98.8634% | 125 |
| `address_cap_25` | 109,052 | 36.35 | 34 | 76 | 91 | 133 | 99.1383% | 98.9176% | 99.3401% | 90 |
| `name_cap_25` | 122,759 | 40.92 | 36 | 94 | 124 | 170 | 99.2053% | 99.1581% | 99.2484% | 83 |
| `tokens_only` | 161,157 | 53.72 | 50 | 111 | 142 | 198 | 99.5021% | 99.3385% | 99.6517% | 52 |
| `default_union` | 161,162 | 53.72 | 50 | 111 | 142 | 198 | 99.5021% | 99.3385% | 99.6517% | 52 |
| `address_overlap_2_only` | 76,097 | 25.37 | 24 | 59 | 75 | 104 | 98.2478% | 97.9154% | 98.5518% | 183 |
| `name_overlap_2_only` | 101,100 | 33.70 | 28 | 86.05 | 116 | 170 | 98.4680% | 98.3564% | 98.5701% | 160 |

The `strict_overlap_2` row is the strongest candidate-count reduction in the
measured set, but it drops recall to 97.21%; it is not a replacement default.
Lowering both token caps to 25 reduces the P95 from 111 to 49 and the maximum
from 198 to 110, but misses 125 links. The default is not dominated by the
reduced configurations when recall is treated as a first-class objective.

### Pareto interpretation

Using total candidates, P95, P99, and recall as the dominance dimensions, the
measured frontier was:

- `exact_only`: 3,140 candidates, P95 3, P99 4, recall 29.7683%.
- `strict_overlap_3`: 10,655, P95 7, P99 13.01, recall 92.1103%.
- `strict_overlap_2`: 16,035, P95 16, P99 30, recall 97.2137%.
- `name_cap_25_address_cap_10`: 43,855, P95 34, P99 46, recall 97.7116%.
- `name_cap_10_address_cap_25`: 51,518, P95 41, P99 57, recall 98.0180%.
- `name_cap_25_address_cap_25`: 70,524, P95 49, P99 64, recall 98.8031%.
- `address_cap_25`: 109,052, P95 76, P99 91, recall 99.1383%.
- `name_cap_25`: 122,759, P95 94, P99 124, recall 99.2053%.
- `tokens_only`: 161,157, P95 111, P99 142, recall 99.5021%.

This frontier shows the expected recall-versus-tail-volume tradeoff; it does not
identify a universally best configuration. The next decision must measure the
end-to-end matching score and the challenge's exact per-S1 ranking formula.

### Candidate-pair submission format

No challenge submission schema, `candidate_pairs.tsv`, or `matching_results.tsv`
documentation exists in this repository. The local production writer therefore
uses the existing pair-feature contract:

- Required default columns: `source1_entity_id`, `candidate_entity_id`.
- Candidate IDs retain the `S2-`/`S3-` source prefix, so a separate source column
  is not needed by the local contract.
- Optional local-analysis columns: `candidate_source`, `blocking_rules`.
- Duplicate identity pairs are rejected/removed by the blocker and the writer
  validates uniqueness.
- Output is tab-separated and sorted by S1 ID then candidate ID.
- No ordering requirement is claimed for the external challenge because none is
  present in repository materials.

Run the production writer with explicit processed files, for example:

```bash
myenv/bin/python scripts/generate_candidate_pairs.py \
  --processed-root processed/sample \
  --source1-name train_source1_sample.tsv \
  --source2-name train_source2_sample.tsv \
  --source3-name train_source3_sample.tsv \
  --output candidate_pairs.tsv
```

The sample writer emitted 161,162 rows with exactly the two required identity
columns, no duplicate pairs, and valid S1/S2/S3 IDs. With provenance enabled it
emits the two optional metadata columns after the identity columns.

### Runtime and scale observations

On the sample (3,000 S1, 4,989 S2, 5,455 S3), the expanded evaluator completed
in approximately 14.5 seconds and reported approximately 351 MB peak RSS on
macOS. The production writer generated the default 161,162-row file in
approximately 0.9 seconds and reported approximately 295 MB peak RSS. These
figures include pandas input loading and repeated evaluation configurations;
they are not a large-data benchmark. Candidate construction uses exact/inverted
indexes and never materializes the 31,332,000-pair Cartesian product. Large
runs should use the existing chunked preprocessing and be benchmarked with the
actual challenge files.

### Next experiment

Do not change the default yet. The recommended next experiment is to run the
same candidate configurations through downstream pair scoring and entity-level
macro F0.5 while recording the challenge's candidate-count penalty. In
particular, compare `default_union`, `name_cap_25_address_cap_25`, and
`strict_overlap_2`; the latter has a much smaller per-S1 tail but loses 291
true links at blocking time.
