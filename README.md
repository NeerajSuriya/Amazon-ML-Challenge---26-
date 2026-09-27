# Amazon Business Entity Resolution

This repository implements a reproducible zero-to-many entity-resolution
pipeline for three business data sources:

- Source 1: deduplicated reference entities (`S1-*`)
- Source 2 and Source 3: noisy records (`S2-*`, `S3-*`)

For every Source-1 entity, the pipeline predicts all matching Source-2 and/or
Source-3 IDs. An entity may have zero, one, or many matches, including matches
in both noisy sources. It never applies one-to-one assignment.

## Dataset contract

A standard dataset has:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Source files contain `entity_id`, `business_name`, `business_address`, and
`country`. Ground truth contains `source1_entity_id` and a comma-separated,
possibly empty, `matched_entity_ids` cell. Country is an open-set string; no
country list is hard-coded, so countries present only in test data (such as
France) remain valid.

## Pipeline

The reproducible command is:

```bash
python scripts/run_submission_pipeline.py \
  --dataset-root dataset \
  --output-dir output
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The command:

1. preprocesses raw TSVs using `src/preprocessing.py`;
2. generates the final deduplicated S1-to-(S2/S3) candidate set with
   `src/blocking.py` (never a full Cartesian product);
3. generates pair features using `src/pair_features.py`;
4. labels candidates from zero-to-many ground truth;
5. trains and scores the frozen `src/matcher.py::ProductionMatcher`;
6. applies the default threshold `match_probability >= 0.85`;
7. aggregates candidates and matches into one row per test S1; and
8. validates the final output invariants.

The orchestration uses temporary processed files by default. The preprocessing
stage supports chunked input, and inference feature generation/scoring supports
`--inference-chunk-size` while preserving candidate row order.

### Checked-in sample

Only a training sample is checked in locally. It can be used as a deterministic
train/inference fixture (this is not a claim about an independent test score):

```bash
python scripts/run_submission_pipeline.py \
  --train-dir dataset/sample \
  --test-dir dataset/sample \
  --ground-truth dataset/sample/train_ground_truth_sample.tsv \
  --train-source1-name train_source1_sample.tsv \
  --train-source2-name train_source2_sample.tsv \
  --train-source3-name train_source3_sample.tsv \
  --train-ground-truth-name train_ground_truth_sample.tsv \
  --test-source1-name train_source1_sample.tsv \
  --test-source2-name train_source2_sample.tsv \
  --test-source3-name train_source3_sample.tsv \
  --evaluation-ground-truth dataset/sample/train_ground_truth_sample.tsv \
  --output-dir output

python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/sample \
  --test-source1-name train_source1_sample.tsv \
  --test-source2-name train_source2_sample.tsv \
  --test-source3-name train_source3_sample.tsv
```

`--evaluation-ground-truth` is opt-in and is intended only for a local
validation fixture. Real test inference does not read test ground truth.

## Preprocessing and blocking

`src/preprocessing.py` preserves raw columns and adds normalized name, address,
and country strings, tokens, character n-grams, and a source namespace. It uses
Unicode normalization, conservative business/address abbreviation expansion,
and no external enrichment.

`src/blocking.py::CandidateBlocker` uses the existing deterministic union of:

- exact normalized name within compatible country;
- exact normalized address within compatible country;
- name-token blocks;
- address-token blocks; and
- candidates sharing both name and address tokens.

Generic token and exact buckets are frequency/size capped. Missing countries
are compatible; known unequal countries are rejected. Candidate identity is
`(source1_entity_id, candidate_entity_id)`, candidates are deduplicated, only
S2/S3 namespaces are emitted, and S1 records with no usable blocking key can
have zero candidates.

`output/candidate_pairs.tsv` is the **final** candidate set passed to the
matcher, not an earlier diagnostic file. Its exact columns are:

```text
source1_entity_id    candidate_entity_ids
```

Every test S1 appears exactly once, including an empty candidate list.

## Features and matcher

Production uses exactly these 20 baseline pair features, in this order:

```text
name_exact
name_ratio
name_token_set_ratio
name_token_sort_ratio
name_jaccard
name_overlap
name_char_ngram_jaccard
name_tfidf_cosine
address_exact
address_ratio
address_token_set_ratio
address_token_sort_ratio
address_jaccard
address_overlap
address_char_ngram_jaccard
address_tfidf_cosine
country_exact
country_nonempty_both
name_length_ratio
address_length_ratio
```

`ProductionMatcher` is protected production infrastructure. It is a
`HistGradientBoostingClassifier` with fixed random state/configuration and
independent pair probabilities. It does not perform one-to-one assignment.
The default production threshold is `0.85`; threshold overrides are explicit
local analysis options and are never tuned automatically from test data. When
an explicit evaluation ground truth is available, several thresholds can be
reported without changing production output:

```bash
--evaluation-ground-truth path/to/local_ground_truth.tsv \
--evaluate-thresholds 0.75,0.85,0.90
```

`LogisticMatcher` in `src/matching.py` is evaluation/reference infrastructure
only and is not the production model. The production dependency is scikit-learn
and the `HistGradientBoostingClassifier` is an MIT-compatible, non-generative
model well below the eight-billion-parameter limit; no external foundation
model is introduced.

## Metrics and validation

`src/metrics.py` implements entity-level macro F0.5. Predictions and truth are
sets per S1, repeated truth rows are unioned, missing prediction keys are empty,
and both empty sets score 1.0. The metric is zero-to-many and is not a global
pair-level score.

`utils/validate_submission.py` checks:

- exact two-column schemas and TSV readability;
- exactly one row for every test S1;
- valid, existing S2/S3 IDs only;
- no duplicate IDs or S1 IDs inside lists;
- empty lists are accepted; and
- every final match is a member of that S1's final candidate list.

## Full data and prohibited enrichment

No standard `dataset/train` or `dataset/test` full-data directories are checked
into this repository. The standard command is ready for the challenge files;
if those paths are absent it fails clearly rather than fabricating a result.
No external business lookup, geocoding, API, web search, or data augmentation
is used.

## Tests and packaging

Run the complete suite with:

```bash
myenv/bin/pytest -q
myenv/bin/python scripts/build_submission_zip.py --team-name <team-name>
```

The package builder creates `<team-name>_submission.zip` containing the
final `output/candidate_pairs.tsv` and `output/matching_results.tsv`, source,
scripts, utilities, `README.md`, `requirements.txt`, and
`Documentation_template.md` under the requested layout. It excludes caches,
experiment artifacts, virtual environments, and secrets. Run the pipeline first
so the two final output files exist.
