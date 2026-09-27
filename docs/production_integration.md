# Frozen production matcher integration

`src/matcher.py` is the frozen production model. This integration layer does
not modify it, the blocking rules, preprocessing, pair features, metrics, or
the 20-feature schema.

## Supported command

The integration script trains the frozen matcher on labeled blocked candidates,
then scores a separate (or explicitly identical) processed source set. All
paths are required; the repository does not contain an authoritative full-data
path or competition submission schema.

```bash
myenv/bin/python scripts/run_production_inference.py \
  --train-source1 /data/processed/train_source1.tsv \
  --train-source2 /data/processed/train_source2.tsv \
  --train-source3 /data/processed/train_source3.tsv \
  --ground-truth /data/train_ground_truth.tsv \
  --inference-source1 /data/processed/test_source1.tsv \
  --inference-source2 /data/processed/test_source2.tsv \
  --inference-source3 /data/processed/test_source3.tsv \
  --scored-output /data/output/scored_candidates.tsv \
  --matches-output /data/output/thresholded_matches.tsv
```

The default threshold is the frozen `MATCH_THRESHOLD = 0.85`; it can be
changed explicitly for controlled local analysis with `--threshold`, but this
script does not tune it from held-out or test labels.

## Input contracts

The six source files must be processed TSVs containing at least:

```text
entity_id
name_norm
address_norm
country_norm
name_tokens
address_tokens
name_char_ngrams
address_char_ngrams
```

List-valued fields use the existing JSON-list representation. The optional
`source` column is validated by the existing blocker and otherwise derived from
the `S1-`, `S2-`, or `S3-` entity-ID prefix.

The ground-truth TSV must contain:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` follows the existing zero-to-many representation. An
empty value means that the S1 entity has no known matches.

## Execution flow

1. Generate training and inference candidates with the existing default
   `CandidateBlocker` configuration.
2. Reject duplicate or empty candidate identities and invalid S1/S2/S3
   namespaces.
3. Fit `PairFeatureGenerator(include_experimental=False)` over the combined
   processed records so its TF-IDF views can transform both populations.
4. Generate the exact 20 baseline features in the frozen matcher order.
5. Attach training labels with the existing `attach_ground_truth_labels`
   utility.
6. Fit `ProductionMatcher` and score every inference candidate independently.
7. Apply `match_probability >= 0.85` by default.

No one-to-one assignment is performed. Multiple S2 and/or S3 candidates may
pass for one S1 entity, and candidate row order is preserved through feature
generation and scoring.

## Outputs

`--scored-output` contains one row for every inference candidate:

```text
source1_entity_id
candidate_entity_id
match_probability
```

`--matches-output` contains the same columns, filtered to rows whose
probability meets the selected threshold. These are local integration
artifacts. The repository does not document an official competition
`matching_results.tsv` or submission schema, so the thresholded file must not
be treated as that schema without an external specification.

The script prints QA counts for training candidates, feature rows, inference
candidates, matcher outputs, selected matches, threshold, and output schemas.
It fails rather than fabricating data when required files or columns are
missing, labels have only one class, IDs are invalid, feature rows are
misaligned, or probabilities fall outside `[0, 1]`.
