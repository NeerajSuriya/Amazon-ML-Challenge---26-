# Amazon Business Entity Resolution Submission

## Team

- Team name: `<team-name>`
- Model: `src/matcher.py::ProductionMatcher`
- Threshold: `0.85`

## Reproduction

From the repository root:

```bash
python scripts/run_submission_pipeline.py \
  --dataset-root dataset \
  --output-dir output
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

For the checked-in local sample, use the sample filenames explicitly:

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

## Output contract

`output/candidate_pairs.tsv` and `output/matching_results.tsv` each contain one
row per test Source-1 entity. Candidate and matched IDs are comma-separated,
empty cells are valid, and matching IDs are a subset of final candidates.
Matching is zero-to-many: no one-to-one assignment is applied.
