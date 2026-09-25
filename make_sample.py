import pandas as pd
from pathlib import Path

BASE = Path("dataset/train")
OUT = Path("dataset/sample")
OUT.mkdir(parents=True, exist_ok=True)

N_S1 = 3000
RANDOM_STATE = 42

print("Loading S1...")
s1 = pd.read_csv(
    BASE / "train_source1.tsv",
    sep="\t",
    dtype=str
)

print("Loading ground truth...")
gt = pd.read_csv(
    BASE / "train_ground_truth.tsv",
    sep="\t",
    dtype=str
)

# ---------------------------------------------------------
# 1. Randomly select 3,000 S1 entities
# ---------------------------------------------------------

sample_s1 = s1.sample(
    n=N_S1,
    random_state=RANDOM_STATE
).copy()

selected_ids = set(sample_s1["entity_id"])

# ---------------------------------------------------------
# 2. Get corresponding ground truth
# ---------------------------------------------------------

sample_gt = gt[
    gt["source1_entity_id"].isin(selected_ids)
].copy()

# ---------------------------------------------------------
# 3. Extract every S2/S3 ID referenced by the GT
# ---------------------------------------------------------

s2_ids = set()
s3_ids = set()

for value in sample_gt["matched_entity_ids"].fillna(""):
    if not value:
        continue

    for entity_id in value.split(","):
        entity_id = entity_id.strip()

        if not entity_id:
            continue

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)

# ---------------------------------------------------------
# 4. Load S2/S3 and keep relevant records
# ---------------------------------------------------------

print("Loading S2...")
s2 = pd.read_csv(
    BASE / "train_source2.tsv",
    sep="\t",
    dtype=str
)

print("Loading S3...")
s3 = pd.read_csv(
    BASE / "train_source3.tsv",
    sep="\t",
    dtype=str
)

sample_s2 = s2[s2["entity_id"].isin(s2_ids)].copy()
sample_s3 = s3[s3["entity_id"].isin(s3_ids)].copy()

# ---------------------------------------------------------
# 5. Sort for reproducibility
# ---------------------------------------------------------

sample_s1 = sample_s1.sort_values("entity_id")
sample_s2 = sample_s2.sort_values("entity_id")
sample_s3 = sample_s3.sort_values("entity_id")
sample_gt = sample_gt.sort_values("source1_entity_id")

# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------

sample_s1.to_csv(
    OUT / "train_source1_sample.tsv",
    sep="\t",
    index=False
)

sample_s2.to_csv(
    OUT / "train_source2_sample.tsv",
    sep="\t",
    index=False
)

sample_s3.to_csv(
    OUT / "train_source3_sample.tsv",
    sep="\t",
    index=False
)

sample_gt.to_csv(
    OUT / "train_ground_truth_sample.tsv",
    sep="\t",
    index=False
)

# ---------------------------------------------------------
# 7. Report
# ---------------------------------------------------------

print("\n==============================")
print("SAMPLE CREATED")
print("==============================")

print(f"S1: {len(sample_s1):,}")
print(f"S2: {len(sample_s2):,}")
print(f"S3: {len(sample_s3):,}")
print(f"GT: {len(sample_gt):,}")

print(f"\nS1 with matches: {(sample_gt['matched_entity_ids'].fillna('').str.len() > 0).sum():,}")
print(f"S1 singletons:  {(sample_gt['matched_entity_ids'].fillna('').str.len() == 0).sum():,}")

print("\nSaved to:")
print(OUT)