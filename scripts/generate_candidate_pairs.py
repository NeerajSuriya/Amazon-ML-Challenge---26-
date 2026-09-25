#!/usr/bin/env python3
"""Generate a deterministic candidate_pairs.tsv from processed source TSVs.

The repository does not contain an authoritative submission-schema document.
The default writer therefore emits the two identity columns consumed by the
pair-feature stage. ``--include-metadata`` adds blocker provenance for local
analysis only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    BLOCKING_RULES,
    CANDIDATE_ID,
    CANDIDATE_SOURCE,
    DEFAULT_RULE_NAMES,
    SOURCE1_ID,
    BlockingConfig,
    CandidateBlocker,
)

IDENTITY_COLUMNS = [SOURCE1_ID, CANDIDATE_ID]
METADATA_COLUMNS = [CANDIDATE_SOURCE, BLOCKING_RULES]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _path_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    explicit = [args.source1, args.source2, args.source3]
    if any(path is not None for path in explicit):
        if not all(path is not None for path in explicit):
            parser.error("--source1, --source2, and --source3 must be supplied together")
        return tuple(path for path in explicit)  # type: ignore[return-value]
    root = args.processed_root
    return (
        root / args.source1_name,
        root / args.source2_name,
        root / args.source3_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "processed" / "train")
    parser.add_argument("--source1", type=Path)
    parser.add_argument("--source2", type=Path)
    parser.add_argument("--source3", type=Path)
    parser.add_argument("--source1-name", default="train_source1.tsv")
    parser.add_argument("--source2-name", default="train_source2.tsv")
    parser.add_argument("--source3-name", default="train_source3.tsv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "candidate_pairs.tsv")
    parser.add_argument("--rules", help="Comma-separated rule names; default uses the configured default union")
    parser.add_argument("--name-token-max-frequency", type=int, default=50)
    parser.add_argument("--address-token-max-frequency", type=int, default=50)
    parser.add_argument("--max-exact-bucket-size", type=int, default=500)
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include candidate_source and blocking_rules after the required identity columns",
    )
    args = parser.parse_args()

    source1_path, source2_path, source3_path = _path_args(parser, args)
    started = time.perf_counter()
    source1, source2, source3 = map(_read, (source1_path, source2_path, source3_path))
    config = BlockingConfig(
        name_token_max_frequency=args.name_token_max_frequency,
        address_token_max_frequency=args.address_token_max_frequency,
        max_exact_bucket_size=args.max_exact_bucket_size,
    )
    blocker = CandidateBlocker(config).fit(source1, source2, source3)
    rules = tuple(part.strip() for part in args.rules.split(",") if part.strip()) if args.rules else None
    candidates = blocker.generate(rules)

    identity = candidates[IDENTITY_COLUMNS].copy()
    if identity.duplicated().any():
        raise RuntimeError("Blocking produced duplicate candidate identity pairs")
    if identity[SOURCE1_ID].eq("").any() or identity[CANDIDATE_ID].eq("").any():
        raise RuntimeError("Blocking produced an empty entity ID")
    if not identity[CANDIDATE_ID].str.startswith(("S2-", "S3-"), na=False).all():
        raise RuntimeError("Blocking produced a candidate outside Source 2/Source 3")

    output = candidates[IDENTITY_COLUMNS + (METADATA_COLUMNS if args.include_metadata else [])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(output):,} candidate pairs to {args.output}")
    print(f"Rules: {'|'.join(rules or DEFAULT_RULE_NAMES)}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")
    print(f"Peak RSS MB: {_rss_mb():.1f}")


if __name__ == "__main__":
    main()
