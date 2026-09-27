#!/usr/bin/env python3
"""Validate aggregated candidate and matching submission TSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.submission import validate_submission_frames  # noqa: E402


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--test-source1-name", default="test_source1.tsv")
    parser.add_argument("--test-source2-name", default="test_source2.tsv")
    parser.add_argument("--test-source3-name", default="test_source3.tsv")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        test_dir = Path(args.test_dir)
        source1 = _read(test_dir / args.test_source1_name, "test source1")
        source2 = _read(test_dir / args.test_source2_name, "test source2")
        source3 = _read(test_dir / args.test_source3_name, "test source3")
        for frame, label in ((source1, "test source1"), (source2, "test source2"), (source3, "test source3")):
            if "entity_id" not in frame.columns:
                raise ValueError(f"{label} is missing required entity_id column")
        summary = validate_submission_frames(
            _read(args.candidate, "candidate output"),
            _read(args.matching, "matching output"),
            test_source1_ids=source1["entity_id"].tolist(),
            test_source2_ids=source2["entity_id"].tolist(),
            test_source3_ids=source3["entity_id"].tolist(),
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("VALIDATION PASSED")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
