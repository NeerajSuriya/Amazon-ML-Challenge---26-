#!/usr/bin/env python3
"""Build a clean challenge submission archive with only final outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_archive(team_name: str, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="submission-package-") as temp_dir:
        staging = Path(temp_dir) / "staging"
        package = staging / "code" / "business_entity_resolution"
        for relative in ("src", "scripts", "utils"):
            source_dir = PROJECT_ROOT / relative
            if source_dir.is_dir():
                shutil.copytree(
                    source_dir,
                    package / relative,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
        for filename in ("README.md", "requirements.txt"):
            shutil.copy2(PROJECT_ROOT / filename, package / filename)
        shutil.copy2(PROJECT_ROOT / "Documentation_template.md", staging / "Documentation_template.md")

        final_output = staging / "output"
        final_output.mkdir(parents=True, exist_ok=True)
        for filename in ("candidate_pairs.tsv", "matching_results.tsv"):
            source = PROJECT_ROOT / "output" / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing final output {source}; run the submission pipeline first"
                )
            shutil.copy2(source, final_output / filename)

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", default="team")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT_ROOT / f"{args.team_name}_submission.zip"
    archive = build_archive(args.team_name, output)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
