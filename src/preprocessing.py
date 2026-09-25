"""
Business Entity Resolution — preprocessing.py

Purpose
-------
Add normalization/derived representations WITHOUT modifying or deleting the
original fields.

Input columns expected:
    entity_id
    business_name
    business_address
    country

Output keeps the raw columns and adds:
    name_norm
    address_norm
    country_norm
    name_tokens
    address_tokens
    name_char_ngrams
    address_char_ngrams
    source

The normalization is deliberately conservative. It is intended to make noisy
records easier to compare while retaining the original values for later
feature engineering and debugging.

Challenge-specific constraints:
- TSV input/output (sep="\\t")
- country is treated as an open-set string; no hard-coded country list
- no external lookup/geocoding/enrichment
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# Conservative canonicalization rules.
# Keep the canonical forms as ordinary tokens so that token overlap remains useful.
NAME_REPLACEMENTS = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "corp": "corporation",
    "corp.": "corporation",
    "co": "company",
    "co.": "company",
    "inc": "incorporated",
    "inc.": "incorporated",
    "llc": "llc",
    "llp": "llp",
}

ADDRESS_REPLACEMENTS = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ste": "suite",
    "ste.": "suite",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "av.": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "hwy": "highway",
    "hwy.": "highway",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "apt": "apartment",
    "apt.": "apartment",
    "no": "number",
    "no.": "number",
}


def _safe_text(value) -> str:
    """Convert missing/non-string values to a clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _unicode_normalize(text: str) -> str:
    """Normalize Unicode compatibility forms and case-fold."""
    text = unicodedata.normalize("NFKC", text)
    return text.casefold()


def _replace_ampersand(text: str) -> str:
    return re.sub(r"\s*&\s*", " and ", text)


def _apply_token_replacements(text: str, replacements: dict[str, str]) -> str:
    """
    Apply word-level replacements before punctuation is removed.
    Sorting by length prevents shorter rules from interfering with longer ones.
    """
    if not text:
        return ""

    # Tokenize broadly while retaining periods temporarily.
    tokens = re.findall(r"[^\s]+", text)
    output = []

    for token in tokens:
        cleaned = token.strip()
        key = cleaned.lower()
        output.append(replacements.get(key, cleaned))

    return " ".join(output)


def _punctuation_to_space(text: str) -> str:
    """
    Convert punctuation/symbols to spaces, while retaining letters/numbers.
    This is intentionally not ASCII-only: non-Latin business names/addresses
    remain usable.
    """
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith(("L", "N","M")):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value) -> str:
    """
    Conservative business-name normalization.

    Examples:
        "ABC Pvt. Ltd." -> "abc private limited"
        "A.B.C. & Co." -> "a b c and company"
    """
    text = _safe_text(value)
    if not text:
        return ""

    text = _unicode_normalize(text)
    text = _replace_ampersand(text)
    text = _apply_token_replacements(text, NAME_REPLACEMENTS)
    text = _punctuation_to_space(text)
    return _collapse_whitespace(text)


def normalize_address(value) -> str:
    """
    Conservative address normalization.

    Examples:
        "12 MG Rd., Bangalore" -> "12 mg road bangalore"
        "12 St. John St." -> "12 street john street"
    """
    text = _safe_text(value)
    if not text:
        return ""

    text = _unicode_normalize(text)
    text = _replace_ampersand(text)
    text = _apply_token_replacements(text, ADDRESS_REPLACEMENTS)
    text = _punctuation_to_space(text)
    return _collapse_whitespace(text)


def normalize_country(value) -> str:
    """
    Country is intentionally treated as an open-set string.
    Do not restrict this to US/India because the test set contains France.
    """
    text = _safe_text(value)
    if not text:
        return ""

    text = _unicode_normalize(text)
    text = _punctuation_to_space(text)
    return _collapse_whitespace(text)


def tokenize(text: str) -> list[str]:
    """
    Whitespace tokenization after normalization.

    Empty strings become [].
    """
    if not text:
        return []
    return text.split()


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """
    Generate character n-grams with boundary markers.

    Boundary markers make prefixes/suffixes informative:
        "starbucks" -> "^st", "sta", ..., "ks$"

    Spaces are retained because word-boundary/spacing information can be useful.
    """
    if not text:
        return []

    padded = f"^{text}$"
    if len(padded) < n:
        return [padded]

    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def _json_list(values: Iterable[str]) -> str:
    """Serialize list-valued features safely for TSV persistence."""
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    ngram_size: int = 3,
    include_char_ngrams: bool = True,
) -> pd.DataFrame:
    """
    Preprocess one pandas DataFrame.

    Raw columns are copied first and never overwritten.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Raw columns are preserved exactly as supplied.
    # Add normalized representations beside them.
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_country)

    out["name_tokens"] = out["name_norm"].map(tokenize)
    out["address_tokens"] = out["address_norm"].map(tokenize)

    if include_char_ngrams:
        out["name_char_ngrams"] = out["name_norm"].map(
            lambda x: char_ngrams(x, n=ngram_size)
        )
        out["address_char_ngrams"] = out["address_norm"].map(
            lambda x: char_ngrams(x, n=ngram_size)
        )
    else:
        out["name_char_ngrams"] = [[] for _ in range(len(out))]
        out["address_char_ngrams"] = [[] for _ in range(len(out))]

    # Source is derived only from the documented entity_id prefix.
    out["source"] = (
        out["entity_id"]
        .astype("string")
        .str.extract(r"^(S[123])-", expand=False)
        .fillna("")
    )

    return out


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ngram_size: int = 3,
    chunksize: int | None = None,
    include_char_ngrams: bool = True,
) -> None:
    """
    Preprocess a TSV file.

    If chunksize is supplied, process the file in chunks so large files do not
    need to fit completely in RAM.

    List-valued columns are JSON-serialized in the output TSV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )

    first = True

    for chunk in reader:
        processed = preprocess_dataframe(
            chunk,
            ngram_size=ngram_size,
            include_char_ngrams=include_char_ngrams,
        )

        # JSON is used only for persistent TSV representation.
        for col in [
            "name_tokens",
            "address_tokens",
            "name_char_ngrams",
            "address_char_ngrams",
        ]:
            processed[col] = processed[col].map(_json_list)

        processed.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False


def preprocess_dataset_tree(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    ngram_size: int = 3,
    chunksize: int = 50_000,
) -> None:
    """
    Convenience function for the challenge directory structure:

        dataset/
          train/
            train_source1.tsv
            train_source2.tsv
            train_source3.tsv
          test/
            test_source1.tsv
            test_source2.tsv
            test_source3.tsv

    Output:

        processed/
          train/...
          test/...
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    for split in ("train", "test"):
        input_dir = dataset_root / split
        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(input_dir.glob("*.tsv")):
            output_path = output_dir / path.name
            print(f"[preprocess] {path} -> {output_path}")
            preprocess_file(
                path,
                output_path,
                ngram_size=ngram_size,
                chunksize=chunksize,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize business entity-resolution TSV data without destroying raw fields."
    )
    parser.add_argument("--input", help="Input TSV file.")
    parser.add_argument("--output", help="Output TSV file.")
    parser.add_argument(
        "--dataset-root",
        help="Process dataset/train and dataset/test trees in one command.",
    )
    parser.add_argument(
        "--output-root",
        default="processed",
        help="Output root when --dataset-root is used (default: processed).",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=3,
        choices=[2, 3, 4, 5],
        help="Character n-gram size (default: 3).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Rows per chunk for large TSV files (default: 50000).",
    )

    args = parser.parse_args()

    if args.dataset_root:
        preprocess_dataset_tree(
            args.dataset_root,
            args.output_root,
            ngram_size=args.ngram_size,
            chunksize=args.chunksize,
        )
        return

    if not args.input or not args.output:
        parser.error("Use either --dataset-root or both --input and --output.")

    preprocess_file(
        args.input,
        args.output,
        ngram_size=args.ngram_size,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
