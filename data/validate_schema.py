"""Schema-validation script. Run this FIRST on the remote, before any
training code, against the extracted MCR-SL CSVs.

Usage:
    python data/validate_schema.py --data-dir ~/mcrsl_project/data/raw/extracted

For each table name in data/schema.py's ASSUMED_TABLES, tries to find a
matching CSV/XLSX by filename (case-insensitive substring match on the table
name), then diffs its columns/dtypes against the assumed schema. Fails loudly
and prints the diff — never silently coerces or drops columns.

This does NOT try to be clever about locating files if the naming doesn't
match at all; if auto-discovery fails for a table, it prints the list of
available files so you can wire up the mapping by hand (see FILE_OVERRIDES).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from data.schema import ASSUMED_TABLES

# Fill in explicit filename overrides here once real filenames are known,
# e.g. {"lesion": "lesion_table.csv"}. Auto-discovery below is a fallback.
FILE_OVERRIDES: dict[str, str] = {}


def find_table_file(data_dir: Path, table_name: str, all_files: list[Path]) -> Path | None:
    if table_name in FILE_OVERRIDES:
        p = data_dir / FILE_OVERRIDES[table_name]
        return p if p.exists() else None
    candidates = [f for f in all_files if table_name.replace("_", "") in f.stem.lower().replace("_", "").replace("-", "")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  ! multiple candidate files for table '{table_name}': {[c.name for c in candidates]}")
        print(f"    -> add an explicit entry to FILE_OVERRIDES in this script")
    return None


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"unsupported file type: {path}")


def validate_table(table_name: str, expected_cols: dict, df: pd.DataFrame) -> list[str]:
    problems = []
    actual_cols = set(df.columns)
    expected_col_names = set(expected_cols.keys())

    missing = expected_col_names - actual_cols
    extra = actual_cols - expected_col_names
    if missing:
        problems.append(f"MISSING expected columns: {sorted(missing)}")
    if extra:
        problems.append(f"UNEXPECTED columns present (not in schema.py, may need to add): {sorted(extra)}")

    for col, expected_kind in expected_cols.items():
        if col not in actual_cols or expected_kind is None:
            continue
        actual_kind = df[col].dtype.kind
        if actual_kind != expected_kind:
            problems.append(
                f"dtype mismatch on '{col}': expected kind '{expected_kind}', got '{actual_kind}' ({df[col].dtype})"
            )
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data dir does not exist: {args.data_dir}")
        sys.exit(1)

    all_files = list(args.data_dir.rglob("*.csv")) + list(args.data_dir.rglob("*.xlsx")) + list(args.data_dir.rglob("*.xls"))
    print(f"Found {len(all_files)} candidate table files under {args.data_dir}:")
    for f in all_files:
        print(f"  {f.relative_to(args.data_dir)}")
    print()

    any_failures = False
    for table_name, expected_cols in ASSUMED_TABLES.items():
        print(f"--- {table_name} ---")
        path = find_table_file(args.data_dir, table_name, all_files)
        if path is None:
            print(f"  ! could not locate a file for table '{table_name}' — skipping validation, fix FILE_OVERRIDES")
            any_failures = True
            continue
        df = load_table(path)
        print(f"  file: {path.relative_to(args.data_dir)}  shape={df.shape}")
        problems = validate_table(table_name, expected_cols, df)
        if problems:
            any_failures = True
            for p in problems:
                print(f"  ! {p}")
        else:
            print("  OK — matches assumed schema")
        print()

    if any_failures:
        print("SCHEMA VALIDATION FAILED — fix data/schema.py to match the real files above, "
              "then re-run before writing/using any data loaders.")
        sys.exit(1)
    print("SCHEMA VALIDATION PASSED.")


if __name__ == "__main__":
    main()
