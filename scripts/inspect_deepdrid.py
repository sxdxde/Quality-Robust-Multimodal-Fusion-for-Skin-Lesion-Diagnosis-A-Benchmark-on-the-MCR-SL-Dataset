"""STEP 0 schema inspection for the DeepDRiD cross-dataset gate.

Answers the three questions the gate asks, by READING THE ACTUAL FILES rather
than assuming anything about DeepDRiD's schema:

  1. Exact column names + value ranges for the DR severity grade.
  2. Exact column name(s), range, and type for image quality. Sub-challenge B
     mentions "overall image quality, artifacts, clarity, field definition",
     which may be several sub-scores rather than one — this dumps all of them
     so the choice of signal is made from real values, not a guess.
  3. Whether Online-Challenge1&2-Evaluation carries usable labels, or whether
     train+validation is all there is.

Reads .docx with the standard library only (a .docx is a zip containing
word/document.xml) — deliberately avoids `pip install python-docx` into the
shared `brats` conda env, per the project's do-not-touch-the-shared-env rule.

Usage:
    python scripts/inspect_deepdrid.py [--root ~/deepdrid]
"""
import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

# Substrings that suggest a column is a grade or a quality score. Used only to
# HIGHLIGHT columns for closer inspection — every column is dumped regardless,
# so a differently-named field cannot be silently missed.
GRADE_HINTS = ["dr", "grade", "level", "severity", "retinopathy", "dme"]
QUALITY_HINTS = ["quality", "artifact", "clarity", "field", "definition", "overall", "usable"]


def docx_text(path: Path) -> str:
    """Extract readable text from a .docx without python-docx."""
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as e:
        return f"<could not read {path.name}: {e}>"
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    lines = [ln.strip() for ln in xml.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def describe_column(df: pd.DataFrame, col: str, max_uniques: int = 12) -> str:
    s = df[col]
    n_null = int(s.isna().sum())
    nun = int(s.nunique(dropna=True))
    out = [f"      dtype={s.dtype}  nulls={n_null}/{len(s)}  distinct={nun}"]
    if nun <= max_uniques:
        vc = s.value_counts(dropna=False).sort_index()
        # plain python scalars — numpy reprs make these tables unreadable
        pretty = {(k.item() if hasattr(k, "item") else k): int(v) for k, v in vc.items()}
        out.append(f"      values: {pretty}")
    else:
        try:
            out.append(f"      range: min={s.min()}  max={s.max()}  mean={s.mean():.3f}")
        except TypeError:
            out.append(f"      sample: {s.dropna().unique()[:6].tolist()}")
    return "\n".join(out)


def inspect_table(path: Path, root: Path):
    print(f"\n{'-' * 78}")
    print(f"TABLE: {path.relative_to(root)}")
    print(f"{'-' * 78}")
    try:
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    except Exception as e:
        print(f"  could not parse: {e}")
        return

    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")

    lower = {c: str(c).lower() for c in df.columns}
    grade_cols = [c for c, lc in lower.items() if any(h in lc for h in GRADE_HINTS)]
    quality_cols = [c for c, lc in lower.items() if any(h in lc for h in QUALITY_HINTS)]

    if grade_cols:
        print("\n  >>> candidate DR-GRADE columns:")
        for c in grade_cols:
            print(f"    {c}:")
            print(describe_column(df, c))

    if quality_cols:
        print("\n  >>> candidate IMAGE-QUALITY columns:")
        for c in quality_cols:
            print(f"    {c}:")
            print(describe_column(df, c))

    other = [c for c in df.columns if c not in grade_cols and c not in quality_cols]
    if other:
        print("\n  other columns (dumped so nothing is missed by the keyword filter):")
        for c in other:
            print(f"    {c}:")
            print(describe_column(df, c))

    print("\n  first 3 rows:")
    print("   " + df.head(3).to_string().replace("\n", "\n   "))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "deepdrid")
    parser.add_argument("--readme-chars", type=int, default=3000,
                        help="how much of each README to print")
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise SystemExit(f"{root} does not exist — run scripts/download_deepdrid.sh first.")

    print("=" * 78)
    print(f"DeepDRiD Step-0 schema inspection — {root}")
    print("=" * 78)

    # --- READMEs ---------------------------------------------------------
    docs = sorted(root.rglob("*.docx"))
    docs = [d for d in docs if ".git" not in d.parts and not d.name.startswith("~$")]
    print(f"\n### {len(docs)} .docx file(s) found")
    for d in docs:
        print(f"\n{'=' * 78}")
        print(f"README: {d.relative_to(root)}")
        print("=" * 78)
        txt = docx_text(d)
        print(txt[:args.readme_chars])
        if len(txt) > args.readme_chars:
            print(f"\n... [{len(txt) - args.readme_chars} more chars — "
                  f"rerun with --readme-chars {len(txt)} for the rest]")

    for pattern in ("*.md", "*.txt"):
        for f in sorted(root.rglob(pattern)):
            if ".git" in f.parts:
                continue
            print(f"\n{'=' * 78}")
            print(f"TEXT: {f.relative_to(root)}")
            print("=" * 78)
            print(f.read_text(errors="replace")[:args.readme_chars])

    # --- Tables ----------------------------------------------------------
    tables = [p for p in sorted(root.rglob("*"))
              if p.suffix.lower() in (".csv", ".xlsx", ".xls") and ".git" not in p.parts]
    print(f"\n\n{'#' * 78}")
    print(f"# {len(tables)} label table(s) found")
    print(f"{'#' * 78}")
    for t in tables:
        inspect_table(t, root)

    # --- Split / image census -------------------------------------------
    print(f"\n\n{'#' * 78}")
    print("# Split census — is Online-Challenge1&2-Evaluation usable as a third split?")
    print(f"{'#' * 78}")
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and ".git" not in p.parts):
        imgs = [f for f in d.iterdir() if f.is_file()
                and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs:
            real = sum(1 for f in imgs[:20]
                       if not f.read_bytes()[:100].startswith(b"version https://git-lfs"))
            note = "" if real else "   [ALL LFS POINTERS — no pixel data yet]"
            print(f"  {str(d.relative_to(root)):<60} {len(imgs):>6} images{note}")

    # --- Patient leakage between the provided splits ---------------------
    # This project's central protocol rule is subject-disjoint splitting.
    # DeepDRiD ships its own train/validation split; if that split puts the
    # same patient on both sides, the comparison would be leak-contaminated
    # in exactly the way MCR-SL's protocol was built to prevent. Cheap to
    # check, and gate-relevant if it fails.
    print(f"\n\n{'#' * 78}")
    print("# Patient overlap between the provided train and validation splits")
    print(f"{'#' * 78}")
    id_cols = ["patient_id", "patient", "Patient_ID", "subject_id", "id"]
    split_ids = {}
    for t in tables:
        name = str(t.relative_to(root)).lower()
        split = "train" if "train" in name else ("val" if ("val" in name or "valid" in name) else None)
        if split is None:
            continue
        try:
            df = pd.read_csv(t) if t.suffix.lower() == ".csv" else pd.read_excel(t)
        except Exception:
            continue
        col = next((c for c in df.columns if str(c) in id_cols
                    or str(c).lower() in [i.lower() for i in id_cols]), None)
        if col is not None:
            split_ids.setdefault(split, set()).update(df[col].dropna().astype(str))

    if "train" in split_ids and "val" in split_ids:
        overlap = split_ids["train"] & split_ids["val"]
        print(f"  train patients: {len(split_ids['train'])}")
        print(f"  val patients:   {len(split_ids['val'])}")
        print(f"  OVERLAP:        {len(overlap)}")
        if overlap:
            print(f"  *** {len(overlap)} patient(s) appear in BOTH splits: "
                  f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}")
            print("  *** The provided split is NOT patient-disjoint. Either regroup by")
            print("  *** patient before training, or treat this as a gate finding.")
        else:
            print("  -> provided split is patient-disjoint. Safe to use as-is.")
    else:
        print("  Could not identify train/val label tables or a patient-id column.")
        print("  Check the column dumps above and verify patient-disjointness by hand")
        print("  before training — do not assume the shipped split is clean.")

    print(f"\n{'=' * 78}")
    print("GATE CHECKLIST — answer these before writing any adapter code:")
    print("=" * 78)
    print("""
  [ ] 1. DR grade column name + exact value set?           (see candidate DR-GRADE above)
  [ ] 2. Quality column(s): one score or several?          (see candidate IMAGE-QUALITY above)
         If several, pick ONE or a simple stated combination -- justify it, don't guess.
         Note its type (ordinal categorical vs continuous) and its full range.
  [ ] 3. Is Online-Challenge1&2-Evaluation LABELLED?       (a split with no labels is unusable)
  [ ] 4. Is the provided train/val split patient-disjoint?  (see overlap check above)
  [ ] 5. Total download size, actually measured?
  [ ] 6. Did all of Step 0 fit inside the 2-hour box?

  If 1-4 cannot be answered from real files, or 5-6 blow the box:
  GATE FAILS -> stop, report plainly, use the future-work framing. That is the
  designed outcome of a gate, not a failure to apologise for.
""")


if __name__ == "__main__":
    main()
