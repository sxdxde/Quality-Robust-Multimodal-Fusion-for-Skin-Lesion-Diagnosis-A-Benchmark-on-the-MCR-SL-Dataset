"""Robustness analyses 5 and 6 — post-hoc, no training, no GPU needed.

Both run on out-of-fold predictions that already exist (the same
`results/oof_predictions_<cfg_tag>.csv` the existing analyses 1/3/4 use), so
this costs nothing beyond reading two spreadsheets.

ANALYSIS 5 — Diagnostic certainty.
  `dermatology_diagnosis.certainty` is each expert's self-reported confidence
  in their own diagnosis (0/25/50/75/100), which is a DIFFERENT axis from
  `image_rating` (photo quality). Asks whether the model's errors track
  diagnostic difficulty the same way they track photo quality — two distinct
  questions that may well give different answers.

  Verification-first: the E002 data loss documented for `image_rating`
  (0/241 non-null) was described by the dataset authors as specific to image
  QUALITY ratings. This script does not assume certainty survived it — it
  counts valid certainty values per expert and reports what it finds, then
  averages over whichever experts actually have them.

ANALYSIS 6 — Intra-subject consistency.
  MCR-SL deliberately collected >=2 lesions per subject. Asks whether errors
  cluster within subjects ("hard subjects") or scatter independently across
  lesions. Scope-checks the multi-lesion subject count FIRST and downgrades
  to descriptive-only reporting when that count is small, matching how the
  n=28 histopathology subset is already handled in this project.

Usage:
    python scripts/robustness_analyses_5_6.py \
        --data-dir ~/mcrsl_project/data/raw/extracted/MCR-SL_dataset

Outputs (under results/):
    robustness_certainty_tercile_<cfg_tag>.csv / .png
    robustness_certainty_vs_quality_<cfg_tag>.csv
    robustness_intra_subject_<cfg_tag>.csv / .png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import build_lesion_table, load_raw_tables

# Small-N thresholds, fixed up front so the reporting standard isn't chosen
# after seeing the result. The task spec names ~10 subjects as the point
# below which analysis 6 is qualitative-only.
MIN_SUBJECTS_FOR_INFERENCE = 10
N_PERMUTATIONS = 10000


def parse_certainty(value) -> float:
    """Schema says int (0/25/50/75/100); tolerate '75%'-style strings and
    genuine missing values rather than coercing silently."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        v = value.strip().rstrip("%").strip()
        if v == "" or v.lower() == "unknown":
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if np.isnan(f) else f


def terciles_or_fewer(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """qcut into 3 equal-frequency buckets, degrading honestly when the value
    distribution is too concentrated to support 3 distinct edges.

    `certainty` is a 5-value discrete field averaged over 3-4 experts, so a
    concentrated distribution (most experts answering 100%) is a real
    possibility. When qcut can't form 3 buckets, grouping by the distinct
    values themselves is far more informative than collapsing everything into
    one bucket — so that's the fallback, provided there aren't too many.
    """
    try:
        binned = pd.qcut(series, 3, labels=["low", "mid", "high"])
        return binned, ["low", "mid", "high"]
    except ValueError:
        pass

    distinct = sorted(series.dropna().unique())
    if len(distinct) <= 8:
        print(f"  NOTE: could not form 3 equal-frequency buckets — the distribution is too "
              f"concentrated. Grouping by the {len(distinct)} distinct value(s) instead, which "
              f"is more informative for a discrete field than one merged bucket.")
        labels = [f"{v:g}" for v in distinct]
        return series.map(lambda v: f"{v:g}" if pd.notna(v) else v), labels

    binned = pd.qcut(series, 3, duplicates="drop")
    cats = [str(c) for c in binned.cat.categories]
    print(f"  NOTE: could not form 3 equal-frequency buckets; fell back to {len(cats)} bucket(s).")
    return binned.astype(str), cats


def stratified_table(df: pd.DataFrame, bucket_col: str, bucket_order: list[str]) -> pd.DataFrame:
    rows = []
    for b in bucket_order:
        sub = df[df[bucket_col] == b]
        malignant = sub[sub["binary_label"] == 1]
        rows.append({
            "bucket": b,
            "n": len(sub),
            "n_malignant": len(malignant),
            "accuracy": sub["correct"].mean() if len(sub) else float("nan"),
            "sensitivity_malignant": malignant["correct"].mean() if len(malignant) else float("nan"),
        })
    return pd.DataFrame(rows)


def analysis_5_certainty(oof: pd.DataFrame, lesion_df: pd.DataFrame, derm_df: pd.DataFrame,
                          results_dir: Path, cfg_tag: str) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("ANALYSIS 5 — diagnostic certainty (distinct axis from image quality)")
    print("=" * 78)

    derm = derm_df.copy()
    derm["certainty_parsed"] = derm["certainty"].apply(parse_certainty)

    print("\nStep 1 — per-expert certainty coverage (does the E002 image_rating loss apply here?):")
    coverage = derm.groupby("expert_id").agg(
        n_rows=("certainty_parsed", "size"),
        n_valid=("certainty_parsed", lambda s: int(s.notna().sum())),
    ).reset_index()
    coverage["pct_valid"] = (coverage["n_valid"] / coverage["n_rows"] * 100).round(1)
    print(coverage.to_string(index=False))

    valid_experts = sorted(coverage[coverage["n_valid"] > 0]["expert_id"].tolist())
    print(f"\nexperts with usable certainty: {valid_experts}")
    if "E002" in valid_experts:
        print("  -> E002 HAS valid certainty despite its image_rating being entirely lost.")
        print("     The documented data loss was specific to image-quality ratings, as the")
        print("     dataset paper states. Averaging certainty over all 4 experts.")
    else:
        print("  -> E002 certainty is ALSO missing — the loss was broader than the dataset")
        print("     paper describes. Averaging over the remaining experts only.")
    print(f"\nobserved distinct certainty values: "
          f"{sorted(derm['certainty_parsed'].dropna().unique().tolist())}")

    # Step 2 — mean certainty per lesion, on the diagnosis image only (same
    # linkage as compute_mean_image_rating in data/loader.py).
    diag_pairs = set(zip(lesion_df["lesion_id"], lesion_df["diagnosis_image_id"]))
    is_diag = derm.apply(lambda r: (r["lesion_id"], r["image_id"]) in diag_pairs, axis=1)
    usable = derm[is_diag & derm["expert_id"].isin(valid_experts)]
    mean_certainty = usable.groupby("lesion_id")["certainty_parsed"].mean()
    print(f"\nStep 2 — lesions with a mean_certainty value: {mean_certainty.notna().sum()}")

    df = oof.merge(
        mean_certainty.rename("mean_certainty"), left_on="lesion_id", right_index=True, how="left",
    ).merge(
        lesion_df[["lesion_id", "mean_image_rating"]], on="lesion_id", how="left",
    ).dropna(subset=["mean_certainty"])

    n = len(df)
    print(f"\nStep 3 — N with both an out-of-fold prediction and a mean_certainty: {n}")
    df["error"] = 1 - df["correct"].astype(int)
    df["confidence"] = np.abs(df["pred_prob"] - 0.5) * 2

    df["bucket"], order = terciles_or_fewer(df["mean_certainty"])
    summary = stratified_table(df, "bucket", order)
    print("\nAccuracy / sensitivity by expert-certainty tercile:")
    print(summary.to_string(index=False))
    summary.to_csv(results_dir / f"robustness_certainty_tercile_{cfg_tag}.csv", index=False)

    rho_err, p_err = spearmanr(df["mean_certainty"], df["error"])
    rho_conf, p_conf = spearmanr(df["mean_certainty"], df["confidence"])
    print(f"\nSpearman(certainty, error)      = {rho_err:+.3f} (p={p_err:.3f}, N={n})")
    print(f"Spearman(certainty, confidence) = {rho_conf:+.3f} (p={p_conf:.3f}, N={n})")

    # Side-by-side with the image-quality axis — the actual point of analysis 5.
    q = df.dropna(subset=["mean_image_rating"])
    rho_q_err, p_q_err = spearmanr(q["mean_image_rating"], q["error"])
    rho_q_conf, p_q_conf = spearmanr(q["mean_image_rating"], q["confidence"])
    rho_cross, p_cross = spearmanr(q["mean_certainty"], q["mean_image_rating"])

    comparison = pd.DataFrame([
        {"axis": "expert certainty (diagnostic difficulty)", "n": n,
         "spearman_vs_error": rho_err, "p_error": p_err,
         "spearman_vs_confidence": rho_conf, "p_confidence": p_conf},
        {"axis": "image rating (photo quality)", "n": len(q),
         "spearman_vs_error": rho_q_err, "p_error": p_q_err,
         "spearman_vs_confidence": rho_q_conf, "p_confidence": p_q_conf},
    ])
    comparison.to_csv(results_dir / f"robustness_certainty_vs_quality_{cfg_tag}.csv", index=False)
    print("\nTwo axes side by side (the point of this analysis):")
    print(comparison.to_string(index=False))
    print(f"\nSpearman(certainty, image_rating) = {rho_cross:+.3f} (p={p_cross:.3f}, N={len(q)}) "
          f"— how far the two axes are measuring the same thing at all")

    # --- Controlling for class composition -------------------------------
    # Malignant lesions concentrate in the low-certainty buckets (experts are
    # less sure about the harder, more suspicious lesions), and the model is
    # weaker on malignant lesions generally. So a pooled certainty-vs-error
    # correlation is partly just picking up class mix. Recomputing within each
    # class separates "certainty tracks difficulty" from "low-certainty
    # buckets simply hold more malignant lesions".
    print("\nControlling for class composition (malignant lesions cluster at low certainty,")
    print("and the model is weaker on them — so the pooled correlation is confounded).")
    print("BOTH axes are controlled the same way: the paper's claim is a COMPARISON")
    print("between them, so controlling only one would make that comparison invalid.")
    strat_rows = []
    for axis_col, axis_name in [("mean_certainty", "certainty"), ("mean_image_rating", "image_rating")]:
        print(f"\n  {axis_name}:")
        for label, name in [(0.0, "non-malignant"), (1.0, "malignant")]:
            sub = df[(df["binary_label"] == label)].dropna(subset=[axis_col])
            n_sub = len(sub)
            n_err = int(sub["error"].sum())
            if n_sub < 3 or sub["error"].nunique() < 2 or sub[axis_col].nunique() < 2:
                print(f"    {name:<14} n={n_sub:<4} — too few/degenerate to correlate "
                      f"({n_err} errors); reported as-is, not computed")
                strat_rows.append({"axis": axis_name, "stratum": name, "n": n_sub,
                                   "n_errors": n_err, "spearman_vs_error": float("nan"),
                                   "p_error": float("nan")})
                continue
            r, p = spearmanr(sub[axis_col], sub["error"])
            flag = "significant" if p < 0.05 else "n.s."
            print(f"    {name:<14} n={n_sub:<4} errors={n_err:<4} "
                  f"Spearman = {r:+.3f} (p={p:.3f}, {flag})")
            strat_rows.append({"axis": axis_name, "stratum": name, "n": n_sub,
                               "n_errors": n_err, "spearman_vs_error": r, "p_error": p})

    strat = pd.DataFrame(strat_rows)
    strat.to_csv(results_dir / f"robustness_certainty_by_class_{cfg_tag}.csv", index=False)
    print("\n  -> If a relationship holds WITHIN each class, that axis is tracking something")
    print("     real. If it vanishes for both, the pooled results were class mix and neither")
    print("     axis reliably predicts model error at this N — a clean null, reportable as one.")

    # Class balance per bucket, so the confound is visible in the table itself.
    summary["pct_malignant"] = (summary["n_malignant"] / summary["n"] * 100).round(1)
    summary.to_csv(results_dir / f"robustness_certainty_tercile_{cfg_tag}.csv", index=False)
    print(f"\nClass balance per bucket (the confound, made explicit):")
    print(summary[["bucket", "n", "n_malignant", "pct_malignant", "accuracy",
                   "sensitivity_malignant"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(summary["bucket"].astype(str), summary["accuracy"], color="tab:purple")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Mean expert diagnostic certainty")
    ax.set_title(f"Accuracy by certainty tercile ({cfg_tag}, N={n})")
    fig.tight_layout()
    fig.savefig(results_dir / f"robustness_certainty_tercile_{cfg_tag}.png", dpi=150)
    plt.close(fig)

    return summary


def analysis_6_intra_subject(oof: pd.DataFrame, lesion_df: pd.DataFrame,
                              results_dir: Path, cfg_tag: str) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("ANALYSIS 6 — intra-subject consistency")
    print("=" * 78)

    df = oof.merge(lesion_df[["lesion_id", "subject_id"]], on="lesion_id", how="left")
    df["correct_int"] = df["correct"].astype(int)

    per_subject = df.groupby("subject_id").agg(
        n_lesions=("lesion_id", "size"), n_correct=("correct_int", "sum"),
    ).reset_index()
    per_subject["accuracy"] = per_subject["n_correct"] / per_subject["n_lesions"]

    multi = per_subject[per_subject["n_lesions"] >= 2].copy()
    print(f"\nStep 1 — scope check (reported BEFORE any further design):")
    print(f"  subjects with >=1 usable lesion : {len(per_subject)}")
    print(f"  subjects with >=2 usable lesions: {len(multi)}")
    print(f"  lesions belonging to those subjects: {int(multi['n_lesions'].sum())}")
    print(f"  lesions-per-subject among them: mean={multi['n_lesions'].mean():.2f}, "
          f"max={int(multi['n_lesions'].max())}")

    qualitative_only = len(multi) < MIN_SUBJECTS_FOR_INFERENCE
    if qualitative_only:
        print(f"\n  -> {len(multi)} < {MIN_SUBJECTS_FOR_INFERENCE} multi-lesion subjects: "
              f"QUALITATIVE ONLY.")
        print("     Reported descriptively, no permutation test — same treatment as the")
        print("     n=28 histopathology subset.")
    else:
        print(f"\n  -> {len(multi)} multi-lesion subjects: enough for the descriptive")
        print("     comparison plus a permutation check, both reported with N stated.")

    all_correct = int((multi["n_correct"] == multi["n_lesions"]).sum())
    all_wrong = int((multi["n_correct"] == 0).sum())
    mixed = len(multi) - all_correct - all_wrong
    dist = pd.DataFrame([
        {"outcome": "all lesions correct", "n_subjects": all_correct},
        {"outcome": "mixed", "n_subjects": mixed},
        {"outcome": "all lesions incorrect", "n_subjects": all_wrong},
    ])
    dist["pct_of_multi_lesion_subjects"] = (dist["n_subjects"] / max(len(multi), 1) * 100).round(1)
    print("\nStep 2 — per-subject outcome distribution (multi-lesion subjects only):")
    print(dist.to_string(index=False))

    multi.sort_values(["accuracy", "n_lesions"]).to_csv(
        results_dir / f"robustness_intra_subject_{cfg_tag}.csv", index=False)

    # Do errors cluster by subject more than chance? Compare the observed
    # count of perfectly-consistent subjects against a null in which lesion
    # correctness is reshuffled across lesions, holding subject sizes fixed.
    overall_acc = df["correct_int"].mean()
    print(f"\nStep 3 — error clustering (overall lesion-level accuracy = {overall_acc:.4f}):")
    if qualitative_only:
        print("  Skipped — too few multi-lesion subjects for this to mean anything.")
        consistent_p = float("nan")
    else:
        sizes = multi["n_lesions"].to_numpy()
        labels = df["correct_int"].to_numpy()
        observed_consistent = all_correct + all_wrong

        rng = np.random.RandomState(42)
        null_counts = np.empty(N_PERMUTATIONS, dtype=int)
        for i in range(N_PERMUTATIONS):
            shuffled = rng.permutation(labels)
            idx = 0
            consistent = 0
            for s in sizes:
                chunk = shuffled[idx:idx + s]
                idx += s
                if chunk.sum() == s or chunk.sum() == 0:
                    consistent += 1
            null_counts[i] = consistent

        consistent_p = float((null_counts >= observed_consistent).mean())
        print(f"  perfectly-consistent subjects (all right or all wrong): "
              f"observed={observed_consistent}/{len(multi)}")
        print(f"  null (correctness reshuffled across lesions, {N_PERMUTATIONS} perms): "
              f"mean={null_counts.mean():.2f}, 95th pct={np.percentile(null_counts, 95):.1f}")
        print(f"  empirical p(null >= observed) = {consistent_p:.4f}")
        if consistent_p < 0.05:
            print("  -> Errors cluster by subject MORE than independent-per-lesion chance:")
            print("     some subjects are genuinely harder across their lesions.")
        else:
            print("  -> No detectable subject-level clustering; errors are consistent with")
            print("     being scattered independently across lesions.")
        print(f"  NOTE: {len(multi)} subjects is modest — report this p-value with the N")
        print("  stated alongside it, never on its own.")

    summary = pd.DataFrame([{
        "cfg_tag": cfg_tag,
        "n_subjects_total": len(per_subject),
        "n_subjects_multi_lesion": len(multi),
        "n_lesions_in_multi_lesion_subjects": int(multi["n_lesions"].sum()),
        "subjects_all_correct": all_correct,
        "subjects_mixed": mixed,
        "subjects_all_incorrect": all_wrong,
        "overall_lesion_accuracy": overall_acc,
        "mean_per_subject_accuracy": multi["accuracy"].mean(),
        "clustering_permutation_p": consistent_p,
        "qualitative_only": qualitative_only,
    }])
    summary.to_csv(results_dir / f"robustness_intra_subject_summary_{cfg_tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(multi["accuracy"], bins=np.linspace(0, 1, 11), color="tab:cyan", edgecolor="white")
    ax.axvline(overall_acc, color="black", linestyle="--", linewidth=1,
               label=f"overall lesion accuracy = {overall_acc:.3f}")
    ax.set_xlabel("Per-subject accuracy")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Per-subject accuracy, subjects with >=2 lesions\n({cfg_tag}, N={len(multi)} subjects)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / f"robustness_intra_subject_{cfg_tag}.png", dpi=150)
    plt.close(fig)

    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cfg-tag", default="channel_gated_qualityFalse",
                         help="which existing oof_predictions_<cfg_tag>.csv to analyze; defaults to "
                              "the designated main method, matching robustness analyses 1/3/4")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.results_dir / f"oof_predictions_{args.cfg_tag}.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} not found — run `python robustness_analysis.py` first to produce the "
            f"out-of-fold predictions these analyses read."
        )

    tables = load_raw_tables(args.data_dir)
    lesion_df = build_lesion_table(tables)

    oof = pd.read_csv(oof_path)
    oof = oof[oof["has_binary_label"]].copy()
    print(f"\nloaded {len(oof)} out-of-fold predictions from {oof_path}")

    analysis_5_certainty(oof, lesion_df, tables["dermatology_diagnosis"], args.results_dir, args.cfg_tag)
    analysis_6_intra_subject(oof, lesion_df, args.results_dir, args.cfg_tag)

    print("\n" + "=" * 78)
    print("ANALYSES 5 & 6 DONE — outputs in results/")
    print("=" * 78)


if __name__ == "__main__":
    main()
