"""Subject-disjoint stratified 5-fold splitting.

Never split at the lesion level — a subject can have multiple lesions, and
images of the same lesion are near-duplicates. Folds are assigned per
subject_id, greedily balancing total malignant-lesion count per fold (a
subject can carry multiple lesions of mixed malignancy).
"""
import random
from collections import defaultdict


def make_subject_disjoint_folds(
    subject_to_malignant_count: dict[str, int],
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """
    subject_to_malignant_count: {subject_id: number of malignant lesions for that subject}
        (subjects with 0 malignant lesions are still included, count=0)

    Returns {subject_id: fold_index in [0, n_folds)}.

    Greedy balancing: sort subjects by malignant-lesion count descending,
    assign each to the fold with the currently-lowest total malignant count
    (ties broken by lowest total subject count, then randomly for reproducible
    tie-breaking under `seed`).
    """
    rng = random.Random(seed)
    subjects = list(subject_to_malignant_count.keys())
    rng.shuffle(subjects)  # randomize tie order before the stable sort below
    subjects.sort(key=lambda s: subject_to_malignant_count[s], reverse=True)

    fold_malignant_totals = [0] * n_folds
    fold_subject_totals = [0] * n_folds
    assignment: dict[str, int] = {}

    for subject_id in subjects:
        count = subject_to_malignant_count[subject_id]
        best_fold = min(
            range(n_folds),
            key=lambda f: (fold_malignant_totals[f], fold_subject_totals[f]),
        )
        assignment[subject_id] = best_fold
        fold_malignant_totals[best_fold] += count
        fold_subject_totals[best_fold] += 1

    return assignment


def fold_summary(subject_to_malignant_count: dict[str, int], assignment: dict[str, int], n_folds: int = 5) -> str:
    lines = []
    per_fold = defaultdict(lambda: {"subjects": 0, "malignant": 0})
    for subject_id, fold in assignment.items():
        per_fold[fold]["subjects"] += 1
        per_fold[fold]["malignant"] += subject_to_malignant_count[subject_id]
    for f in range(n_folds):
        stats = per_fold[f]
        lines.append(f"fold {f}: {stats['subjects']} subjects, {stats['malignant']} malignant lesions")
    return "\n".join(lines)
