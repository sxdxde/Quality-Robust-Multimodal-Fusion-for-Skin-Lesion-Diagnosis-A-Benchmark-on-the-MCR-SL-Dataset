"""VERIFIED MCR-SL schema, confirmed against the real extracted files on
2026-08-19 (`~/mcrsl_project/data/raw/extracted/MCR-SL_dataset/*.xlsx`).
Supersedes the earlier prose-derived guess from CLAUDE.md. Key corrections
vs. the CLAUDE.md description:

- Files are .xlsx, not .csv, named exactly after each table
  (lesion.xlsx, subject.xlsx, image.xlsx, dermatology_diagnosis.xlsx,
  histopathology_diagnosis.xlsx, unified_diagnosis.xlsx).
- `malignancy` is itself a 3-way field: "Malignant" (42) / "Non-malignant"
  (192) / "unknown" (6) — used directly as binary ground truth; the 6
  "unknown" lesions are EXCLUDED from the binary task (no valid label).
- `image.modality` values are "dermoscopy" / "clinical", not "dermoscopic".
- `diameter`, `height`, `weight` are pandas object dtype: numeric strings
  mixed with the literal string "unknown" — parsed as numeric-with-missing,
  not a plain float column.
- Histopathology-confirmed lesions: 28 (not 29 as CLAUDE.md's prose said).
- `image.xlsx` has 21 lesion_ids (263 images) not present in `lesion.xlsx`
  ("orphans" — no metadata/label available) — dropped via inner join in
  data/dataset.py; logged there, not silently ignored.
- Quality-rating coverage: 238/240 lesions have all 3 usable-expert
  (E001/E003/E004) ratings on their diagnosis_image_id row; 2 lesions
  (L0013, L0205) have none. No partial-coverage cases.
"""

LESION_TABLE = {
    "subject_id": "O",
    "lesion_id": "O",
    "referral_diagnosis": "O",
    "lesion_status_when_captured": "O",
    "location": "O",
    "location_group": "O",
    "diameter": "O",  # numeric-with-"unknown" string, parse explicitly
    "malignancy": "O",  # "Malignant" / "Non-malignant" / "unknown"
    "lesion_diagnosis": "O",  # NEVER use as a model input — leaks the label
    "diagnosis_image_id": "O",
}

SUBJECT_TABLE = {
    "subject_id": "O",
    "derived_from": "O",
    "year_of_birth": "O",
    "age": "i",
    "sex": "O",
    "height": "O",  # numeric-with-"unknown"
    "weight": "O",  # numeric-with-"unknown"
    "natural_hair_color": "O",
    "skin_reaction_to_sun": "O",
    "moles_body_18": "O",
    "moles_bigger_5mm": "O",
    "moles_bigger_20cm": "O",  # constant ("No" for all 60 subjects) — no signal, dropped in dataset.py
    "moles_body": "O",
    "sunburn_number": "O",  # messy (ints + ">10" + "unknown") — dropped in favor of sunburn_number_group
    "sunburn_age": "O",  # multi-value free text ("8, 17, 22...") — dropped, too messy at N=240
    "sunburn_number_group": "O",
    "sunbed": "O",
    "h_cancer": "O",
    "h_skin_cancer": "O",
    "h_skin_cancer_relatives": "O",
    "organ_transplant": "O",
    "immunosuppresion": "O",
}

IMAGE_TABLE = {
    "image_id": "O",
    "lesion_id": "O",
    "modality": "O",  # "dermoscopy" / "clinical"
}

DERMATOLOGY_DIAGNOSIS_TABLE = {
    "diagnosis_id": "O",
    "lesion_id": "O",
    "image_id": "O",
    "expert_id": "O",
    "diagnosis": "O",
    "2nd_option": "O",
    "certainty": "i",
    "image_rating": "f",
    "time": "f",
}

HISTOPATHOLOGY_DIAGNOSIS_TABLE = {
    "diagnosis_id": "O",
    "lesion_id": "O",
    "procedure": "O",
    "tumor_thickness": "f",
    "diagnosis": "O",
}

UNIFIED_DIAGNOSIS_TABLE = {
    "diagnosis_id": "O",
    "lesion_id": "O",
    "histopathology_diagnosis": "O",
    "diagnosis_id_histopathology": "O",
    "dermatology_diagnosis": "O",
    "unified_diagnosis": "O",
}

ASSUMED_TABLES = {
    "lesion": LESION_TABLE,
    "subject": SUBJECT_TABLE,
    "image": IMAGE_TABLE,
    "dermatology_diagnosis": DERMATOLOGY_DIAGNOSIS_TABLE,
    "histopathology_diagnosis": HISTOPATHOLOGY_DIAGNOSIS_TABLE,
    "unified_diagnosis": UNIFIED_DIAGNOSIS_TABLE,
}

# 9-class unified diagnosis label set. "UNK" rows (5/240) are excluded from
# both the binary and 9-class tasks (not a diagnosis, an unresolved case).
UNIFIED_DIAGNOSIS_CLASSES = ["NEV", "SK", "BCC", "AK", "ATY", "MEL", "SCC", "ANG", "DF"]
SMALL_N_CLASSES = {"MEL": 8, "SCC": 4, "ANG": 4, "DF": 2}  # verified counts

MALIGNANT = "Malignant"
NON_MALIGNANT = "Non-malignant"
UNKNOWN_MALIGNANCY = "unknown"

# Expert IDs whose image_rating is usable (E002's ratings are all null — lost
# per CLAUDE.md, confirmed: 0/241 non-null vs 241/241 for E001/E003/E004).
USABLE_RATING_EXPERTS = ["E001", "E003", "E004"]
DROPPED_RATING_EXPERT = "E002"

# --- Metadata field configuration for models/metadata_encoder.py ---
# Categorical vocabularies are fixed from the full dataset's known value sets
# (a structural fact about the field's domain, not a per-fold statistic —
# unlike numeric z-score stats, this isn't test-fold leakage). The literal
# string "unknown" present in the raw data maps to the encoder's reserved
# "unknown" embedding index, never to one of the listed real categories.
CATEGORICAL_FIELDS = {
    # subject-level
    "sex": ["Male", "Female"],
    "natural_hair_color": ["Fair blonde", "Dark brown, black", "Brown", "Red or auburn", "Blonde"],
    "skin_reaction_to_sun": ["Brown without first becoming red", "Red with pain", "Red"],
    "derived_from": ["Plastic surgery", "Dermatology", "Volunteer"],
    "moles_body_18": ["Few", "Some", "Many"],
    "moles_body": ["Few", "Some", "Many"],
    "moles_bigger_5mm": ["Yes", "No"],
    "sunburn_number_group": ["0", "1-2", "3-5", ">5"],
    "sunbed": ["Yes", "No"],
    "h_cancer": ["Yes", "No"],
    "h_skin_cancer": ["Yes", "No"],
    "h_skin_cancer_relatives": ["Yes", "No"],
    "organ_transplant": ["Yes", "No"],
    "immunosuppresion": ["Yes", "No"],
    # lesion-level
    "referral_diagnosis": ["Melanoma", "Voluntary sample", "Nevus", "BCC", "SK", "Morbus bowen carcinoma"],
    "lesion_status_when_captured": ["Lesion", "Biopsied lesion"],
    "location_group": ["Back", "Face", "Torso", "Legs", "Arms", "Head"],
}

# Numeric fields; height/weight/diameter need "unknown"-string -> missing
# parsing (see data/dataset.py parse_numeric_with_unknown), age is already
# clean int64 with no missing marker observed.
NUMERICAL_FIELDS = ["age", "height", "weight", "diameter"]

# Explicitly dropped fields and why (do not add these back without updating
# this comment + CLAUDE.md's "future work" note):
#   moles_bigger_20cm  - constant across all 60 subjects, zero signal
#   sunburn_age        - multi-value free text, not a scalar/categorical
#   sunburn_number      - superseded by the cleaner sunburn_number_group
#   location            - 24-way, redundant with location_group (which the
#                          dataset paper's own stats table used)
#   year_of_birth        - redundant with numeric `age`
#   lesion_diagnosis, malignancy, diagnosis_image_id, unified_diagnosis,
#   lesion_id, subject_id - identifiers or label-leaking fields, never inputs
