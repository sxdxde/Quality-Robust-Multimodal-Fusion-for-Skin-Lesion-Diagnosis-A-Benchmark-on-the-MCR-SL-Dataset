"""Assumed MCR-SL schema, transcribed from the dataset paper description in
CLAUDE.md. **NOT YET VERIFIED against the real CSVs** — this file exists so
validate_schema.py has something concrete to diff against. Once the real
files are inspected, fix any mismatches here first, then re-run
validate_schema.py until it passes cleanly. Do not write data/dataset.py
loaders against this file until it has been verified.

Each table maps expected column name -> expected pandas dtype kind, using
the same shorthand as `Series.dtype.kind`:
  'i' = signed int, 'u' = unsigned int, 'f' = float, 'O' = object (string),
  'b' = bool
Columns marked dtype=None are not dtype-checked (validate_schema.py only
checks presence for these), because the dataset paper's prose doesn't pin
down the exact numeric vs. categorical encoding used in the actual file
(e.g. Likert-style int codes stored as either int or string).
"""

LESION_TABLE = {
    "lesion_id": "O",
    "referral_diagnosis": "O",
    "lesion_status_when_captured": "O",
    "location": "O",
    "location_group": "O",
    "diameter": "f",
    "malignancy": None,  # boolean/int encoding unconfirmed
    "lesion_diagnosis": "O",
    "diagnosis_image_id": "O",
}

SUBJECT_TABLE = {
    "subject_id": "O",
    "derived_from": None,
    "age": "f",
    "sex": "O",
    "height": "f",
    "weight": "f",
    "natural_hair_color": "O",
    "skin_reaction_to_sun": "O",
    "sunbed": None,
    "h_cancer": None,
    "h_skin_cancer": None,
    "h_skin_cancer_relatives": None,
    "organ_transplant": None,
    "immunosuppresion": None,
    # mole-count fields: exact names unconfirmed, expect something like
    # "mole_count_*" or "total_mole_count" — verify and fill in.
}

IMAGE_TABLE = {
    "image_id": "O",
    "lesion_id": "O",
    "modality": "O",
}

DERMATOLOGY_DIAGNOSIS_TABLE = {
    "diagnosis_id": "O",
    "lesion_id": "O",
    "image_id": "O",
    "expert_id": "O",
    "diagnosis": "O",
    "2nd_option": None,
    "certainty": None,
    "image_rating": "f",
    "time": None,
}

HISTOPATHOLOGY_DIAGNOSIS_TABLE = {
    "lesion_id": "O",
    "procedure": "O",
    "tumor_thickness": "f",
    "diagnosis": "O",
}

UNIFIED_DIAGNOSIS_TABLE = {
    "lesion_id": "O",
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

# 9-class unified diagnosis label set (CLAUDE.md); small-N classes flagged
# per the brief's instruction to never present per-class metrics for these
# without flagging them.
UNIFIED_DIAGNOSIS_CLASSES = ["NEV", "SK", "BCC", "AK", "ATY", "MEL", "SCC", "ANG", "DF"]
SMALL_N_CLASSES = {"MEL": 8, "SCC": 5, "ANG": 4, "DF": 2}  # expected counts, verify

# Expert IDs whose image_rating is usable (E002 is known lost per CLAUDE.md).
USABLE_RATING_EXPERTS = ["E001", "E003", "E004"]
DROPPED_RATING_EXPERT = "E002"
