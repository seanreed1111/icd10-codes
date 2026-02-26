# MDACE — MIMIC Documents Annotated with Code Evidence

**Paper:** Cheng et al., ACL 2023 — https://aclanthology.org/2023.acl-long.416
**Original repo:** https://github.com/3mcloud/MDACE (now private/removed)
**Annotations sourced from:** https://github.com/JoakimEdin/explainable-medical-coding
**License:** MIT (annotations). Note: models trained on this data inherit MIMIC's non-commercial restriction.

**Citation:**
```
@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua and Jafari, Rana and Russell, April and Klopfer, Russell
              and Lu, Edmond and Striner, Benjamin and Gormley, Matthew",
    booktitle = "Proceedings of ACL (Volume 1: Long Papers)",
    year = "2023",
    url = "https://aclanthology.org/2023.acl-long.416",
    pages = "7534--7550",
}
```

---

## Overview

MDACE is the first publicly available dataset for **code evidence extraction**: given a clinical
discharge note, identify the exact text span(s) that justify each assigned ICD code.

| Chart type | Documents | Evidence spans |
|------------|-----------|----------------|
| Inpatient  | 302       | 3,934          |
| Profee     | 52        | 5,563          |

Annotations were created by **professional medical coders** on a subset of MIMIC-III records.

**Important:** This directory contains only the annotation JSON files. The actual discharge
note text (the `TEXT` column from MIMIC-III's `NOTEEVENTS.csv`) is NOT included and requires
separate PhysioNet credentialing: https://physionet.org/content/mimiciii/1.4/

---

## Directory Structure

```
mdace/
├── Inpatient/
│   ├── ICD-10/1.0/    # 302 annotation JSONs  (e.g. 100197-ICD-10.json)
│   └── ICD-9/1.0/     # 302 annotation JSONs  (e.g. 100197-ICD-9.json)
├── Profee/
│   ├── ICD-10/1.0/    # 52  annotation JSONs
│   └── ICD-9/1.0/     # 52  annotation JSONs
└── splits/
    ├── inpatient/
    │   ├── MDace-ev-train.csv        # 181 hadm_ids  — evidence extraction task
    │   ├── MDace-ev-val.csv          # 60  hadm_ids
    │   ├── MDace-ev-test.csv         # 61  hadm_ids
    │   ├── MDace-code-ev-train.csv   # 47,900 rows  — code+evidence task (MIMIC-III full)
    │   ├── MDace-code-ev-val.csv     # 1,691 rows
    │   └── MDace-code-ev-test.csv    # 3,131 rows
    └── profee/
        ├── MDace-ev-train.csv        # 31  hadm_ids
        ├── MDace-ev-val.csv          # 10  hadm_ids
        ├── MDace-ev-test.csv         # 11  hadm_ids
        ├── MDace-code-ev-train.csv   # 47,750 rows
        ├── MDace-code-ev-val.csv     # 1,641 rows
        └── MDace-code-ev-test.csv    # 3,331 rows
```

**Two ICD versions are provided for each chart** (ICD-9 and ICD-10), because MIMIC-III
natively uses ICD-9 but the authors also mapped codes to ICD-10 for cross-version research.
Use `ICD-10/` for this project.

**Two split types:**
- `MDace-ev-*` — small splits containing only the 302/52 MDACE-annotated hadm_ids; used to
  evaluate evidence extraction on the annotated documents
- `MDace-code-ev-*` — large splits of all MIMIC-III hadm_ids; used to train/evaluate code
  prediction jointly with evidence extraction

---

## Annotation JSON Format

Each file is named `{hadm_id}-ICD-10.json` where `hadm_id` is the MIMIC-III hospital admission ID.

```json
{
  "hadm_id": 100197,
  "comment": "",
  "notes": [
    {
      "note_id": 25762,
      "category": "Discharge summary",
      "description": "Report",
      "annotations": [
        {
          "begin": 374,
          "end": 377,
          "code": "I61.8",
          "code_system": "ICD-10-CM",
          "description": "Other nontraumatic intracerebral hemorrhage",
          "type": "Human"
        }
      ]
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `hadm_id` | MIMIC-III hospital admission ID — join to `NOTEEVENTS.csv` |
| `note_id` | MIMIC-III `ROW_ID` in `NOTEEVENTS.csv` |
| `begin` / `end` | Character offsets into the note text for the evidence span |
| `code` | ICD code with dot notation (e.g. `I61.8`) |
| `code_system` | `ICD-10-CM` or `ICD-10-PCS` |
| `type` | Always `"Human"` — annotated by professional coders |

---

## Usage Example

```python
import json
from pathlib import Path

ANNOTATIONS = Path("data/mdace/Inpatient/ICD-10/1.0")

doc = json.loads((ANNOTATIONS / "100197-ICD-10.json").read_text())
for note in doc["notes"]:
    for ann in note["annotations"]:
        # ann["begin"]:ann["end"] indexes into the discharge note text
        # (text must be fetched from MIMIC-III NOTEEVENTS.csv)
        print(f"  [{ann['begin']}:{ann['end']}] {ann['code']} — {ann['description']}")
```

---

## What You Need from MIMIC-III

The annotation offsets are character positions into the raw note text. To reconstruct evidence
spans you need `NOTEEVENTS.csv` from MIMIC-III:

1. Apply for PhysioNet access: https://physionet.org/content/mimiciii/1.4/
2. After download, look up rows where `ROW_ID == note["note_id"]`
3. Slice `row["TEXT"][ann["begin"]:ann["end"]]` to get the evidence span

---

## What's in This Directory

```
Inpatient/ICD-10/1.0/   — 302 annotation JSONs
Inpatient/ICD-9/1.0/    — 302 annotation JSONs
Profee/ICD-10/1.0/      — 52  annotation JSONs
Profee/ICD-9/1.0/       — 52  annotation JSONs
splits/inpatient/        — 6 split CSVs (MDace-ev-* and MDace-code-ev-*)
splits/profee/           — 6 split CSVs
```

There may also be an empty `_tmp/` directory left over from the git sparse-checkout used to
download these files — it contains nothing and can be deleted.

---

## Key Notes

- **Each JSON maps a `hadm_id` to a list of `{begin, end, code, code_system}` annotations** —
  character offsets into the discharge note text.
- **The note text is not here.** It requires MIMIC-III from PhysioNet. The annotations alone
  are usable for understanding the structure and writing loading/evaluation code.
- **Use `ICD-10/`** and the `MDace-ev-*` splits for evidence extraction evaluation on this project.
- **`_tmp/`** is an empty artifact from the git clone — safe to delete manually.
