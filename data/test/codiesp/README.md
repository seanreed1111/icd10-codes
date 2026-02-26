# CodiEsp — ICD-10 Benchmark Dataset

**Source:** CLEF eHealth 2020 Shared Task
**License:** CC BY 4.0
**Zenodo corpus:** https://zenodo.org/records/3693570
**Zenodo gold standard:** https://zenodo.org/records/3837305
**Project page:** https://temu.bsc.es/codiesp/

---

## Overview

1,000 Spanish clinical case documents manually coded by professional coders using CIE-10
(the Spanish adaptation of ICD-10-CM/PCS). Machine-translated English versions of each
document are included in `text_files_en/`.

| Split | Documents |
|-------|-----------|
| train | 500 |
| dev | 250 |
| test | 250 |
| background (gold only) | 2,751 unannotated |

---

## Directory Structure

```
codiesp/
├── corpus/                            # Zenodo 3693570 (v2 — train/dev labels; no test labels)
│   └── final_dataset_v2_to_publish/
│       ├── train/
│       │   ├── trainD.tsv             # Diagnosis code annotations
│       │   ├── trainP.tsv             # Procedure code annotations
│       │   ├── trainX.tsv             # Explainability annotations (evidence spans)
│       │   ├── text_files/            # Spanish clinical case text (.txt)
│       │   └── text_files_en/         # English machine-translated text (.txt)
│       ├── dev/  (same structure)
│       └── test/
│           ├── text_files/            # Text only — no labels (use gold/ for labels)
│           └── text_files_en/
│
└── gold/                              # Zenodo 3837305 (v4 — full gold standard inc. test)
    └── final_dataset_v4_to_publish/
        ├── train/   (same structure as corpus/train)
        ├── dev/     (same structure as corpus/dev)
        ├── test/
        │   ├── testD.tsv              # Gold diagnosis labels for test split
        │   ├── testP.tsv              # Gold procedure labels for test split
        │   ├── testX.tsv             # Gold explainability labels for test split
        │   ├── text_files/
        │   └── text_files_en/
        └── background/
            ├── text_files/            # 2,751 unannotated background docs
            └── text_files_en/
```

**Use `gold/` for all evaluation.** It is the authoritative v4 release and includes test labels.

---

## Annotation Format

The `.tsv` files are tab-separated with no header: `doc_id <TAB> icd10_code`

```
S0004-06142005000700014-1    n44.8
S0004-06142005000700014-1    z20.818
S0004-06142005000700014-1    r60.9
```

- `doc_id` matches the filename stem in `text_files/` (e.g., `S0004-06142005000700014-1.txt`)
- ICD-10 codes are lowercase (normalize to uppercase to match ICD-10-CM standard)
- One row per code; a document can have multiple codes

Sub-tasks:
- **D** (Diagnosis) — ICD-10-CM codes
- **P** (Procedure) — ICD-10-PCS codes
- **X** (Explainability) — evidence text spans that justify each code assignment

---

## Usage Example

```python
import pandas as pd
from pathlib import Path

BASE = Path("data/codiesp/gold/final_dataset_v4_to_publish")

# Load diagnosis annotations
train_d = pd.read_csv(BASE / "train/trainD.tsv", sep="\t", header=None,
                      names=["doc_id", "code"])
train_d["code"] = train_d["code"].str.upper()  # normalize to uppercase

# Load corresponding English text
def load_text(split: str, doc_id: str) -> str:
    path = BASE / split / "text_files_en" / f"{doc_id}.txt"
    return path.read_text(encoding="utf-8")

# Example: first document's text + codes
doc_id = train_d["doc_id"].iloc[0]
text = load_text("train", doc_id)
codes = train_d[train_d["doc_id"] == doc_id]["code"].tolist()
print(doc_id, codes)
```

---

## Notes

- Documents are formal clinical case reports — more structured than nursing notes or ED summaries.
- Multi-label: each document has between 1 and ~20 ICD-10 codes.
- For the anthuria project (lookup by condition description), the most useful evaluation is
  checking whether the top-1 or top-k returned code appears in the gold label set for a
  document whose text contains that condition description.
