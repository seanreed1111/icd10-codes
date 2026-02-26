# COT-SYMP-ICD10-2024

**Source:** https://huggingface.co/datasets/Inje/COT-SYMP-ICD10-2024
**License:** Not explicitly declared on dataset card — verify before production use.

---

## Overview

12,132 synthetic question-answer pairs mapping a condition/symptom description to an ICD-10
code, with a chain-of-thought reasoning trace explaining the assignment.

| Split | Rows | Columns |
|-------|------|---------|
| train | 12,132 | `question`, `answer` |

(Single split only — no dev/test.)

---

## Format

**`question`** — a natural-language condition description in the form:
```
"What is the ICD-10 code for the <condition description>?"
```

**`answer`** — a chain-of-thought reasoning trace ending with the ICD-10 code after `####`:
```
The provided description indicates a follow-up care scenario after an orthopedic treatment
or surgery. The key terms are 'orthopedic' and 'follow-up care.' ICD-10 codes in the 'Z'
range are generally used for factors influencing health status... Z47 refers to 'Follow-up
care involving orthopedic surgery'. The 'unspecified' component...
Fever, Fatigue, Body aches... #### Z47.9
```

The final ICD-10 code is always the last token after `####`.

---

## Sample Rows

| # | Question (truncated) | Code |
|---|----------------------|------|
| 0 | Orthopaedic follow-up care, unspecified | Z47.9 |
| 500 | Atresia of oesophagus without fistula | Q39.0 |
| 5000 | Viral infection, unspecified | B34.9 |
| 12131 | Acute pharyngitis due to other specified organisms | J02.8 |

---

## Loading

```python
from datasets import load_from_disk

# Load from local disk (no network needed after first download)
ds = load_from_disk("data/COT-SYMP-ICD10-2024")
train = ds["train"]   # 12,132 rows

# Or reload from HuggingFace (set HF_TOKEN in .env for higher rate limits)
# from datasets import load_dataset
# ds = load_dataset("Inje/COT-SYMP-ICD10-2024")

# Extract the final code from an answer
def extract_code(answer: str) -> str:
    return answer.split("####")[-1].strip()

row = train[0]
print(row["question"])
print(extract_code(row["answer"]))  # → "Z47.9" NOTE: need to remove the decimal point for evals
```

---

## Files on Disk

```
COT-SYMP-ICD10-2024/        (8.5 MB total)
├── dataset_dict.json        # top-level split manifest
└── train/
    ├── data-00000-of-00001.arrow   # Arrow-format data
    ├── dataset_info.json
    └── state.json
```

---

## Caveats

- **Synthetic data** — questions and answers were generated, not drawn from real clinical notes
  coded by professionals. Use for rapid prototyping and iteration, not final benchmarking.
- **Single split** — no predefined train/dev/test separation. Shuffle and split manually if
  needed for evaluation.
- **ICD version** — codes appear to be ICD-10 (not ICD-10-CM specifically); some codes may
  not match the ICD-10-CM April 2026 reference exactly. Normalize before comparing.
- **HF_TOKEN** — available in `.env`. Pass via `HF_HUB_TOKEN` env var or
  `huggingface_hub.login()` for higher rate limits on future HuggingFace downloads.
