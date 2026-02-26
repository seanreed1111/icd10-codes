# Data Directory

This directory contains ICD-10-CM reference data and benchmark datasets for testing code prediction.

---

## Directory Structure

```
data/
├── raw/               # CDC/NCHS ICD-10-CM April 1, 2026 reference files
├── processed/         # Enriched/joined outputs used by KnowledgeBase
├── test/              # Test fixtures
└── codiesp/           # CodiEsp benchmark dataset (see below)
```

See `raw/README.md` for full documentation of the ICD-10-CM reference files.

---

## Benchmark Datasets for ICD-10 Code Prediction

The table below summarizes publicly available datasets suitable for evaluating ICD-10 code
prediction from natural language text, ranked by ease of access.

### Immediately Accessible (no registration)

| Dataset | Size | ICD Version | Access |
|---------|------|-------------|--------|
| **CodiEsp** | 1,000 clinical cases | ICD-10-CM (Spanish CIE-10) | Free download, CC BY 4.0 |
| **COT-SYMP-ICD10-2024** | 12,132 QA pairs (synthetic) | ICD-10 | HuggingFace, no login |

### Requires Free PhysioNet Credentialing (~1–3 days, CITI course)

| Dataset | Size | ICD Version | Notes |
|---------|------|-------------|-------|
| **MIMIC-IV + MIMIC-IV-ICD Benchmark** | ~330,000 admissions | ICD-10-CM + ICD-9 | Gold standard; real discharge summaries with ready-made train/test splits |
| **MIMIC-IV Synthetic Low-Resource** | 9,606 notes | ICD-10-CM | GPT-3.5 generated summaries for rare codes |
| **MIMIC-III + AnEMIC Benchmark** | ~52,700 discharge summaries | ICD-9 (ICD-10 variant available) | Most-cited in literature |
| **MDACE** | 354 docs + evidence spans | ICD-9 | Human-annotated text evidence for each code; useful for explainability |

### Free Registration / DUA Only

| Dataset | Size | ICD Version | Notes |
|---------|------|-------------|-------|
| **n2c2 / i2b2** | ~1,000–5,000 notes per task | None natively | Clinical NLP tasks; discharge summaries available |
| **MTSamples (Kaggle)** | 4,998 transcriptions | None natively | No ICD labels; requires annotation |

---

## CodiEsp (`codiesp/`)

**Source:** CLEF eHealth 2020 Shared Task
**License:** CC BY 4.0
**Zenodo:** https://zenodo.org/records/3693570 (corpus) | https://zenodo.org/records/3837305 (gold standard)
**Project page:** https://temu.bsc.es/codiesp/

1,000 Spanish clinical case documents manually coded by professional coders using CIE-10
(the Spanish adaptation of ICD-10-CM/PCS). Machine-translated English versions are included.

| Split | Documents |
|-------|-----------|
| Train | 500 |
| Development | 250 |
| Test | 250 |
| Background (unannotated) | 2,751 |

Three sub-tasks:
- **CodiEsp-D** — Diagnosis codes (ICD-10-CM)
- **CodiEsp-P** — Procedure codes (ICD-10-PCS)
- **CodiEsp-X** — Explainability: submit evidence text spans alongside predicted codes

**Caveats:** These are clinical case reports (didactic), not real-world discharge summaries.
Vocabulary is formal. Useful for benchmarking code lookup precision on short condition descriptions.

---

## MIMIC-IV (apply separately)

**Source:** PhysioNet https://physionet.org/content/mimiciv/3.1/
**Benchmark paper:** https://arxiv.org/abs/2304.13998
**Processing code:** https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing

The field's gold standard. ~330,000 real hospital admissions with discharge summaries coded
by professional coders. Four benchmark splits (ICD-10 full, ICD-10 top-50, ICD-9 full, ICD-9 top-50).

Access requires: PhysioNet account + CITI "Data or Specimens Only Research" course + DUA.

---

## COT-SYMP-ICD10-2024 (HuggingFace)

**Source:** https://huggingface.co/datasets/Inje/COT-SYMP-ICD10-2024

```python
from datasets import load_dataset
ds = load_dataset("Inje/COT-SYMP-ICD10-2024")
```

12,132 synthetic symptom-description → ICD-10 code pairs with chain-of-thought reasoning traces.
Good for rapid prototyping. Not real clinical notes — use for iteration before graduating to MIMIC.

---

## Further Reading

- Curated paper/dataset list: https://github.com/acadTags/Awesome-medical-coding-NLP
- Systematic review (73 papers, 2014–2024): https://journals.sagepub.com/doi/10.1177/20552076251404518
- Revisiting MIMIC-IV benchmark (ACL 2024): https://aclanthology.org/2024.cl4health-1.23.pdf
