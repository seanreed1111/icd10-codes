# ICD-10-CM Reference Data — April 1, 2026

All files in this directory are derived from the official CDC/NCHS ICD-10-CM April 1, 2026 release,
effective April 1, 2026 – September 30, 2026.

**Primary source:** [CDC NCHS FTP — 2026-update](https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026-update/)

---

## File Overview

The ICD-10-CM classification is hierarchical. The four CSV files correspond to successive levels
of that hierarchy, from broadest to most specific:

```
Chapter  (22 total)
  └── Section  (297 total)
        └── Category  (1,918 total)  ← 3-character codes
              └── Code  (74,719 total)  ← billable codes, 4–7 characters
```

---

## Files

### `icd10cm-chapters-April-1-2026.csv`
**22 rows | columns: `chapter`, `description`**

The 22 clinical chapters that partition the entire ICD-10-CM code space. Each chapter covers a
broad domain of medicine.

| chapter | description |
|---------|-------------|
| 1 | Certain infectious and parasitic diseases (A00-B99) |
| 2 | Neoplasms (C00-D49) |
| 3 | Diseases of the blood and blood-forming organs... (D50-D89) |

**Potential uses:**
- High-level grouping for dashboards and executive reporting
- Filtering or faceting claims data by clinical domain
- Training data labels for broad disease-area classification models

---

### `icd10cm-sections-April-1-2026.csv`
**297 rows | columns: `section_range`, `chapter`, `description`**

Sections are named groupings of related 3-character categories within a chapter. The
`section_range` column holds the code range (e.g., `A00-A09`), which can be used to determine
which section any given code falls under.

| section_range | chapter | description |
|---------------|---------|-------------|
| A00-A09 | 1 | Intestinal infectious diseases (A00-A09) |
| A15-A19 | 1 | Tuberculosis (A15-A19) |
| C00-C14 | 2 | Malignant neoplasms of lip, oral cavity and pharynx (C00-C14) |

**Potential uses:**
- Mid-level grouping that is more granular than chapters but less noisy than individual codes
- Mapping codes to clinical service lines or specialty areas
- Epidemiological analysis by disease family

---

### `icd10cm-categories-April-1-2026.csv`
**1,918 rows | columns: `ICD10-CM-CODE`, `description`, `section`, `chapter`**

The 3-character category codes are the most clinically meaningful summary level — they identify
a specific disease or condition without requiring the full specificity of a billable code. The
`section` and `chapter` columns allow direct join to the other reference files.

| ICD10-CM-CODE | description | section | chapter |
|---------------|-------------|---------|---------|
| A00 | Cholera | A00-A09 | 1 |
| A01 | Typhoid and paratyphoid fevers | A00-A09 | 1 |
| I10 | Essential (primary) hypertension | I10-I1A | 9 |

**Potential uses:**
- Enriching billing or claims records with a human-readable disease label
- Grouping patients by condition for cohort analysis
- Feature engineering for ML models (category as a rolled-up feature vs. raw code)
- Lookup table to validate or describe codes received from EHR systems
- Joining against the codes file to roll detailed codes up to category level

---

### `icd10cm-codes-April-1-2026.csv`
**74,719 rows | columns: `ICD10-CM-CODE`, `description`**

The complete set of billable ICD-10-CM diagnosis codes (4–7 characters). These are the codes
that appear on actual claims and clinical documentation. Derived from the official CMS flat-file
(`icd10cm-codes-April-1-2026.txt`).

| ICD10-CM-CODE | description |
|---------------|-------------|
| A000 | Cholera due to Vibrio cholerae 01, biovar cholerae |
| A001 | Cholera due to Vibrio cholerae 01, biovar eltor |
| I10 | Essential (primary) hypertension |

**Potential uses:**
- Validating diagnosis codes submitted on claims
- Description lookup for any billable code encountered in raw data
- Joining to the categories file to add rolled-up labels
- Building autocomplete or search tools for code selection

---

## Joining the Files

The files are designed to join together via the code prefix:

```python
# Example: add category description to a codes dataframe
import pandas as pd

codes = pd.read_csv('icd10cm-codes-April-1-2026.csv')
categories = pd.read_csv('icd10cm-categories-April-1-2026.csv')

# Extract 3-character prefix from each billable code
codes['category'] = codes['ICD10-CM-CODE'].str[:3]

enriched = codes.merge(
    categories.rename(columns={'ICD10-CM-CODE': 'category', 'description': 'category_description'}),
    on='category',
    how='left'
)
```

---

## Source Files

| File | Source |
|------|--------|
| `icd10cm-codes-April-1-2026.txt` | Original flat-file from CMS April 1, 2026 release |
| `icd10c-tabular-April-1-2026.xml` | CDC NCHS tabular XML (inside `icd10cm-April-1-2026-XML.zip`) |
