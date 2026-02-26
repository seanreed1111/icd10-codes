# ICD-10 Code Lookup

A tool for looking up ICD-10-CM diagnosis codes by natural-language condition description, designed for nurses and clinical staff. Given a condition (e.g., "acute respiratory failure"), it returns the best matching code along with its category description and sibling codes so the user can confirm the correct level of specificity.

## Data

Raw reference data is sourced from the **CDC/NCHS ICD-10-CM April 1, 2026 release** (effective April 1 – September 30, 2026). The ICD-10-CM hierarchy has four levels:

```
Chapter (22)  →  Section (297)  →  Category (1,918 three-char codes)  →  Code (74,719 billable codes)
```

The processed dataset (`data/processed/icd10cm-codes-enriched-April-1-2026.csv`) joins all 74,719 billable codes with their category, section, and chapter metadata.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

### Build the enriched dataset

Run once from the project root to join codes with category/section/chapter metadata:

```bash
uv run python src/scripts/enrich_codes.py
```

### Load the knowledge base

```python
from src.codes import KnowledgeBase

kb = KnowledgeBase()
print(kb.entries[0])
```

### Extract codes from external sources (optional)

Scrapes codes from a website and two PDFs (Optum, LTC coding guides):

```bash
uv run python src/scripts/extract_icd10.py
```

PDF paths are hardcoded in the script and must exist locally.

## Project structure

```
data/
  raw/         # Source CSVs and XMLs from CDC NCHS; icd10_codes-simplified.csv from scraping
  processed/   # Enriched CSV produced by enrich_codes.py
src/
  codes.py             # KnowledgeBase and ICD10Code dataclasses
  scripts/
    enrich_codes.py    # Joins codes → categories → produces processed CSV
    extract_icd10.py   # Scrapes/extracts codes from web + PDFs
tests/
  test_data.json       # Sample test cases (condition → expected_code)
thoughts/              # Design notes
```

## Development

```bash
uv run ty check src/    # type check
uv run pytest           # run tests
```
