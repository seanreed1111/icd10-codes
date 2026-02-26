# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Build a tool that lets nurses look up ICD-10-CM diagnosis codes by natural-language condition description. The goal is to return the best matching code along with its category description and sibling codes within that category, so the nurse can confirm the correct code.

## Commands

This project uses `uv` for dependency management and `ty` for type checking.

```bash
# Install dependencies
uv sync

# Run a script
uv run python src/codes.py
uv run python src/scripts/enrich_codes.py
uv run python src/scripts/extract_icd10.py

# Type check
uv run ty check src/

# Run tests (pytest is a dependency; no test files yet beyond test_data.json)
uv run pytest
```

## Architecture

### Data pipeline (two-stage)

**Stage 1 — Extraction** (`src/scripts/extract_icd10.py`): Scrapes ICD-10 codes from a website and two PDFs (Optum, LTC), deduplicates by normalized code, and writes `data/raw/icd10_codes-simplified.csv`.

**Stage 2 — Enrichment** (`src/scripts/enrich_codes.py`): Joins the full 74,719-row codes CSV to the categories CSV (left join on first 3 chars of code), then writes the enriched result to `data/processed/icd10cm-codes-enriched-April-1-2026.csv`. Run this from the project root.

### Knowledge base (`src/codes.py`)

`KnowledgeBase` reads the enriched CSV and populates a list of `ICD10Code` dataclass instances. Each `ICD10Code` holds: `code`, `description`, `description_aliases`, a `Category` object, and a `Chapter` (stored as a raw string from the CSV for now). This is the main in-memory representation for search.

### ICD-10-CM hierarchy (from `data/raw/`)

```
Chapter (22)  →  Section (297)  →  Category (1,918, 3-char codes)  →  Code (74,719, 4–7 chars)
```

The enriched CSV (`data/processed/`) combines the codes level with category, section, and chapter metadata. Joining is done by taking `code[:3]` as the category key.

### Search design intent (`thoughts/search-thoughts.md`)

When returning a code lookup result, also return the category description and all sibling codes in that category so the nurse can confirm the right specificity.

## Key file paths

| Path | Contents |
|------|----------|
| `data/raw/icd10cm-codes-April-1-2026.csv` | 74,719 billable codes (source of truth) |
| `data/raw/icd10cm-categories-April-1-2026.csv` | 1,918 category codes with section/chapter |
| `data/processed/icd10cm-codes-enriched-April-1-2026.csv` | Joined output used by `KnowledgeBase` |
| `data/raw/icd10_codes-simplified.csv` | Smaller set scraped/extracted by `extract_icd10.py` |
| `tests/test_data.json` | Sample search test cases (`expected_code`, `condition`) |

## Notes

- `ty` (not mypy) is the type checker; config is in `pyproject.toml` under `[tool.ty]`. Several third-party packages (pandas, requests, bs4, etc.) are set to `replace-imports-with-any` to suppress unresolved-import errors.
- `enrich_codes.py` must be run from the project root because it uses relative paths (`data/raw/`, `data/processed/`).
- Raw data files are sourced from the CDC NCHS ICD-10-CM April 1, 2026 release (effective April 1 – September 30, 2026).
