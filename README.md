# ICD-10 Code Lookup

A tool for looking up ICD-10-CM diagnosis codes by natural-language condition description, designed for nurses and clinical staff. Given a condition (e.g., "acute respiratory failure"), it returns the best matching code along with its category description and sibling codes so the user can confirm the correct level of specificity.

## CLI quick start

After running `uv sync` and building the enriched dataset (see [Setup](#setup) and [Usage](#usage) below), search from the terminal:

```bash
# Basic search — returns top 5 matches ranked by cosine similarity
uv run python src/cli.py "type 2 diabetes with hyperglycemia"

# Limit results
uv run python src/cli.py "acute respiratory failure" --top-k 3
uv run python src/cli.py "acute respiratory failure" -k 3

# Include sibling codes from the same 3-char category
uv run python src/cli.py "cholera unspecified" --siblings
uv run python src/cli.py "cholera unspecified" -s

# Combine flags
uv run python src/cli.py "chronic obstructive pulmonary disease" -k 3 -s
```

If the package entry point is installed (`uv run` installs it automatically):

```bash
icd10-search "type 2 diabetes with hyperglycemia" -k 3 -s
```

Example output with `--siblings`:

```
[1] E1165  score=0.7821
    Type 2 diabetes mellitus with hyperglycemia
    Category: E11  (9 sibling(s))
      > E1165  Type 2 diabetes mellitus with hyperglycemia
        E1100  Type 2 diabetes mellitus with hyperosmolarity without nonketotic ...
        E1110  Type 2 diabetes mellitus with ketoacidosis without coma
        ...
```

---

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

Install pre-commit hooks (runs ruff lint + format on every commit):

```bash
uv run pre-commit install
```

## Usage

### 1. Build the enriched dataset

Run once from the project root to join codes with category/section/chapter metadata:

```bash
uv run python src/scripts/enrich_codes.py
```

This reads `data/raw/icd10cm-codes-April-1-2026.csv` and `data/raw/icd10cm-categories-April-1-2026.csv` and writes `data/processed/icd10cm-codes-enriched-April-1-2026.csv`.

### 2. Load the knowledge base

`KnowledgeBase` reads the enriched CSV into a list of `ICD10Code` dataclass instances. Each entry carries: `code`, `description`, `description_aliases`, `category` (3-char code), and `chapter`.

```python
from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
print(kb.entries[0])
# ICD10Code(code='A000', description='Cholera due to Vibrio cholerae 01, biovar cholerae', ...)
```

The knowledge base can be serialised to Parquet for fast subsequent loads:

```python
from pathlib import Path
from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.save(Path("data/processed/kb.parquet"))

# Later — skip re-parsing the CSV
kb2 = KnowledgeBase.load_from_parquet(Path("data/processed/kb.parquet"))
```

### 3. Search with TF-IDF

#### How it works

`TfidfRetriever` (`src/retriever.py`) fits a **TF-IDF vectorizer** over all 74,719 code descriptions at construction time and answers queries with **cosine similarity**.

**Vectorizer configuration:**

| Setting | Value | Reason |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures both single words and two-word phrases (e.g. "type 1", "ketoacidosis without") |
| `stop_words` | `"english"` | Removes common words (the, with, of…) that add noise |
| `token_pattern` | `r"(?u)\b(?:\w\w+\|\d)\b"` | Includes single digits so "type 1" and "type 2" tokenize differently |

At query time the query string is vectorized with the same fitted vocabulary and scored against every document using `linear_kernel` (equivalent to cosine similarity on L2-normalised vectors). Results with a score of exactly zero are dropped — only codes that share at least one token with the query are returned.

**Scoring and ranking:**

- Scores range from `0.0` (no overlap) to `1.0` (identical text).
- Results are ordered by descending score; `rank` starts at 1.
- The returned list may be shorter than `top_k` if fewer than `top_k` codes have a non-zero score.

#### Return types

```python
@dataclass
class SearchResult:
    rank: int          # 1-based position in result list
    code: str          # ICD-10-CM code, e.g. "E1165"
    description: str   # Full description text
    score: float       # Cosine similarity, rounded to 4 decimal places

@dataclass
class SearchResultWithSiblings:
    rank: int
    code: str
    description: str
    score: float
    category_code: str              # First 3 chars of code, e.g. "E11"
    siblings: list[dict[str, str]]  # All codes sharing that category prefix
    # Each sibling: {"code": "E1165", "description": "..."}
```

#### Usage

```python
from src.knowledge_base import KnowledgeBase
from src.retriever import TfidfRetriever

kb = KnowledgeBase()
retriever = TfidfRetriever(kb)

# Basic search — returns ranked SearchResult objects
results = retriever.search("type 2 diabetes with hyperglycemia", top_k=5)
for r in results:
    print(r.rank, r.code, r.description, r.score)

# Search with siblings — also returns all codes in the same 3-char category
results = retriever.search_with_siblings("chronic obstructive pulmonary disease", top_k=3)
for r in results:
    print(r.rank, r.code, r.score)
    for sib in r.siblings:
        print("  sibling:", sib["code"], sib["description"])
```

`search_with_siblings` is the primary interface: it returns the best-matching codes **plus** every sibling code sharing the same 3-character category prefix, letting the nurse verify they have chosen the correct specificity level.

#### Edge cases

| Input | Behaviour |
|---|---|
| Empty string `""` | Returns `[]` immediately (vectorizer produces an all-zero vector) |
| `top_k=0` | Returns `[]` |
| `top_k` larger than corpus | Returns at most as many results as there are codes with non-zero score |
| Query with no matching tokens | Returns `[]` |

### 4. Extract codes from external sources (optional)

Scrapes additional codes from a website and two PDFs (Optum, LTC coding guides):

```bash
uv run python src/scripts/extract_icd10.py
```

PDF paths are hardcoded in the script and must exist locally. Output is written to `data/raw/icd10_codes-simplified.csv`.

## Running tests

```bash
uv run pytest
```

The test suite covers:

| File | What it tests |
|------|---------------|
| `tests/test_knowledge_base.py` | `KnowledgeBase` CSV load, `save()` / `load_from_parquet()` round-trip, entry integrity |
| `tests/test_retriever.py` | `TfidfRetriever.search()` ranking, scoring, edge cases; `search_with_siblings()` sibling correctness; integration tests against `tests/data/test_v1.json` |
| `tests/test_cli.py` | CLI output format, `--top-k` / `-k` flag, `--siblings` / `-s` flag, no-results message |

Useful flags:

```bash
uv run pytest -v                  # verbose — shows each test name
uv run pytest -k "test_roundtrip" # run only tests matching a pattern
uv run pytest --tb=short          # shorter tracebacks
uv run pytest -x                  # stop after first failure
```

## Type checking

```bash
uv run ty check src/
```

## Project structure

```
data/
  raw/
    icd10cm-codes-April-1-2026.csv        # 74,719 billable codes (source of truth)
    icd10cm-categories-April-1-2026.csv   # 1,918 category codes with section/chapter
    icd10_codes-simplified.csv            # Smaller set from extract_icd10.py
  processed/
    icd10cm-codes-enriched-April-1-2026.csv  # Joined output used by KnowledgeBase
src/
  cli.py                 # Typer CLI — `search` command with --top-k and --siblings flags
  knowledge_base.py      # KnowledgeBase, ICD10Code, Category, Chapter dataclasses
  retriever.py           # TfidfRetriever, SearchResult, SearchResultWithSiblings
  scripts/
    enrich_codes.py      # Joins codes → categories → writes processed CSV
    extract_icd10.py     # Scrapes/extracts codes from web + PDFs
  tfidf-demo/            # Step-by-step TF-IDF exploration scripts
tests/
  test_cli.py            # CLI output format, flags, no-results behaviour
  test_knowledge_base.py # KnowledgeBase unit + round-trip tests
  test_retriever.py      # Retriever unit + integration tests
  data/
    test_v1.json         # Integration test cases (condition → expected_code)
```
