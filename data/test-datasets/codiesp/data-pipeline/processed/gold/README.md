# CodiEsp Ground Truth — Processed Gold Data

## Overview

`codiesp_ground_truth.parquet` is the ground-truth diagnosis code dataset derived from
the CodiEsp dev split. It maps each clinical text file to the semicolon-separated list
of ICD-10-CM diagnosis codes assigned by human annotators.

## Source Data

| Item | Path |
|------|------|
| Annotation TSV | `data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/devX.tsv` |
| Clinical text files | `data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en/` |

The CodiEsp dataset was released as part of the IberLEF 2020 shared task on clinical
NLP for Spanish. The `devX.tsv` file contains annotation rows with columns:
`file_stem`, `type`, `code`, `description`, `span`.

## Processing Steps

Performed by `src/scripts/parse_codiesp.py` (run from the project root):

1. **Filter to text files only** — only rows whose `file_stem` matches a `.txt` file
   in `text_files_en/` are retained, discarding any annotations without a corresponding
   text.
2. **Filter to DIAGNOSTICO type** — only rows where `type == "DIAGNOSTICO"` are kept;
   procedure codes and other annotation types are excluded.
3. **Normalize codes** — each code is uppercased and all non-word characters are
   stripped (e.g. dots removed: `"M54.5"` → `"M545"`).
4. **Aggregate per file** — codes for the same `file_stem` are joined with `";"` in
   original row order, producing one row per file.
5. **Save to Parquet** — the result is written via `save_to_parquet()` to this
   directory as `codiesp_ground_truth.parquet`.

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `file_stem` | `String` | Filename without extension (e.g. `"S0010"`) |
| `codes` | `String` | Semicolon-separated, normalized ICD-10-CM codes (e.g. `"M545;M791"`) |

## Regenerating

Run from the project root:

```bash
uv run python src/scripts/parse_codiesp.py
```

This overwrites `codiesp_ground_truth.parquet` in this directory.

## Loading in Python

```python
from scripts.parse_codiesp import load_from_parquet

results = load_from_parquet()
# returns list[tuple[str, str]] — [(file_stem, codes), ...]
```
