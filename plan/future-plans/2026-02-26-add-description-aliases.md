# Plan: Add description_aliases column to enriched ICD-10 CSV

## Context

The `KnowledgeBase` dataclass already has a `description_aliases: list[str]` field and handles it on load (`knowledge_base.py:44-47`), but the enriched CSV has no such column yet. Nurses need to find codes via lay terms (e.g., "high blood pressure" → `I10 Essential hypertension`). This script generates those aliases using the Claude API and writes a new v2 CSV.

---

## Files to create / modify

| File | Action |
|------|--------|
| `src/scripts/add_description_aliases.py` | **Create** — the main enrichment script |
| `pyproject.toml` | **Modify** — add `anthropic` dependency |

Input: `data/processed/icd10cm-codes-enriched-April-1-2026.csv` (74,719 rows)
Output: `data/processed/icd10cm-codes-enriched-v2-April-1-2026.csv`
Checkpoint: `data/processed/aliases_checkpoint.jsonl` (append-only, enables resume)

---

## pyproject.toml changes

1. Add `"anthropic>=0.40.0"` to `[project].dependencies` (alphabetical order, after existing entries)
2. Add `"anthropic", "anthropic.**"` to `[tool.ty.analysis].replace-imports-with-any`

---

## Script structure (`src/scripts/add_description_aliases.py`)

Follow the same pattern as `src/scripts/enrich_codes.py`: path constants at top, `main()` entry point, run from project root.

### Key constants
```python
MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 75          # ~997 batches total
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10  # doubles each retry
```

### System prompt
Instruct the model to return **only a JSON array** with one element per input (same order), each a semicolon-separated string of 1–4 lay-term synonyms. Empty string `""` if no meaningful alias exists. Do not include the original term itself.

### User prompt per batch
Numbered list of N descriptions, asking for a JSON array of exactly N strings.

### Functions

**`load_checkpoint(path) -> dict[str, str]`**
Read `aliases_checkpoint.jsonl`, return `{code: aliases_str}`. Skip corrupt lines.

**`append_checkpoint(path, code, aliases)`**
Append one JSON line per processed row — called immediately after each batch so a crash loses at most one batch.

**`fetch_aliases_for_batch(client, descriptions) -> list[str]`**
Call `client.messages.create(...)`, parse JSON response. Retry up to 3x on `APIError`/`RateLimitError` with exponential backoff. On repeated parse failures, fall back to `[""] * len(descriptions)` and log a warning (don't halt the run).

**`main()`**
1. Validate `ANTHROPIC_API_KEY` env var
2. Load CSV with `pandas.read_csv`
3. Load checkpoint → skip already-done codes
4. Iterate remaining rows in batches, calling `fetch_aliases_for_batch`, then `append_checkpoint`
5. Reload full checkpoint, `df["description_aliases"] = df["ICD10-CM-CODE"].map(completed).fillna("")`
6. Reorder columns: `ICD10-CM-CODE, description, description_aliases, category_code, category_description, section, chapter`
7. Write to `OUTPUT_CSV`

---

## Column format

`description_aliases` stores a **semicolon-separated string** (e.g., `"high blood pressure;elevated blood pressure"`). This is human-readable in Excel and avoids CSV quoting conflicts. Empty string for codes with no useful aliases.

---

## Resume behaviour

Checkpoint keyed on `ICD10-CM-CODE`. Re-running the script after interruption skips already-checkpointed rows and continues. To force a full re-run: delete `data/processed/aliases_checkpoint.jsonl`.

---

## Estimated cost

~997 API calls × Haiku pricing ≈ **~$2 total** for all 74,719 rows.

---

## Verification

```bash
# 1. Add anthropic, sync deps
uv sync

# 2. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run (resumes if interrupted)
uv run python src/scripts/add_description_aliases.py

# 4. Spot-check output
python -c "
import pandas as pd
df = pd.read_csv('data/processed/icd10cm-codes-enriched-v2-April-1-2026.csv')
print(df.columns.tolist())
print(df[df['description'].str.contains('hypertension', case=False)][['description','description_aliases']].head())
print(f'Rows: {len(df):,}, aliases non-empty: {(df[\"description_aliases\"] != \"\").sum():,}')
"
```

Expected: 74,719 rows, `description_aliases` column present, hypertension rows show "high blood pressure" variant aliases.
