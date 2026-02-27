# Multi-Label ICD-10 Retrieval from Clinical Notes — v3

**Date:** 2026-02-26
**Supersedes:** `multi-label-icd10-retrieval-v2.md` (v2)
**Key change in v3:** Handling 3-character category codes in ground truth predictions.

---

## 0. The Category Code Problem

### Discovery

The CodiEsp ground truth contains **48 unique 3-character codes** (428 assignments, 12.5% of all code assignments). These codes fall into two groups:

| Group | Count | In KB? | Retrievable? | Example |
|-------|-------|--------|-------------|---------|
| 3-char billable codes | 43 | Yes (as `ICD10-CM-CODE`) | Yes, directly | `I10` (Essential hypertension), `R52` (Pain, unspecified) |
| 3-char category-only codes | 5 | No (only in `category_code` column) | **No** — not in retriever index | `G35` (Multiple sclerosis), `R05` (Cough) |

**Impact:** 174 of 250 dev documents (70%) contain at least one 3-char code. 26 documents contain one of the 5 category-only codes that the retriever cannot currently predict.

### The 5 unreachable category codes

These exist in the categories CSV (`category_code` column) but NOT as rows in the KB billable codes:

| Category | Description (from categories CSV) | Children in KB | Child examples |
|----------|----------------------------------|----------------|----------------|
| `G35` | Multiple sclerosis | 8 | `G35A`, `G35B0`, `G35B1`, ... |
| `H17` | Corneal scars and opacities | 18 | `H1700`, `H1701`, `H1702`, ... |
| `N63` | Unspecified lump in breast | 17 | `N630`, `N6310`, `N6311`, ... |
| `R05` | Cough | 6 | `R051`, `R052`, `R053`, ... |
| `R51` | Headache | 2 | `R510`, `R519` |

### Why this happens

In ICD-10-CM, some 3-character codes are **billable** (they have no children, e.g., `I10` = Essential hypertension — you submit `I10` on a claim). Others are **non-billable category headers** (e.g., `G35` = Multiple sclerosis — you must submit a child like `G35A`).

The CodiEsp annotators appear to have used category codes when the clinical note lacked enough specificity to assign a more granular child code. This is common in clinical coding practice — "code to the highest level of specificity documented."

---

## 1. Three Strategies for Category Codes

### Strategy 1: Expand the KB to Include Category Descriptions (Recommended)

**Idea:** Add rows for all 1,918 categories (or just the 5 missing ones) to the `KnowledgeBase`. Each row uses the category code as `ICD10-CM-CODE` and the category description as `description`. The TF-IDF retriever then indexes these alongside the 74k billable codes.

**Pros:**
- Simplest conceptual change
- The retriever can now directly predict `G35` if "multiple sclerosis" appears in the note
- No post-processing logic needed
- Works with all pipeline options (A through E) unchanged

**Cons:**
- Increases KB from 74,719 → ~76,637 entries (trivial impact on TF-IDF performance)
- May slightly dilute retrieval quality if a category description is very similar to a child description (e.g., "Cough" vs "Cough, unspecified")

```python
# src/pipeline_components.py — add to NoteLoaderTransformer section

import polars as pl
from pathlib import Path
from knowledge_base import KnowledgeBase, ICD10Code, Category

CATEGORIES_PATH = Path("data/raw/icd10cm-categories-April-1-2026.csv")


def expand_kb_with_categories(
    kb: KnowledgeBase,
    categories_path: Path = CATEGORIES_PATH,
) -> KnowledgeBase:
    """
    Add category-level entries to the KnowledgeBase so the retriever
    can directly predict 3-character category codes.

    Category entries are only added when the category code does NOT already
    exist as a billable code in the KB. This avoids duplicating codes like
    I10 that are both a category and a billable code.

    Parameters
    ----------
    kb : KnowledgeBase
        The original KB with 74,719 billable code entries.
    categories_path : Path
        Path to the categories CSV with columns:
        category_code, category_description, section, chapter

    Returns
    -------
    KnowledgeBase
        A new KB with additional entries for non-billable categories.
    """
    # Load categories CSV
    cats_df = pl.read_csv(categories_path)

    # Codes already in the KB (both billable 3-char codes and all longer codes)
    existing_codes = {e.code for e in kb.entries}

    # Only add categories that are NOT already billable codes
    new_entries: list[ICD10Code] = []
    for row in cats_df.iter_rows(named=True):
        cat_code = row["category_code"]
        if cat_code not in existing_codes:
            new_entries.append(
                ICD10Code(
                    code=cat_code,
                    description=row["category_description"],
                    description_aliases=[],
                    category=Category(code=cat_code, description=row["category_description"]),
                    chapter=None,  # or populate from CSV if available
                )
            )

    # Build a new KB with the extra entries
    expanded = KnowledgeBase.__new__(KnowledgeBase)
    expanded.file_path = kb.file_path
    expanded.entries = kb.entries + new_entries
    print(f"Expanded KB: {len(kb.entries)} → {len(expanded.entries)} entries "
          f"(+{len(new_entries)} category-only codes)")
    return expanded
```

**Integration with any pipeline:**

```python
# In any build_pipeline_* function, expand the KB before passing to predictor:
kb = KnowledgeBase(KB_PATH)
kb = expand_kb_with_categories(kb)  # <-- one line change

pipeline = build_pipeline_e(kb, TEXT_DIR)
```

### Strategy 2: Post-Prediction Roll-Up (Child → Category)

**Idea:** After the retriever predicts a child code (e.g., `G35A`), check whether the ground truth expects the parent category code (`G35`). In evaluation, "roll up" predicted child codes to their category prefix and count a match if the category matches.

This changes the **evaluation logic**, not the retriever. There are two sub-variants:

#### 2a: Soft evaluation — accept category match

```python
def soft_match_predicted_to_gold(
    predicted: set[str],
    gold: set[str],
) -> tuple[set[str], set[str]]:
    """
    For each 3-char gold code that is a category (not billable),
    if the predicted set contains ANY child of that category,
    consider it a match by adding the category code to predictions.

    Returns (expanded_predicted, gold) for evaluation.
    """
    # Identify gold category codes that have no direct match in predictions
    expanded = set(predicted)
    for gold_code in gold:
        if len(gold_code) == 3 and gold_code not in predicted:
            # Check if any predicted code starts with this category
            if any(p.startswith(gold_code) for p in predicted):
                expanded.add(gold_code)
    return expanded, gold
```

**Usage in evaluation:**

```python
pairs = []
for pred, stem in zip(predictions, stems):
    gold = gold_map[stem]
    expanded_pred, gold = soft_match_predicted_to_gold(pred, gold)
    pairs.append((expanded_pred, gold))
metrics = macro_average(pairs)
```

**Pros:** Zero model changes; more forgiving evaluation that rewards partial specificity.
**Cons:** The retriever never actually outputs the category code; you're inflating precision/recall by post-hoc matching. Doesn't help in production where you need to output the actual code the nurse should select.

#### 2b: Post-prediction code rolling

```python
def roll_up_to_categories(
    predicted: set[str],
    category_only_codes: set[str],
) -> set[str]:
    """
    For codes whose 3-char prefix is a category-only code (not billable),
    add the category to the prediction set.

    This handles the case where the retriever predicts G35A but the
    ground truth expects G35. We emit BOTH G35A and G35.

    Parameters
    ----------
    predicted : set[str]
        The raw predicted codes from the retriever.
    category_only_codes : set[str]
        The set of 3-char codes that are categories but NOT billable.
        Pre-compute this once from the KB.
    """
    rolled = set(predicted)
    for code in predicted:
        prefix = code[:3]
        if prefix in category_only_codes:
            rolled.add(prefix)
    return rolled


# Pre-compute category-only codes (do this once at startup):
def get_category_only_codes(kb: KnowledgeBase, categories_path: Path) -> set[str]:
    """Return 3-char codes that are categories but NOT billable."""
    cats_df = pl.read_csv(categories_path)
    all_categories = set(cats_df["category_code"].to_list())
    billable = {e.code for e in kb.entries}
    return all_categories - billable
```

**Pros:** Captures the annotator's intent; emits the category code when a child is found.
**Cons:** Adds a post-processing step; may add false positives (predicting `G35A` for "sclerosis" would roll up to `G35` even if the note has nothing to do with MS).

### Strategy 3: Dual-Index Search (Separate Category Retriever)

**Idea:** Build TWO TF-IDF indexes: one over billable code descriptions (existing), one over the 1,918 category descriptions. For each note, search both indexes. Merge the results.

```
note text
    │
    ├──→ TF-IDF index A (74k billable codes) → predicted billable codes
    │
    └──→ TF-IDF index B (1.9k categories)    → predicted category codes
                                                 (keep only 3-char hits)
    │
    ▼ merge + deduplicate
    │
    ▼ predicted code set
```

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import polars as pl
from pathlib import Path

from knowledge_base import KnowledgeBase


class DualIndexPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label predictor with two TF-IDF indexes:
      1. Billable code descriptions (74k entries)
      2. Category descriptions (1.9k entries)

    Merges top-k results from both indexes.

    Parameters
    ----------
    kb : KnowledgeBase
        The original knowledge base (billable codes).
    categories_path : Path
        Path to the categories CSV.
    top_k_codes : int
        Candidates from the billable code index.
    top_k_categories : int
        Candidates from the category index.
    code_threshold : float
        Minimum cosine score for billable codes.
    category_threshold : float
        Minimum cosine score for category codes.
        Should typically be higher than code_threshold since category
        descriptions are shorter and more generic.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        categories_path: Path,
        top_k_codes: int = 30,
        top_k_categories: int = 10,
        code_threshold: float = 0.04,
        category_threshold: float = 0.10,
    ) -> None:
        self.kb = kb
        self.categories_path = categories_path
        self.top_k_codes = top_k_codes
        self.top_k_categories = top_k_categories
        self.code_threshold = code_threshold
        self.category_threshold = category_threshold

    def fit(self, X: list[str], y=None) -> "DualIndexPredictor":
        # --- Index 1: Billable codes ---
        self._code_list = [e.code for e in self.kb.entries]
        self._code_descs = [e.description for e in self.kb.entries]
        self._code_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b(?:\w\w+|\d)\b",
        )
        self._code_matrix = self._code_vec.fit_transform(self._code_descs)

        # --- Index 2: Categories ---
        cats_df = pl.read_csv(self.categories_path)
        self._cat_codes = cats_df["category_code"].to_list()
        self._cat_descs = cats_df["category_description"].to_list()
        self._cat_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b(?:\w\w+|\d)\b",
        )
        self._cat_matrix = self._cat_vec.fit_transform(self._cat_descs)

        self.classes_ = np.array(self._code_list + self._cat_codes)
        return self

    def _search_index(
        self, query_text: str, vectorizer, matrix, code_list, top_k, threshold
    ) -> set[str]:
        """Search a single TF-IDF index and return codes above threshold."""
        q = vectorizer.transform([query_text])
        scores = linear_kernel(q, matrix).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        return {
            code_list[i]
            for i in top_idx
            if scores[i] >= threshold
        }

    def predict(self, X: list[str]) -> list[set[str]]:
        results: list[set[str]] = []
        for text in X:
            # Search billable codes
            code_hits = self._search_index(
                text, self._code_vec, self._code_matrix,
                self._code_list, self.top_k_codes, self.code_threshold,
            )
            # Search categories
            cat_hits = self._search_index(
                text, self._cat_vec, self._cat_matrix,
                self._cat_codes, self.top_k_categories, self.category_threshold,
            )
            # Merge both sets
            results.append(code_hits | cat_hits)
        return results
```

**Pros:** Separate thresholds for categories vs. billable codes; full control.
**Cons:** More complexity; two indexes to maintain; category descriptions are often just 1–3 words, which makes TF-IDF less effective for them.

---

## 2. Recommendation

### Use Strategy 1 (Expand KB) + Strategy 2b (Roll-Up) together

These two strategies are complementary and non-conflicting:

| Strategy | What it solves |
|----------|---------------|
| Strategy 1: Expand KB | Makes the 5 missing category codes directly searchable via their descriptions |
| Strategy 2b: Roll-up | Catches cases where the retriever finds a child code (e.g., `G35B0`) but the annotator used the category (`G35`). Also helps when the expanded KB matches the category directly |

**Combined pipeline flow:**

```
clinical note
       │
       ▼  NoteLoaderTransformer
       │
       ▼  TextCleaner
       │
       ▼  SlidingWindowChunker           (from Option E in v2)
       │
       ▼  ChunkedRetrievalPredictor      searches the EXPANDED KB
       │                                 (74,719 billable + ~1,700 category-only entries)
       │
       ▼  raw predicted set              e.g., {I10, G35, G35A, N390, ...}
       │
       ▼  CategoryRollUpTransformer      for any predicted child whose parent
       │                                 is a category-only code, add the parent
       │
       ▼  final predicted set            e.g., {I10, G35, G35A, N390, ...}
```

### Full Pipeline Code (Strategy 1 + 2b + Option E)

```python
# -----------------------------------------------------------------------
# src/pipeline_v3.py
# Multi-label ICD-10 retrieval pipeline with category code handling.
#
# Combines:
#   - Strategy 1: Expanded KB (adds category-only codes to the index)
#   - Strategy 2b: Post-prediction roll-up (child → category)
#   - Option E: Sliding-window chunk retrieval
# -----------------------------------------------------------------------

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from knowledge_base import KnowledgeBase, ICD10Code, Category
from retriever import TfidfRetriever


# ===================================================================
# Project paths (relative to project root)
# ===================================================================

PROJECT_ROOT     = Path(__file__).parent.parent
KB_PATH          = PROJECT_ROOT / "data" / "processed" / "icd10cm-codes-enriched-April-1-2026.csv"
CATEGORIES_PATH  = PROJECT_ROOT / "data" / "raw" / "icd10cm-categories-April-1-2026.csv"
GT_PATH          = PROJECT_ROOT / "data" / "test-datasets" / "codiesp" / "data-pipeline" / "processed" / "gold" / "codiesp_ground_truth.parquet"
TEXT_DIR         = PROJECT_ROOT / "data" / "test-datasets" / "codiesp" / "gold" / "final_dataset_v4_to_publish" / "dev" / "text_files_en"


# ===================================================================
# Step 0: KB expansion — add category-only codes
# ===================================================================

def expand_kb_with_categories(
    kb: KnowledgeBase,
    categories_path: Path = CATEGORIES_PATH,
) -> KnowledgeBase:
    """
    Add category-level entries to the KnowledgeBase.

    Only adds categories whose 3-char code does NOT already appear
    as a billable code in the KB. This avoids duplicating codes like
    I10 that are both a category header and a billable code.

    After expansion the retriever can directly predict category codes
    like G35, H17, N63, R05, R51 that appear in the CodiEsp ground truth.

    Parameters
    ----------
    kb : KnowledgeBase
        Original KB (74,719 billable code entries).
    categories_path : Path
        Categories CSV with columns: category_code, category_description, section, chapter.

    Returns
    -------
    KnowledgeBase
        New KB with additional entries for non-billable categories.
        Original entries are preserved unmodified.
    """
    cats_df = pl.read_csv(categories_path)
    existing_codes = {e.code for e in kb.entries}

    new_entries: list[ICD10Code] = []
    for row in cats_df.iter_rows(named=True):
        cat_code = str(row["category_code"])
        if cat_code not in existing_codes:
            new_entries.append(
                ICD10Code(
                    code=cat_code,
                    description=str(row["category_description"]),
                    description_aliases=[],
                    category=Category(code=cat_code, description=str(row["category_description"])),
                    chapter=None,
                )
            )

    expanded = KnowledgeBase.__new__(KnowledgeBase)
    expanded.file_path = kb.file_path
    expanded.entries = kb.entries + new_entries
    print(
        f"KB expanded: {len(kb.entries):,} → {len(expanded.entries):,} entries "
        f"(+{len(new_entries)} category-only codes)"
    )
    return expanded


def get_category_only_codes(
    kb: KnowledgeBase,
    categories_path: Path = CATEGORIES_PATH,
) -> set[str]:
    """
    Return the set of 3-char codes that are ICD-10-CM categories
    but NOT billable codes in the KB.

    Used by CategoryRollUpTransformer to know which prefixes to roll up.
    """
    cats_df = pl.read_csv(categories_path)
    all_categories = set(cats_df["category_code"].to_list())
    billable_codes = {e.code for e in kb.entries}
    return all_categories - billable_codes


# ===================================================================
# Pipeline step 1: Load text from disk
# ===================================================================

class NoteLoaderTransformer(BaseEstimator, TransformerMixin):
    """
    Transform file stems → raw text strings.

    Input  X : list[str]   file stems (e.g., 'S0004-06142005000900016-1')
    Output   : list[str]   UTF-8 text contents

    Parameters
    ----------
    text_dir : Path
        Directory containing <file_stem>.txt files.
    """

    def __init__(self, text_dir: Path) -> None:
        self.text_dir = text_dir

    def fit(self, X: list[str], y=None) -> "NoteLoaderTransformer":
        return self  # stateless

    def transform(self, X: list[str]) -> list[str]:
        return [
            (Path(self.text_dir) / f"{stem}.txt").read_text(
                encoding="utf-8", errors="replace"
            )
            for stem in X
        ]


# ===================================================================
# Pipeline step 2: Normalize text
# ===================================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Collapse whitespace and optionally lowercase.

    Input  X : list[str]
    Output   : list[str]
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def fit(self, X: list[str], y=None) -> "TextCleaner":
        return self

    def transform(self, X: list[str]) -> list[str]:
        result: list[str] = []
        for text in X:
            text = re.sub(r"\s+", " ", text).strip()
            if self.lowercase:
                text = text.lower()
            result.append(text)
        return result


# ===================================================================
# Pipeline step 3: Sliding-window chunker
# ===================================================================

class SlidingWindowChunker(BaseEstimator, TransformerMixin):
    """
    Split each text into overlapping sentence-window chunks.

    Input  X : list[str]          one text per document
    Output   : list[list[str]]    one list-of-chunks per document

    Parameters
    ----------
    window : int
        Number of sentences per chunk.
    stride : int
        Step between chunk starts. stride < window → overlap.
    min_words : int
        Minimum tokens for a sentence to be included.
    include_full_text : bool
        Append the full document as an extra chunk (preserves global context).
    """

    def __init__(
        self,
        window: int = 3,
        stride: int = 2,
        min_words: int = 3,
        include_full_text: bool = True,
    ) -> None:
        self.window = window
        self.stride = stride
        self.min_words = min_words
        self.include_full_text = include_full_text

    def fit(self, X: list[str], y=None) -> "SlidingWindowChunker":
        return self

    def transform(self, X: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for text in X:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", text.strip())
                if len(s.split()) >= self.min_words
            ]
            chunks: list[str] = []
            for i in range(0, max(1, len(sentences) - self.window + 1), self.stride):
                chunk = " ".join(sentences[i : i + self.window])
                if chunk:
                    chunks.append(chunk)
            if self.include_full_text:
                chunks.append(text)
            result.append(chunks)
        return result


# ===================================================================
# Pipeline step 4: Chunked retrieval predictor (searches expanded KB)
# ===================================================================

class ChunkedRetrievalPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label predictor: TF-IDF search per chunk, aggregate, threshold.

    fit()    → builds TF-IDF index over KB entries (including category entries
               if the KB was expanded with expand_kb_with_categories()).
    predict() → for each document's chunks, retrieves top_k codes per chunk,
                aggregates scores across chunks, filters by votes + threshold.

    Parameters
    ----------
    kb : KnowledgeBase
        Pre-loaded (and optionally expanded) knowledge base.
    top_k_per_chunk : int
        How many candidate codes to retrieve per chunk.
    min_chunk_score : float
        Per-chunk cosine score floor. Hits below this are discarded.
    min_votes : int
        A code must appear in at least this many chunks to survive.
    aggregation : str
        "max"  → best single-chunk score (good for sparse mentions)
        "mean" → average across chunks where the code appeared
        "sum"  → total score (rewards repeated mentions)
    final_threshold : float
        Aggregated score must exceed this to appear in the prediction.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        top_k_per_chunk: int = 10,
        min_chunk_score: float = 0.04,
        min_votes: int = 1,
        aggregation: str = "max",
        final_threshold: float = 0.04,
    ) -> None:
        self.kb = kb
        self.top_k_per_chunk = top_k_per_chunk
        self.min_chunk_score = min_chunk_score
        self.min_votes = min_votes
        self.aggregation = aggregation
        self.final_threshold = final_threshold
        self._retriever: TfidfRetriever | None = None

    def fit(self, X: list[list[str]], y=None) -> "ChunkedRetrievalPredictor":
        # TfidfRetriever.__init__ fits TF-IDF on kb.entries descriptions.
        # If kb was expanded, the index now includes category descriptions.
        self._retriever = TfidfRetriever(self.kb)
        self.classes_ = np.array([e.code for e in self.kb.entries])
        return self

    def _predict_one(self, chunks: list[str]) -> set[str]:
        assert self._retriever is not None
        code_scores: dict[str, list[float]] = defaultdict(list)
        for chunk in chunks:
            hits = self._retriever.search(chunk, top_k=self.top_k_per_chunk)
            for h in hits:
                if h.score >= self.min_chunk_score:
                    code_scores[h.code].append(h.score)

        # Filter by vote count
        filtered = {
            code: scores
            for code, scores in code_scores.items()
            if len(scores) >= self.min_votes
        }

        # Aggregate
        aggregated: dict[str, float] = {}
        for code, scores in filtered.items():
            if self.aggregation == "max":
                aggregated[code] = max(scores)
            elif self.aggregation == "mean":
                aggregated[code] = sum(scores) / len(scores)
            else:  # "sum"
                aggregated[code] = sum(scores)

        return {
            code for code, score in aggregated.items()
            if score >= self.final_threshold
        }

    def predict(self, X: list[list[str]]) -> list[set[str]]:
        return [self._predict_one(chunks) for chunks in X]


# ===================================================================
# Pipeline step 5: Category roll-up (Strategy 2b)
# ===================================================================

class CategoryRollUpTransformer(BaseEstimator, TransformerMixin):
    """
    Post-prediction step: when a predicted code's 3-char prefix is a
    category-only code (not billable), add that category to the prediction set.

    This handles annotator behavior where the ground truth uses the category
    code (e.g., G35) but the retriever finds a child (e.g., G35A).

    Example:
        Input:  {'G35A', 'I10', 'N390'}
        Output: {'G35A', 'G35', 'I10', 'N390'}
        (G35 added because G35 is a category-only code and G35A starts with G35)
        (I10 not duplicated because I10 IS a billable code, not category-only)

    Parameters
    ----------
    category_only_codes : set[str]
        The set of 3-char codes that are categories but not billable.
        Pre-compute with get_category_only_codes().
    """

    def __init__(self, category_only_codes: set[str]) -> None:
        self.category_only_codes = category_only_codes

    def fit(self, X: list[set[str]], y=None) -> "CategoryRollUpTransformer":
        return self

    def transform(self, X: list[set[str]]) -> list[set[str]]:
        result: list[set[str]] = []
        for pred_set in X:
            rolled = set(pred_set)
            for code in pred_set:
                prefix = code[:3]
                # Only roll up if the prefix is a non-billable category
                if prefix in self.category_only_codes:
                    rolled.add(prefix)
            result.append(rolled)
        return result


# ===================================================================
# Evaluation helpers
# ===================================================================

def precision_recall_f1(predicted: set[str], gold: set[str]) -> dict[str, float]:
    """Per-document precision, recall, F1."""
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gold)      if gold      else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_average(pairs: list[tuple[set[str], set[str]]]) -> dict[str, float]:
    """Macro-average P/R/F1 over (predicted, gold) pairs."""
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, gold in pairs:
        m = precision_recall_f1(pred, gold)
        for k in agg:
            agg[k] += m[k]
    n = len(pairs)
    return {k: round(v / n, 4) for k, v in agg.items()}


def load_gold_map(gt_path: Path = GT_PATH) -> dict[str, set[str]]:
    """Load ground truth → {file_stem: set_of_unique_codes}."""
    df = pl.read_parquet(gt_path)
    return {
        row["file_stem"]: set(row["codes"].split(";"))
        for row in df.iter_rows(named=True)
    }


# ===================================================================
# Pipeline builder
# ===================================================================

def build_pipeline(
    kb: KnowledgeBase,
    text_dir: Path = TEXT_DIR,
    categories_path: Path = CATEGORIES_PATH,
    # Chunker params
    window: int = 3,
    stride: int = 2,
    # Retrieval params
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
    final_threshold: float = 0.04,
    # Category handling
    expand_kb: bool = True,
    enable_rollup: bool = True,
) -> Pipeline:
    """
    Build the full end-to-end v3 pipeline.

    Input:  list[str]       file stems
    Output: list[set[str]]  predicted ICD-10 code sets

    Parameters
    ----------
    kb : KnowledgeBase
        Original KB (will be expanded if expand_kb=True).
    text_dir : Path
        Directory with clinical note .txt files.
    categories_path : Path
        Categories CSV for KB expansion and roll-up.
    expand_kb : bool
        If True, add non-billable category codes to the KB index (Strategy 1).
    enable_rollup : bool
        If True, add a post-prediction CategoryRollUpTransformer (Strategy 2b).
    """
    # --- Optionally expand the KB ----------------------------------------
    if expand_kb:
        kb = expand_kb_with_categories(kb, categories_path)

    # --- Build pipeline steps --------------------------------------------
    steps: list[tuple[str, BaseEstimator]] = [
        ("loader",    NoteLoaderTransformer(text_dir)),
        ("cleaner",   TextCleaner(lowercase=True)),
        ("chunker",   SlidingWindowChunker(window=window, stride=stride)),
        ("predictor", ChunkedRetrievalPredictor(
            kb,
            top_k_per_chunk=top_k_per_chunk,
            min_chunk_score=min_chunk_score,
            min_votes=min_votes,
            aggregation=aggregation,
            final_threshold=final_threshold,
        )),
    ]

    # --- Optionally add category roll-up ---------------------------------
    if enable_rollup:
        # We use the ORIGINAL kb (before expansion) to find category-only codes,
        # because after expansion those codes ARE in the KB and would not appear
        # in the category_only_codes set.
        cat_only = get_category_only_codes(
            KnowledgeBase(KB_PATH),  # original KB
            categories_path,
        )
        steps.append(("rollup", CategoryRollUpTransformer(cat_only)))

    return Pipeline(steps=steps, verbose=True)


# ===================================================================
# End-to-end evaluation
# ===================================================================

def evaluate(
    pipeline: Pipeline,
    stems: list[str],
    gold_map: dict[str, set[str]],
    label: str = "",
) -> dict[str, float]:
    """Run pipeline.predict() and compute macro-avg metrics."""
    predictions: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    metrics = macro_average(pairs)
    if label:
        print(f"{label:<40}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    return metrics


# ===================================================================
# Main: run with multiple configurations
# ===================================================================

if __name__ == "__main__":
    import itertools

    # --- Load resources --------------------------------------------------
    kb_original = KnowledgeBase(KB_PATH)
    gold_map    = load_gold_map()
    stems       = sorted(gold_map.keys())

    print("=" * 72)
    print("Multi-Label ICD-10 Retrieval — v3 Evaluation")
    print("=" * 72)
    print(f"Documents: {len(stems)}")
    print(f"KB size (original): {len(kb_original.entries):,}")
    print()

    # --- Ablation: test each strategy separately -------------------------
    print("--- Ablation: Strategy Impact ---")

    # Baseline: no expansion, no rollup
    p1 = build_pipeline(kb_original, expand_kb=False, enable_rollup=False)
    p1.fit(stems)  # type: ignore[arg-type]
    evaluate(p1, stems, gold_map, label="Baseline (no expansion, no rollup)")

    # Strategy 1 only: expand KB
    p2 = build_pipeline(kb_original, expand_kb=True, enable_rollup=False)
    p2.fit(stems)  # type: ignore[arg-type]
    evaluate(p2, stems, gold_map, label="+ Strategy 1 (expand KB)")

    # Strategy 2b only: rollup
    p3 = build_pipeline(kb_original, expand_kb=False, enable_rollup=True)
    p3.fit(stems)  # type: ignore[arg-type]
    evaluate(p3, stems, gold_map, label="+ Strategy 2b (rollup only)")

    # Both: expand + rollup
    p4 = build_pipeline(kb_original, expand_kb=True, enable_rollup=True)
    p4.fit(stems)  # type: ignore[arg-type]
    evaluate(p4, stems, gold_map, label="+ Strategy 1 + 2b (expand + rollup)")

    # --- Parameter sweep on the best config ------------------------------
    print()
    print("--- Parameter Sweep (expand + rollup) ---")
    pipeline = build_pipeline(kb_original, expand_kb=True, enable_rollup=True)
    pipeline.fit(stems)  # type: ignore[arg-type]

    best_f1, best_cfg = 0.0, {}
    for window, top_k, threshold in itertools.product(
        [2, 3, 5],
        [5, 10, 20],
        [0.03, 0.05, 0.08],
    ):
        pipeline.set_params(
            chunker__window=window,
            predictor__top_k_per_chunk=top_k,
            predictor__final_threshold=threshold,
        )
        m = evaluate(pipeline, stems, gold_map,
                     label=f"  w={window} k={top_k} t={threshold:.2f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_cfg = {"window": window, "top_k": top_k, "threshold": threshold, **m}

    print()
    print(f"Best: {best_cfg}")
```

---

## 3. Soft Evaluation Variant (Bonus)

For reporting purposes, you may also want a "soft" evaluation mode that gives partial
credit for child→category matches even without roll-up. This is useful to understand how
much the retriever is "almost right" when it predicts a child code but the gold expects
the category.

```python
def soft_match_evaluation(
    predicted: set[str],
    gold: set[str],
    category_only_codes: set[str],
) -> dict[str, float]:
    """
    Evaluate with soft matching: for any gold code that is a category-only code,
    a predicted child code counts as a match.

    This does NOT modify predictions — it modifies how we COUNT matches.

    Example:
        gold = {'G35', 'I10'}
        pred = {'G35A', 'I10'}
        strict:  TP=1 (I10)   precision=1/2  recall=1/2
        soft:    TP=2 (I10 + G35 matched by G35A)  precision=2/2  recall=2/2
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = 0
    matched_preds: set[str] = set()

    for g in gold:
        if g in predicted:
            # Direct match
            tp += 1
            matched_preds.add(g)
        elif len(g) == 3 and g in category_only_codes:
            # Soft match: check if any predicted code is a child of this category
            child_match = next((p for p in predicted if p.startswith(g) and p not in matched_preds), None)
            if child_match:
                tp += 1
                matched_preds.add(child_match)

    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gold)      if gold      else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

## 4. Decision Log

| Decision | Options considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| How to handle 5 missing category codes | (1) Expand KB, (2) Roll-up, (3) Dual-index | 1 + 2 combined | Expansion makes codes directly searchable; roll-up catches child→category aliasing. Both are simple, composable. Dual-index adds complexity for marginal gain. |
| Where to place roll-up in pipeline | (a) Inside predictor, (b) As a separate pipeline step | (b) Separate step | Keeps predictor logic clean; can toggle roll-up via `enable_rollup` param; follows sklearn Pipeline composability. |
| KB expansion scope | (a) Only 5 missing codes, (b) All ~1,700 non-billable categories | (b) All categories | Minimal cost (1,700 extra TF-IDF rows is trivial); makes the system robust to future GT that uses other category codes. |
| Roll-up targets | (a) All 3-char prefixes, (b) Only category-only prefixes | (b) Only category-only | Rolling up to billable 3-char codes (like I10) would create spurious duplicates — I10 is already a directly predictable code. |

---

## 5. Summary of What Changed from v2

| Aspect | v2 | v3 |
|--------|----|----|
| KB contents | 74,719 billable codes only | 74,719 billable + ~1,700 category-only codes |
| Category handling | None | `expand_kb_with_categories()` + `CategoryRollUpTransformer` |
| Pipeline steps | 4 (loader → cleaner → chunker → predictor) | 5 (+ rollup step) |
| Evaluation | Single mode | Ablation: baseline / expand / rollup / both |
| Soft evaluation | Not available | `soft_match_evaluation()` for analysis |

---

## 6. File Layout

```
src/
  knowledge_base.py          # existing (unchanged)
  retriever.py               # existing (unchanged)
  pipeline_v3.py             # NEW: complete pipeline with all strategies

plan/thoughts/
  multi-label-icd10-retrieval.md      # v1 (historical)
  multi-label-icd10-retrieval-v2.md   # v2 (historical)
  multi-label-icd10-retrieval-v3.md   # v3 (this document)
```
