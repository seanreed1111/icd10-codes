# Multi-Label ICD-10 Retrieval from Clinical Notes — v2

**Date:** 2026-02-26
**Supersedes:** `multi-label-icd10-retrieval.md` (v1)

---

## 0. Context and Data Facts

### Problem

Each clinical note in the CodiEsp dev set has **multiple** correct ICD-10-CM codes.
The existing `TfidfRetriever.search()` is designed for a single short query → single best code.
We need to predict the **full set** of codes for a given note.

| Metric | Value |
|--------|-------|
| Dev documents | 250 |
| Mean unique codes per doc | 10.7 |
| Min / Max codes per doc | 1 / 33 |
| KnowledgeBase size | 74,719 billable codes |
| KB columns | `ICD10-CM-CODE`, `description`, `category_code`, `category_description`, `section`, `chapter` |

### File paths (project-relative)

```
data/
  processed/
    icd10cm-codes-enriched-April-1-2026.csv   # 74,719 rows, knowledge base source
  test-datasets/codiesp/
    data-pipeline/processed/gold/
      codiesp_ground_truth.parquet             # {file_stem: str, codes: str (;-delimited)}
    gold/final_dataset_v4_to_publish/dev/
      text_files_en/                           # 250 clinical notes (*.txt)
src/
  knowledge_base.py    # KnowledgeBase, ICD10Code, Category dataclasses
  retriever.py         # TfidfRetriever (single-label)
```

### Ground truth schema

```python
# Schema: {'file_stem': String, 'codes': String}
# 'codes' is a semicolon-separated string, may contain duplicates.
# Example row:
#   file_stem : 'S0004-06142005000900016-1'
#   codes     : 'Q6211;N2889;N390;R319;N23;N280;Q6211;D1809;N135;K269;N289;N200;K5900'
#   unique set: {'D1809', 'K269', 'K5900', 'N135', 'N200', 'N23', 'N280', 'N289', 'N390', 'Q6211', 'R319'}
```

---

## 1. Design Principle: sklearn `Pipeline` Throughout

Every option below is structured as a `sklearn.pipeline.Pipeline` that flows:

```
file_stems (list[str])
      │
      ▼  NoteLoaderTransformer          reads .txt → list[str]
      │
      ▼  TextCleaner                    normalize whitespace, strip
      │
      ▼  <option-specific step(s)>      vectorize / embed / retrieve
      │
      ▼  <predictor step>               → list[set[str]]  predicted code sets
```

The pipeline is always called as:

```python
pipeline.fit(file_stems)          # unsupervised, or:
pipeline.fit(file_stems, Y_bin)   # supervised (Option C)

predictions: list[set[str]] = pipeline.predict(dev_stems)
```

### Shared base transformers (used by all options)

```python
# src/pipeline_components.py
"""
Reusable sklearn-compatible transformers for the ICD-10 multi-label pipeline.

All transformers follow the sklearn TransformerMixin contract:
    fit(X, y=None)  → self
    transform(X)    → transformed X
so they compose seamlessly inside sklearn.pipeline.Pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# ---------------------------------------------------------------------------
# Step 1 — Load raw text from disk
# ---------------------------------------------------------------------------

class NoteLoaderTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a list of file stems into a list of raw text strings.

    Input  X : list[str]  — file stem names, e.g. 'S0004-06142005000900016-1'
    Output   : list[str]  — UTF-8 decoded file contents

    Parameters
    ----------
    text_dir : Path
        Directory containing .txt files named <file_stem>.txt
    """

    def __init__(self, text_dir: Path) -> None:
        self.text_dir = text_dir

    def fit(self, X: list[str], y=None) -> "NoteLoaderTransformer":
        # Stateless — nothing to learn from the data.
        return self

    def transform(self, X: list[str]) -> list[str]:
        texts: list[str] = []
        for stem in X:
            path = Path(self.text_dir) / f"{stem}.txt"
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        return texts


# ---------------------------------------------------------------------------
# Step 2 — Normalize text
# ---------------------------------------------------------------------------

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Lightweight clinical text normalizer.

    - Collapses runs of whitespace / newlines to a single space
    - Strips leading/trailing whitespace
    - Optionally lowercases (default: True)

    Input  X : list[str]
    Output   : list[str]
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def fit(self, X: list[str], y=None) -> "TextCleaner":
        return self

    def transform(self, X: list[str]) -> list[str]:
        cleaned: list[str] = []
        for text in X:
            text = re.sub(r"\s+", " ", text).strip()
            if self.lowercase:
                text = text.lower()
            cleaned.append(text)
        return cleaned


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(predicted: set[str], gold: set[str]) -> dict[str, float]:
    """
    Compute micro precision, recall, and F1 for a single document.

    Edge case: both empty → perfect score (the model correctly predicted nothing).
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gold)      if gold      else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_average(pairs: list[tuple[set[str], set[str]]]) -> dict[str, float]:
    """
    Macro-average P / R / F1 over a list of (predicted_set, gold_set) pairs.

    Macro-average: compute per-document metrics, then average.
    This treats each document equally regardless of how many codes it has.
    """
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, gold in pairs:
        m = precision_recall_f1(pred, gold)
        for k in agg:
            agg[k] += m[k]
    n = len(pairs)
    return {k: v / n for k, v in agg.items()}


def load_gold_map(gt_path: Path) -> dict[str, set[str]]:
    """
    Load the CodiEsp ground-truth parquet and return a dict:
        {file_stem: set_of_unique_icd_codes}

    The 'codes' column is a semicolon-delimited string that may contain
    duplicate entries (same code listed twice). We deduplicate.
    """
    df = pl.read_parquet(gt_path)
    return {
        row["file_stem"]: set(row["codes"].split(";"))
        for row in df.iter_rows(named=True)
    }
```

---

## 2. Option A — Full-Document TF-IDF Threshold Pipeline

### How it works

Feed the **entire clinical note** as a single query into `TfidfRetriever.search()`.
Apply a cosine-similarity threshold; keep all codes that exceed it.

```
note text  →  TF-IDF transform  →  cosine similarity against 74k code descriptions
           →  keep codes where score ≥ threshold
```

**Pros:** Zero new infrastructure; one line of code change on existing retriever.
**Cons:** Long notes dilute the query vector — rare codes score poorly. High threshold
         loses recall; low threshold floods predictions with noise.

### Pipeline

```python
# -----------------------------------------------------------------------
# Option A: Full-document TF-IDF retrieval pipeline
# -----------------------------------------------------------------------
# src/pipeline_option_a.py

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever
from pipeline_components import NoteLoaderTransformer, TextCleaner


class FullDocRetrievalPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label 'classifier' backed by TF-IDF retrieval over the KnowledgeBase.

    fit()    — builds the TfidfRetriever index (TF-IDF over code descriptions).
               The *document* TF-IDF is used only at query time, not here.
    predict() — for each note, run search(note, top_k) and threshold by score.

    Parameters
    ----------
    kb : KnowledgeBase
        Pre-loaded knowledge base with 74k ICD-10 entries.
    top_k : int
        Maximum number of candidate codes to retrieve per document.
    threshold : float
        Minimum cosine similarity score to include a code in the prediction.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        top_k: int = 50,
        threshold: float = 0.05,
    ) -> None:
        self.kb = kb
        self.top_k = top_k
        self.threshold = threshold
        self._retriever: TfidfRetriever | None = None

    def fit(self, X: list[str], y=None) -> "FullDocRetrievalPredictor":
        """
        Build the TF-IDF index over code descriptions.

        X : list[str] — cleaned note texts (consumed by the pipeline but not
                         used during index construction; the index is built over
                         code descriptions, not documents).
        """
        # TfidfRetriever.__init__ fits TF-IDF on kb.entries descriptions.
        self._retriever = TfidfRetriever(self.kb)
        # Store all known codes so downstream code can inspect classes_.
        self.classes_ = np.array([e.code for e in self.kb.entries])
        return self

    def predict(self, X: list[str]) -> list[set[str]]:
        """
        Return a predicted code set for each note in X.

        X : list[str] — cleaned note texts
        Returns: list[set[str]] — one set of ICD codes per document
        """
        assert self._retriever is not None, "Call fit() before predict()."
        results: list[set[str]] = []
        for text in X:
            hits = self._retriever.search(text, top_k=self.top_k)
            results.append({h.code for h in hits if h.score >= self.threshold})
        return results


def build_pipeline_a(
    kb: KnowledgeBase,
    text_dir,
    top_k: int = 50,
    threshold: float = 0.05,
) -> Pipeline:
    """
    Build the full end-to-end Option A pipeline.

    Input to pipeline: list[str] file stems
    Output:            list[set[str]] predicted code sets
    """
    return Pipeline(
        steps=[
            # Step 1: read .txt files from disk
            ("loader", NoteLoaderTransformer(text_dir)),
            # Step 2: clean / normalize text
            ("cleaner", TextCleaner(lowercase=True)),
            # Step 3: retrieve + threshold
            ("predictor", FullDocRetrievalPredictor(kb, top_k=top_k, threshold=threshold)),
        ],
        # verbose=True shows timing per step
        verbose=True,
    )


# -----------------------------------------------------------------------
# End-to-end script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from pipeline_components import load_gold_map, macro_average

    # --- Paths -----------------------------------------------------------
    KB_PATH  = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
    GT_PATH  = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
    TEXT_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

    # --- Load shared resources -------------------------------------------
    kb       = KnowledgeBase(KB_PATH)                # 74,719 ICD entries
    gold_map = load_gold_map(GT_PATH)                # {file_stem: set[str]}
    stems    = sorted(gold_map.keys())               # 250 dev file stems

    # --- Build & fit pipeline --------------------------------------------
    # fit() here constructs the TF-IDF index over code descriptions.
    # The file stems are threaded through NoteLoaderTransformer → TextCleaner
    # but the predictor's fit() ignores the texts (retrieval is unsupervised).
    pipeline = build_pipeline_a(kb, TEXT_DIR, top_k=50, threshold=0.05)
    pipeline.fit(stems)  # type: ignore[arg-type]

    # --- Predict ---------------------------------------------------------
    # predict() calls loader → cleaner → predictor.predict() in sequence.
    predictions: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]

    # --- Evaluate --------------------------------------------------------
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    metrics = macro_average(pairs)
    print(f"Option A  |  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # --- Grid-search threshold -------------------------------------------
    # Pipeline.set_params() uses __ notation to reach nested step parameters.
    print("\nThreshold sweep:")
    for t in np.arange(0.02, 0.15, 0.01):
        pipeline.set_params(predictor__threshold=float(t))
        preds: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
        m = macro_average([(p, gold_map[s]) for p, s in zip(preds, stems)])
        print(f"  threshold={t:.2f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
```

---

## 3. Option B — Sentence-Level Retrieval Pipeline

### How it works

Split each note into sentences, retrieve codes per sentence, aggregate by max/mean/sum score.
A code is included in the prediction if its aggregated score ≥ threshold.

```
note text
    │
    ▼  SentenceSplitter     →  ["sentence 1", "sentence 2", ...]
    │
    ▼  per-sentence search  →  code → [score_s1, score_s3, ...]
    │
    ▼  aggregate (max)      →  code → final_score
    │
    ▼  threshold            →  predicted set
```

**Pros:** Each sentence is a focused, short query; better recall for codes that appear
         in isolated clinical mentions.
**Cons:** Short sentences may have zero TF-IDF overlap with code descriptions;
         requires aggregation tuning.

### Pipeline

```python
# -----------------------------------------------------------------------
# Option B: Sentence-level retrieval pipeline
# -----------------------------------------------------------------------
# src/pipeline_option_b.py

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever
from pipeline_components import NoteLoaderTransformer, TextCleaner


class SentenceSplitter(BaseEstimator, TransformerMixin):
    """
    Transform a list of note texts into a list of sentence lists.

    Input  X : list[str]         — one text per document
    Output   : list[list[str]]   — one list-of-sentences per document

    Parameters
    ----------
    min_words : int
        Discard sentence fragments shorter than this many tokens.
    """

    def __init__(self, min_words: int = 4) -> None:
        self.min_words = min_words

    def fit(self, X: list[str], y=None) -> "SentenceSplitter":
        return self

    def transform(self, X: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for text in X:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            result.append(
                [s.strip() for s in sentences if len(s.split()) >= self.min_words]
            )
        return result


class SentenceLevelRetrievalPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label predictor: retrieve per sentence, aggregate scores.

    fit()     — builds TF-IDF index over KB code descriptions.
    predict() — for each document (list of sentences), search per sentence,
                aggregate, threshold.

    Parameters
    ----------
    kb : KnowledgeBase
    top_k_per_sentence : int
        Candidate codes retrieved per sentence.
    min_sentence_score : float
        Per-sentence score threshold; hits below this are discarded before aggregation.
    aggregation : str
        How to combine per-sentence scores for the same code.
        "max"  → best single-sentence score
        "mean" → average across sentences where the code appeared
        "sum"  → total score (biases toward codes mentioned many times)
    final_threshold : float
        Aggregated score must exceed this to be included in the prediction.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        top_k_per_sentence: int = 10,
        min_sentence_score: float = 0.04,
        aggregation: str = "max",
        final_threshold: float = 0.05,
    ) -> None:
        self.kb = kb
        self.top_k_per_sentence = top_k_per_sentence
        self.min_sentence_score = min_sentence_score
        self.aggregation = aggregation
        self.final_threshold = final_threshold
        self._retriever: TfidfRetriever | None = None

    def fit(self, X: list[list[str]], y=None) -> "SentenceLevelRetrievalPredictor":
        self._retriever = TfidfRetriever(self.kb)
        self.classes_ = np.array([e.code for e in self.kb.entries])
        return self

    def _predict_one(self, sentences: list[str]) -> set[str]:
        """Predict codes for a single document's sentence list."""
        assert self._retriever is not None
        # Accumulate per-code scores across all sentences
        code_scores: dict[str, list[float]] = defaultdict(list)
        for sentence in sentences:
            hits = self._retriever.search(sentence, top_k=self.top_k_per_sentence)
            for h in hits:
                if h.score >= self.min_sentence_score:
                    code_scores[h.code].append(h.score)

        # Aggregate
        aggregated: dict[str, float] = {}
        for code, scores in code_scores.items():
            if self.aggregation == "max":
                aggregated[code] = max(scores)
            elif self.aggregation == "mean":
                aggregated[code] = sum(scores) / len(scores)
            else:  # "sum"
                aggregated[code] = sum(scores)

        return {code for code, score in aggregated.items() if score >= self.final_threshold}

    def predict(self, X: list[list[str]]) -> list[set[str]]:
        """X is a list of sentence lists (output of SentenceSplitter)."""
        return [self._predict_one(sentences) for sentences in X]


def build_pipeline_b(
    kb: KnowledgeBase,
    text_dir,
    top_k_per_sentence: int = 10,
    min_sentence_score: float = 0.04,
    aggregation: str = "max",
    final_threshold: float = 0.05,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("loader",    NoteLoaderTransformer(text_dir)),
            ("cleaner",   TextCleaner(lowercase=True)),
            ("splitter",  SentenceSplitter(min_words=4)),
            ("predictor", SentenceLevelRetrievalPredictor(
                kb,
                top_k_per_sentence=top_k_per_sentence,
                min_sentence_score=min_sentence_score,
                aggregation=aggregation,
                final_threshold=final_threshold,
            )),
        ],
        verbose=True,
    )


# -----------------------------------------------------------------------
# End-to-end script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from pipeline_components import load_gold_map, macro_average

    KB_PATH  = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
    GT_PATH  = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
    TEXT_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

    kb       = KnowledgeBase(KB_PATH)
    gold_map = load_gold_map(GT_PATH)
    stems    = sorted(gold_map.keys())

    pipeline = build_pipeline_b(kb, TEXT_DIR)
    pipeline.fit(stems)  # type: ignore[arg-type]

    predictions: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    metrics = macro_average(pairs)
    print(f"Option B  |  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # Tune aggregation strategy via set_params()
    for agg in ("max", "mean", "sum"):
        pipeline.set_params(predictor__aggregation=agg)
        preds: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
        m = macro_average([(p, gold_map[s]) for p, s in zip(preds, stems)])
        print(f"  aggregation={agg:<4}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
```

---

## 4. Option C — Supervised Multi-Label Classifier Pipeline

### How it works

Train a `OneVsRestClassifier` using labelled training documents.
TF-IDF vectorizes the notes; each code becomes a binary target.
The `MultiLabelBinarizer` maps label sets to/from binary matrix rows.

```
Training:
  file_stems, label_sets → MultiLabelBinarizer.fit_transform → Y_bin (n_docs × n_labels)
  file_stems             → Pipeline.fit(X, Y_bin)
                              NoteLoader → TextCleaner → TfidfVectorizer → OvR-LR

Inference:
  file_stems → Pipeline.predict → Y_pred (binary matrix)
             → mlb.inverse_transform → list[tuple[str,...]]
             → list[set[str]]
```

**Pros:** Learns task-specific patterns; high precision when training set is large.
**Cons:** Requires train split; label space grows with training set (codes not seen in
         training cannot be predicted); class imbalance is extreme (most codes appear rarely).

> **Note:** The CodiEsp full train split (~1000 documents) is needed for meaningful
> training. Using only dev for both train and eval will overfit badly.

### Pipeline

```python
# -----------------------------------------------------------------------
# Option C: Supervised multi-label classification pipeline
# -----------------------------------------------------------------------
# src/pipeline_option_c.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from pipeline_components import NoteLoaderTransformer, TextCleaner, load_gold_map, macro_average


def build_pipeline_c(
    text_dir,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Build the supervised classification pipeline.

    The pipeline accepts raw file stems as X.
    It does NOT include the MultiLabelBinarizer — that lives outside because
    it transforms the *label* space (y), not the feature space (X).

    Pipeline steps:
        loader   : file stems → raw texts
        cleaner  : raw texts  → normalized texts
        tfidf    : texts      → sparse TF-IDF matrix (n_docs × max_features)
        clf      : matrix     → binary label predictions (n_docs × n_labels)
    """
    return Pipeline(
        steps=[
            # Step 1: load raw text from disk given file stems
            ("loader",  NoteLoaderTransformer(text_dir)),
            # Step 2: normalize (lowercase, collapse whitespace)
            ("cleaner", TextCleaner(lowercase=True)),
            # Step 3: TF-IDF vectorization
            #   - ngram_range (1,2): unigrams + bigrams capture "heart failure",
            #     "type 2", etc.
            #   - max_features=50_000: cap vocabulary size to keep matrix manageable
            #   - sublinear_tf=True: apply log(tf)+1, dampens high-frequency terms
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words="english",
                max_features=max_features,
                sublinear_tf=True,          # log normalization of term frequency
                min_df=2,                   # ignore terms that appear in < 2 docs
            )),
            # Step 4: One-vs-Rest logistic regression
            #   OneVsRestClassifier trains one binary classifier per label.
            #   n_jobs=-1 parallelizes across available CPU cores.
            ("clf", OneVsRestClassifier(
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,                    # inverse regularization strength
                    solver="lbfgs",
                    class_weight="balanced", # compensates for extreme class imbalance
                ),
                n_jobs=-1,
            )),
        ],
        verbose=True,
    )


def prepare_labels(
    gt_df: pl.DataFrame,
    stems: list[str],
    mlb: MultiLabelBinarizer,
    fit: bool = False,
) -> np.ndarray:
    """
    Convert the ground-truth parquet into a binary label matrix aligned to stems.

    Parameters
    ----------
    gt_df   : polars DataFrame with columns {file_stem, codes}
    stems   : ordered list of file stems (determines row order)
    mlb     : a MultiLabelBinarizer instance
    fit     : if True, call mlb.fit_transform(); if False, call mlb.transform()

    Returns
    -------
    Y : np.ndarray of shape (len(stems), n_labels), dtype uint8
    """
    code_map: dict[str, list[str]] = {
        row["file_stem"]: list(set(row["codes"].split(";")))
        for row in gt_df.iter_rows(named=True)
    }
    label_sets: list[list[str]] = [code_map.get(s, []) for s in stems]
    if fit:
        return mlb.fit_transform(label_sets)
    return mlb.transform(label_sets)


# -----------------------------------------------------------------------
# End-to-end script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    TRAIN_GT_PATH = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/train_ground_truth.parquet")
    DEV_GT_PATH   = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
    TRAIN_DIR     = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/train/text_files_en")
    DEV_DIR       = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

    # --- Load ground-truth -----------------------------------------------
    train_gt = pl.read_parquet(TRAIN_GT_PATH)
    dev_gt   = pl.read_parquet(DEV_GT_PATH)

    train_stems = [row["file_stem"] for row in train_gt.iter_rows(named=True)]
    dev_stems   = [row["file_stem"] for row in dev_gt.iter_rows(named=True)]

    # --- Binarize labels -------------------------------------------------
    # MultiLabelBinarizer lives *outside* the pipeline because it transforms y
    # (labels), not X (features). It must be fit on training labels only.
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y_train = prepare_labels(train_gt, train_stems, mlb, fit=True)
    Y_dev   = prepare_labels(dev_gt,   dev_stems,   mlb, fit=False)

    print(f"Label space: {len(mlb.classes_)} unique codes across training set")
    print(f"Y_train shape: {Y_train.shape}")  # (n_train, n_labels)

    # --- Build and fit the pipeline --------------------------------------
    # pipeline.fit(X, y):
    #   X = list of file stems  → loader → cleaner → tfidf.fit_transform
    #   y = Y_train (binary matrix)  → clf.fit
    pipeline = build_pipeline_c(TRAIN_DIR)
    pipeline.fit(train_stems, Y_train)

    # --- Predict on dev --------------------------------------------------
    # pipeline.predict(X):
    #   X = list of file stems  → loader → cleaner → tfidf.transform → clf.predict
    # Returns: binary matrix (n_dev, n_labels)
    Y_pred = pipeline.predict(dev_stems)

    # Convert binary rows back to code sets using the fitted mlb
    # row.nonzero()[0] gives indices of positive (=1) columns
    y_pred_sets: list[set[str]] = [
        {mlb.classes_[i] for i in row.nonzero()[0]}
        for row in Y_pred
    ]

    # Gold sets aligned to dev_stems
    gold_map = load_gold_map(DEV_GT_PATH)
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(y_pred_sets, dev_stems)]
    metrics = macro_average(pairs)
    print(f"Option C  |  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # --- Tune regularization via set_params() ----------------------------
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, f1_score

    # Rebuild pipeline for grid search (dev used as proxy; use train split for real tuning)
    pipeline_gs = build_pipeline_c(DEV_DIR)

    # f1_samples averages F1 over samples, treating multi-label rows as units
    scorer = make_scorer(f1_score, average="samples", zero_division=0)

    param_grid = {
        "clf__estimator__C": [0.1, 0.5, 1.0, 5.0],
        "tfidf__max_features": [20_000, 50_000],
    }

    gs = GridSearchCV(
        pipeline_gs,
        param_grid,
        scoring=scorer,
        cv=3,       # 3-fold CV within the passed dataset
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(dev_stems, Y_dev)  # demo only — normally fit on train split
    print(f"Best params: {gs.best_params_}  Best F1: {gs.best_score_:.3f}")
```

---

## 5. Option D — Semantic Embedding Pipeline

### How it works

Embed code descriptions with a medical language model once (offline).
At inference, embed each note and retrieve codes by cosine similarity in embedding space.

```
Fit (one-time, slow):
    74,719 code descriptions → SentenceTransformer → embeddings matrix (74k × 768)
                             → FAISS index

Predict (fast):
    note text → SentenceTransformer → query vector (1 × 768)
              → FAISS top-k search → candidate codes + cosine scores
              → threshold → predicted set
```

**Pros:** Captures semantic equivalence ("myocardial infarction" ≈ "heart attack");
          no vocabulary mismatch.
**Cons:** Requires `sentence-transformers` and `faiss-cpu`; ~10–30s to encode all 74k codes.

### Pipeline

```python
# -----------------------------------------------------------------------
# Option D: Semantic embedding pipeline
# -----------------------------------------------------------------------
# src/pipeline_option_d.py
#
# Extra dependencies (add to pyproject.toml):
#   sentence-transformers>=3.0
#   faiss-cpu>=1.8          (or faiss-gpu for GPU machines)

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from knowledge_base import KnowledgeBase
from pipeline_components import NoteLoaderTransformer, TextCleaner

# Lazy imports — only needed when this module is actually used
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False


# Medical sentence-transformer model.
# Alternatives:
#   "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" — entity linking, good for code descriptions
#   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" — general biomedical
DEFAULT_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"


class SemanticRetrievalPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label predictor using sentence-transformer embeddings + FAISS.

    fit()    — encodes all KB code descriptions into a FAISS flat index.
               This is the slow step (~30s on CPU for 74k codes).
    predict() — encodes each note, searches FAISS, thresholds by cosine score.

    Parameters
    ----------
    kb : KnowledgeBase
    model_name : str
        HuggingFace model ID for SentenceTransformer.
    top_k : int
        Number of nearest-neighbour codes to retrieve.
    threshold : float
        Minimum cosine similarity (range 0–1) to include in prediction.
    batch_size : int
        Batch size for encoding; increase if you have more GPU memory.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 30,
        threshold: float = 0.55,
        batch_size: int = 256,
    ) -> None:
        if not _SEMANTIC_AVAILABLE:
            raise ImportError("Install sentence-transformers and faiss-cpu to use Option D.")
        self.kb = kb
        self.model_name = model_name
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._index = None
        self._codes: list[str] = []

    def fit(self, X: list[str], y=None) -> "SemanticRetrievalPredictor":
        """
        Encode all KB descriptions and build a FAISS inner-product index.

        Inner product == cosine similarity when vectors are L2-normalised.
        """
        self._model = SentenceTransformer(self.model_name)
        self._codes = [e.code for e in self.kb.entries]
        descriptions = [e.description for e in self.kb.entries]

        print(f"Encoding {len(descriptions):,} code descriptions ...")
        embeddings: np.ndarray = self._model.encode(
            descriptions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2 normalise → inner product = cosine
            convert_to_numpy=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        # IndexFlatIP: exact inner-product search (no approximation).
        # For >500k codes, replace with IndexIVFFlat for speed.
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"FAISS index built: {self._index.ntotal:,} vectors, dim={dim}")
        self.classes_ = np.array(self._codes)
        return self

    def predict(self, X: list[str]) -> list[set[str]]:
        assert self._model is not None and self._index is not None, "Call fit() first."
        # Encode all query notes in a single batch for efficiency
        query_vecs: np.ndarray = self._model.encode(
            X,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Batch FAISS search: shape (n_docs, top_k)
        scores, indices = self._index.search(query_vecs, self.top_k)

        results: list[set[str]] = []
        for doc_scores, doc_indices in zip(scores, indices):
            predicted = {
                self._codes[int(idx)]
                for score, idx in zip(doc_scores, doc_indices)
                if float(score) >= self.threshold and idx >= 0
            }
            results.append(predicted)
        return results


def build_pipeline_d(
    kb: KnowledgeBase,
    text_dir,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 30,
    threshold: float = 0.55,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("loader",    NoteLoaderTransformer(text_dir)),
            # Do NOT lowercase — transformer models are case-sensitive
            ("cleaner",   TextCleaner(lowercase=False)),
            ("predictor", SemanticRetrievalPredictor(
                kb, model_name=model_name, top_k=top_k, threshold=threshold,
            )),
        ],
        verbose=True,
    )


# -----------------------------------------------------------------------
# End-to-end script
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    from pipeline_components import load_gold_map, macro_average

    KB_PATH  = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
    GT_PATH  = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
    TEXT_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

    kb       = KnowledgeBase(KB_PATH)
    gold_map = load_gold_map(GT_PATH)
    stems    = sorted(gold_map.keys())

    # fit() encodes 74k descriptions — runs once, slow on CPU (~30s)
    pipeline = build_pipeline_d(kb, TEXT_DIR)
    pipeline.fit(stems)  # type: ignore[arg-type]

    predictions: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    metrics = macro_average(pairs)
    print(f"Option D  |  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
```

---

## 6. Option E — Chunk + Retrieve + Aggregate Pipeline (Recommended)

### How it works

Split the note into overlapping sentence windows ("chunks").
Retrieve codes per chunk using `TfidfRetriever`.
Aggregate across all chunks; apply a final threshold.
This is the best balance of quality vs. complexity using existing infrastructure.

```
note text
    │
    ▼  SlidingWindowChunker     →  ["chunk_1", "chunk_2", ..., full_text]
    │                               each chunk = window_size consecutive sentences
    │                               stride controls overlap
    ▼  per-chunk TF-IDF search  →  code → [score_c1, score_c2, ...]
    │
    ▼  aggregate (max/mean/sum) →  code → final_score
    │
    ▼  threshold + min_votes    →  predicted set
```

### Pipeline

```python
# -----------------------------------------------------------------------
# Option E: Sliding-window chunk retrieval pipeline (recommended)
# -----------------------------------------------------------------------
# src/pipeline_option_e.py

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever
from pipeline_components import NoteLoaderTransformer, TextCleaner


class SlidingWindowChunker(BaseEstimator, TransformerMixin):
    """
    Transform a list of texts into a list of chunk-lists using a sliding window.

    Input  X : list[str]          — one text per document
    Output   : list[list[str]]    — one list-of-chunks per document

    Each chunk = window_size consecutive sentences joined as a single string.
    The full text is always appended as an extra chunk so global context is preserved.

    Parameters
    ----------
    window : int
        Number of sentences per chunk.
    stride : int
        Step size between chunk start positions.
        stride < window  →  overlapping chunks (better context).
        stride == window →  non-overlapping chunks.
    min_words : int
        Minimum token count for a sentence to be included.
    include_full_text : bool
        If True, always append the entire document as one final chunk.
        This helps global-context codes like "patient history of X".
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
            # Sliding window over sentences
            for i in range(0, max(1, len(sentences) - self.window + 1), self.stride):
                chunk = " ".join(sentences[i : i + self.window])
                if chunk:
                    chunks.append(chunk)
            if self.include_full_text:
                chunks.append(text)  # always include full document as final chunk
            result.append(chunks)
        return result


class ChunkedRetrievalPredictor(BaseEstimator, ClassifierMixin):
    """
    Multi-label predictor: retrieve per chunk, aggregate, threshold.

    fit()    — builds TF-IDF index over KB code descriptions.
    predict() — for each document's chunks, retrieves candidates per chunk,
                aggregates scores, applies vote and score thresholds.

    Parameters
    ----------
    kb : KnowledgeBase
    top_k_per_chunk : int
        Candidate codes retrieved per chunk.
    min_chunk_score : float
        Per-chunk score minimum. Hits below this are ignored.
    min_votes : int
        A code must appear in at least this many chunks to survive aggregation.
        min_votes=1 → any occurrence counts; higher values require corroboration.
    aggregation : str
        "max"  — use the best score across all chunks (good for sparse signals)
        "mean" — average score (penalizes codes seen in very few chunks)
        "sum"  — total score (rewards codes mentioned frequently)
    final_threshold : float
        Aggregated score must exceed this to be included in the final prediction set.
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
        """
        Build TF-IDF index over KB code descriptions.
        X (chunk lists) is passed through but not used during index construction.
        """
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

        # Apply vote filter
        filtered = {
            code: scores
            for code, scores in code_scores.items()
            if len(scores) >= self.min_votes
        }

        # Aggregate scores per code
        aggregated: dict[str, float] = {}
        for code, scores in filtered.items():
            if self.aggregation == "max":
                aggregated[code] = max(scores)
            elif self.aggregation == "mean":
                aggregated[code] = sum(scores) / len(scores)
            else:  # "sum"
                aggregated[code] = sum(scores)

        # Final threshold
        return {code for code, score in aggregated.items() if score >= self.final_threshold}

    def predict(self, X: list[list[str]]) -> list[set[str]]:
        """X is the output of SlidingWindowChunker.transform()."""
        return [self._predict_one(chunks) for chunks in X]


def build_pipeline_e(
    kb: KnowledgeBase,
    text_dir,
    window: int = 3,
    stride: int = 2,
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
    final_threshold: float = 0.04,
) -> Pipeline:
    """
    Build the recommended end-to-end Option E pipeline.

    Input : list[str] file stems
    Output: list[set[str]] predicted code sets
    """
    return Pipeline(
        steps=[
            # Step 1: read clinical note text from disk
            ("loader",    NoteLoaderTransformer(text_dir)),
            # Step 2: normalize text (lowercase, collapse whitespace)
            ("cleaner",   TextCleaner(lowercase=True)),
            # Step 3: split into overlapping sentence chunks
            ("chunker",   SlidingWindowChunker(
                window=window, stride=stride, include_full_text=True
            )),
            # Step 4: retrieve codes per chunk, aggregate, threshold
            ("predictor", ChunkedRetrievalPredictor(
                kb,
                top_k_per_chunk=top_k_per_chunk,
                min_chunk_score=min_chunk_score,
                min_votes=min_votes,
                aggregation=aggregation,
                final_threshold=final_threshold,
            )),
        ],
        verbose=True,
    )


# -----------------------------------------------------------------------
# End-to-end evaluation script with parameter sweep
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import itertools
    from pipeline_components import load_gold_map, macro_average

    KB_PATH  = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
    GT_PATH  = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
    TEXT_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

    kb       = KnowledgeBase(KB_PATH)
    gold_map = load_gold_map(GT_PATH)
    stems    = sorted(gold_map.keys())

    # Build and fit (constructs TF-IDF index over KB descriptions)
    pipeline = build_pipeline_e(kb, TEXT_DIR)
    pipeline.fit(stems)  # type: ignore[arg-type]

    # Baseline prediction
    predictions: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    metrics = macro_average(pairs)
    print(f"Option E (default)  |  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    # --- Parameter sweep using set_params() ------------------------------
    # Pipeline.set_params uses double-underscore notation to reach nested params:
    #   "chunker__window" → pipeline.steps["chunker"].window
    #   "predictor__aggregation" → pipeline.steps["predictor"].aggregation
    print("\nParameter sweep:")
    best_f1, best_cfg = 0.0, {}
    for window, top_k, threshold in itertools.product(
        [2, 3, 5],          # chunk window sizes
        [5, 10, 20],        # top-k codes per chunk
        [0.03, 0.05, 0.08], # final score thresholds
    ):
        # Update nested parameters without rebuilding the pipeline
        pipeline.set_params(
            chunker__window=window,
            predictor__top_k_per_chunk=top_k,
            predictor__final_threshold=threshold,
        )
        # Re-fit is needed only if the retriever changes; chunker changes don't require it.
        # Since we only change chunker and threshold params, no re-fit needed here.
        preds: list[set[str]] = pipeline.predict(stems)  # type: ignore[assignment]
        m = macro_average([(p, gold_map[s]) for p, s in zip(preds, stems)])
        cfg = {"window": window, "top_k": top_k, "threshold": threshold}
        print(f"  {cfg}  →  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        if m["f1"] > best_f1:
            best_f1, best_cfg = m["f1"], {**cfg, **m}

    print(f"\nBest config: {best_cfg}")
```

---

## 7. Comparing All Pipelines Head-to-Head

```python
# scripts/compare_all_pipelines.py
"""
Run all five options on the dev set and print a comparison table.

Usage:
    uv run python scripts/compare_all_pipelines.py
"""

from pathlib import Path

import polars as pl
from knowledge_base import KnowledgeBase
from pipeline_components import load_gold_map, macro_average
from pipeline_option_a import build_pipeline_a
from pipeline_option_b import build_pipeline_b
from pipeline_option_e import build_pipeline_e

KB_PATH  = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
GT_PATH  = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
TEXT_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

kb       = KnowledgeBase(KB_PATH)
gold_map = load_gold_map(GT_PATH)
stems    = sorted(gold_map.keys())

PIPELINES = {
    "A: full-doc threshold":    build_pipeline_a(kb, TEXT_DIR, top_k=50, threshold=0.05),
    "B: sentence-level (max)":  build_pipeline_b(kb, TEXT_DIR, aggregation="max"),
    "E: chunked (window=3)":    build_pipeline_e(kb, TEXT_DIR, window=3, stride=2),
    "E: chunked (window=5)":    build_pipeline_e(kb, TEXT_DIR, window=5, stride=3),
}

print(f"\n{'Option':<30} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print("-" * 62)
for name, pipeline in PIPELINES.items():
    pipeline.fit(stems)  # type: ignore[arg-type]
    predictions: list = pipeline.predict(stems)
    pairs = [(pred, gold_map[stem]) for pred, stem in zip(predictions, stems)]
    m = macro_average(pairs)
    print(f"{name:<30} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}")
```

---

## 8. Evaluation Metrics Reference

```python
# Full metric suite using sklearn.metrics for multi-label evaluation

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    hamming_loss,        # fraction of labels that are incorrectly predicted
    f1_score,            # with average="samples", "macro", or "micro"
    precision_score,
    recall_score,
    jaccard_score,       # intersection-over-union per sample
)

def sklearn_multilabel_metrics(
    y_pred_sets: list[set[str]],
    y_gold_sets: list[set[str]],
) -> dict[str, float]:
    """
    Compute a full suite of sklearn multi-label metrics.

    Requires binarizing label sets with MultiLabelBinarizer first.
    Uses 'samples' averaging — computes per-document, then averages.
    """
    # Fit binarizer on the union of all codes seen in gold + predictions
    all_codes = sorted({c for s in y_gold_sets + y_pred_sets for c in s})
    mlb = MultiLabelBinarizer(classes=all_codes)
    mlb.fit([all_codes])

    Y_true = mlb.transform(y_gold_sets)
    Y_pred = mlb.transform(y_pred_sets)

    return {
        "hamming_loss":        hamming_loss(Y_true, Y_pred),
        "f1_samples":          f1_score(Y_true, Y_pred, average="samples", zero_division=0),
        "f1_micro":            f1_score(Y_true, Y_pred, average="micro",   zero_division=0),
        "f1_macro":            f1_score(Y_true, Y_pred, average="macro",   zero_division=0),
        "precision_samples":   precision_score(Y_true, Y_pred, average="samples", zero_division=0),
        "recall_samples":      recall_score(Y_true, Y_pred, average="samples", zero_division=0),
        "jaccard_samples":     jaccard_score(Y_true, Y_pred, average="samples", zero_division=0),
    }
```

---

## 9. Comparison Summary

| Option | Pipeline steps | New deps | Requires train labels | Recall potential | Recommended |
|--------|---------------|----------|-----------------------|-----------------|-------------|
| A — Full-doc TF-IDF | loader → cleaner → predictor | None | No | Low–Med | Baseline only |
| B — Sentence-level | loader → cleaner → splitter → predictor | None | No | Med | Good fallback |
| C — OvR Classifier | loader → cleaner → tfidf → OvR-LR | None | **Yes** | Med–High | With train split |
| D — Semantic | loader → cleaner → SemanticPredictor | `sentence-transformers`, `faiss-cpu` | No | High | GPU available |
| **E — Chunked** | loader → cleaner → chunker → predictor | None | No | Med–High | **Start here** |

---

## 10. Suggested File Layout

```
src/
  knowledge_base.py          # existing
  retriever.py               # existing single-label TF-IDF retriever
  pipeline_components.py     # new: NoteLoaderTransformer, TextCleaner,
                             #      precision_recall_f1, macro_average, load_gold_map
  pipeline_option_a.py       # new: FullDocRetrievalPredictor + build_pipeline_a
  pipeline_option_b.py       # new: SentenceSplitter + SentenceLevelRetrievalPredictor
  pipeline_option_c.py       # new: supervised OvR pipeline (needs train split)
  pipeline_option_d.py       # new: SemanticRetrievalPredictor (needs extra deps)
  pipeline_option_e.py       # new: SlidingWindowChunker + ChunkedRetrievalPredictor ← start here

scripts/
  compare_all_pipelines.py   # new: head-to-head evaluation table
```
