# Multi-Label ICD-10 Retrieval from Clinical Notes

**Date:** 2026-02-26
**Context:** The CodiEsp dev split has 250 clinical notes (English). Each note has on average **10.7 unique ICD-10-CM codes** (range 1–33). The existing `TfidfRetriever` in `src/retriever.py` is designed for single-query → top-k lookup. We need a new function (or class) that accepts a full clinical note and returns *all* plausible codes.

---

## 1. Problem Framing

| Aspect | Single-label (current) | Multi-label (new goal) |
|--------|----------------------|----------------------|
| Input | Short condition phrase | Full clinical note |
| Output | Top-k codes, rank-ordered | Set of predicted codes |
| Ground truth | One correct code | Set of 1–33 correct codes |
| Evaluation | Precision@1, MRR | Precision@k, Recall@k, F1, Jaccard |

### Ground truth format

```python
# data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet
# Schema: {'file_stem': String, 'codes': String}
# 'codes' is a semicolon-separated string of ICD codes (may contain duplicates)
# Example: 'Q6211;N2889;N390;R319;N23;N280;Q6211;D1809;N135;K269;N289;N200;K5900'

import polars as pl

df = pl.read_parquet("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
# Normalise: deduplicate and sort
df = df.with_columns(
    pl.col("codes")
    .str.split(";")
    .list.unique()
    .list.sort()
    .alias("code_set")
)
```

### Evaluation helpers

```python
def precision_recall_f1(predicted: set[str], gold: set[str]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gold)    if gold      else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_batch(results: list[tuple[set[str], set[str]]]) -> dict[str, float]:
    """results: list of (predicted_set, gold_set) tuples."""
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, gold in results:
        m = precision_recall_f1(pred, gold)
        for k in agg:
            agg[k] += m[k]
    n = len(results)
    return {k: v / n for k, v in agg.items()}
```

---

## 2. Option A — Threshold-Based TF-IDF on the Full Document

**Idea:** Feed the entire clinical note as a single query into the existing `TfidfRetriever.search()` using a large `top_k`, then keep all results above a cosine-similarity threshold.

**Pros:** Zero new infrastructure; directly extends existing code.
**Cons:** Long documents dilute the query vector; single threshold is crude; tends to over-retrieve non-specific codes.

### Code

```python
# src/multilabel_retriever.py

from dataclasses import dataclass, field
from pathlib import Path

from retriever import TfidfRetriever
from knowledge_base import KnowledgeBase


@dataclass
class MultiLabelResult:
    file_stem: str
    predicted_codes: list[str]
    scored_codes: list[tuple[str, float]]  # (code, score), descending


def predict_codes_threshold(
    retriever: TfidfRetriever,
    note_text: str,
    top_k: int = 50,
    threshold: float = 0.05,
) -> list[str]:
    """Run TF-IDF retrieval on the whole document, keep codes above threshold."""
    hits = retriever.search(note_text, top_k=top_k)
    return [h.code for h in hits if h.score >= threshold]


def run_batch(
    retriever: TfidfRetriever,
    text_dir: Path,
    top_k: int = 50,
    threshold: float = 0.05,
) -> list[MultiLabelResult]:
    results = []
    for txt_file in sorted(text_dir.glob("*.txt")):
        note = txt_file.read_text(encoding="utf-8", errors="replace")
        hits = retriever.search(note, top_k=top_k)
        above = [(h.code, h.score) for h in hits if h.score >= threshold]
        results.append(
            MultiLabelResult(
                file_stem=txt_file.stem,
                predicted_codes=[c for c, _ in above],
                scored_codes=above,
            )
        )
    return results
```

### Tuning threshold

```python
import polars as pl
import numpy as np

# Grid-search threshold on dev set
gt = pl.read_parquet("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
gold_map = {
    row["file_stem"]: set(row["codes"].split(";"))
    for row in gt.iter_rows(named=True)
}

for threshold in np.arange(0.01, 0.20, 0.01):
    batch = run_batch(retriever, text_dir, top_k=100, threshold=float(threshold))
    pairs = [
        (set(r.predicted_codes), gold_map.get(r.file_stem, set()))
        for r in batch
    ]
    metrics = evaluate_batch(pairs)
    print(f"threshold={threshold:.2f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
```

---

## 3. Option B — Sentence-Level Retrieval with Aggregation

**Idea:** Split the clinical note into sentences. For each sentence, retrieve the top-k codes. Aggregate all retrieved codes across sentences, keeping those that appear frequently enough or score above a threshold. This avoids the dilution problem of a long-document query.

**Pros:** Each sentence is a focused query; better recall for rare codes mentioned only once; mirrors the clinical coding workflow.
**Cons:** More API calls; sentence splitter quality matters; requires aggregation logic.

### Code

```python
import re
from collections import defaultdict

def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '.', '!', '?'."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Drop very short fragments
    return [s.strip() for s in sentences if len(s.split()) >= 4]


def predict_codes_sentence_level(
    retriever: TfidfRetriever,
    note_text: str,
    top_k_per_sentence: int = 5,
    min_sentence_score: float = 0.05,
    min_vote_count: int = 1,       # code must appear in >= N sentences
    aggregation: str = "max",      # "max" | "mean" | "sum"
) -> list[tuple[str, float]]:
    """
    Returns (code, aggregated_score) pairs, sorted by score descending.
    """
    sentences = split_sentences(note_text)
    code_scores: dict[str, list[float]] = defaultdict(list)

    for sentence in sentences:
        hits = retriever.search(sentence, top_k=top_k_per_sentence)
        for h in hits:
            if h.score >= min_sentence_score:
                code_scores[h.code].append(h.score)

    # Filter by vote count
    code_scores = {
        code: scores
        for code, scores in code_scores.items()
        if len(scores) >= min_vote_count
    }

    # Aggregate
    if aggregation == "max":
        agg = {code: max(scores) for code, scores in code_scores.items()}
    elif aggregation == "mean":
        agg = {code: sum(scores) / len(scores) for code, scores in code_scores.items()}
    else:  # sum
        agg = {code: sum(scores) for code, scores in code_scores.items()}

    return sorted(agg.items(), key=lambda x: x[1], reverse=True)
```

### Usage

```python
text_dir = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

results = []
for txt_file in sorted(text_dir.glob("*.txt")):
    note = txt_file.read_text(encoding="utf-8", errors="replace")
    scored = predict_codes_sentence_level(
        retriever, note,
        top_k_per_sentence=10,
        min_sentence_score=0.04,
        min_vote_count=1,
        aggregation="max",
    )
    predicted = {code for code, _ in scored}
    gold = gold_map.get(txt_file.stem, set())
    results.append((predicted, gold))

print(evaluate_batch(results))
```

---

## 4. Option C — Multi-Label sklearn Classifier

**Idea:** Use the ground truth labels to train a proper multi-label classifier. Represent each document as a TF-IDF vector, use `MultiLabelBinarizer` to encode the label set, then train a `OneVsRestClassifier` (or `LinearSVC` per label).

**Pros:** Learns from data; can achieve high precision; fast inference.
**Cons:** Requires training data; 74k+ possible codes makes label space enormous (use only codes that appear in training set); dev-set-only evaluation will be optimistic.

> **Note:** With only 250 dev documents we'd need the full CodiEsp train split for meaningful training. This option is shown for completeness — it becomes viable once training data is available.

### Code

```python
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

# Load train split (replace path as needed)
train_gt = pl.read_parquet("data/test-datasets/codiesp/data-pipeline/processed/gold/train_ground_truth.parquet")

def load_notes(gt_df: pl.DataFrame, text_dir: Path) -> tuple[list[str], list[list[str]]]:
    texts, labels = [], []
    for row in gt_df.iter_rows(named=True):
        p = text_dir / f"{row['file_stem']}.txt"
        if p.exists():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
            labels.append(list(set(row["codes"].split(";"))))
    return texts, labels

train_dir = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/train/text_files_en")
X_train_raw, y_train = load_notes(train_gt, train_dir)

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(y_train)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50_000)),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0), n_jobs=-1)),
])
pipeline.fit(X_train_raw, Y_train)

# Predict on dev
dev_dir = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")
dev_gt  = pl.read_parquet("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
X_dev_raw, y_dev = load_notes(dev_gt, dev_dir)

Y_pred = pipeline.predict(X_dev_raw)
y_pred_sets = [set(mlb.classes_[i] for i in row.nonzero()[0]) for row in Y_pred]
y_gold_sets = [set(codes) for codes in y_dev]

pairs = list(zip(y_pred_sets, y_gold_sets))
print(evaluate_batch(pairs))
```

---

## 5. Option D — Semantic Embeddings with Threshold

**Idea:** Embed both the clinical note (or its sentences) and all ICD-10 code descriptions using a medical language model (e.g., `sentence-transformers` with `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`). Return codes whose embedding similarity to the note exceeds a threshold.

**Pros:** Captures semantic equivalence ("myocardial infarction" ↔ "heart attack"); language-model quality.
**Cons:** Requires GPU or slow CPU inference; FAISS index overhead; embedding 74k codes takes time upfront.

### Code

```python
# pip install sentence-transformers faiss-cpu  (or faiss-gpu)
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

class SemanticMultiLabelRetriever:
    def __init__(self, kb: KnowledgeBase, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)
        self._codes = [e.code for e in kb.entries]
        self._descriptions = [e.description for e in kb.entries]

        print("Encoding knowledge base...")
        embeddings = self._model.encode(
            self._descriptions, batch_size=256, show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._embeddings = np.array(embeddings, dtype="float32")

        # Build FAISS flat index (inner product = cosine when normalised)
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

    def predict_codes(
        self,
        note_text: str,
        top_k: int = 30,
        threshold: float = 0.6,
    ) -> list[tuple[str, float]]:
        query_vec = self._model.encode(
            [note_text], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self._index.search(query_vec, top_k)
        results = [
            (self._codes[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if float(score) >= threshold
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)
```

---

## 6. Option E — Chunk + Retrieve + Rerank (Recommended Pipeline)

**Idea:** Combine the best of Options A/B with a lightweight reranker. Split the document into overlapping chunks of ~3–5 sentences. Retrieve top-k codes per chunk using the existing `TfidfRetriever`. Collect all candidates across chunks, then rescore using a cross-encoder or simple voting, and apply a threshold.

This approach is practical, uses existing infrastructure, and is easy to iterate on.

```
Note
 ├── chunk_1 → TF-IDF top-k → candidates
 ├── chunk_2 → TF-IDF top-k → candidates
 │   ...
 └── chunk_n → TF-IDF top-k → candidates
                       ↓
              aggregate + deduplicate
                       ↓
              score threshold / vote filter
                       ↓
              predicted code set
```

### Code

```python
from collections import defaultdict
import re
from pathlib import Path
from retriever import TfidfRetriever


def chunk_text(text: str, window: int = 3, stride: int = 2) -> list[str]:
    """Sliding-window chunker over sentences."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 3]
    chunks = []
    for i in range(0, max(1, len(sentences) - window + 1), stride):
        chunk = " ".join(sentences[i : i + window])
        if chunk:
            chunks.append(chunk)
    # always include the full text as an extra "chunk" for context
    chunks.append(text)
    return chunks


def predict_codes_chunked(
    retriever: TfidfRetriever,
    note_text: str,
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
) -> list[tuple[str, float]]:
    """
    Chunk the note, retrieve per chunk, aggregate.

    Returns list of (code, score) sorted by score descending.
    """
    chunks = chunk_text(note_text)
    code_scores: dict[str, list[float]] = defaultdict(list)

    for chunk in chunks:
        hits = retriever.search(chunk, top_k=top_k_per_chunk)
        for h in hits:
            if h.score >= min_chunk_score:
                code_scores[h.code].append(h.score)

    # Filter by minimum vote count
    filtered = {
        code: scores
        for code, scores in code_scores.items()
        if len(scores) >= min_votes
    }

    # Aggregate scores
    if aggregation == "max":
        agg = {code: max(scores) for code, scores in filtered.items()}
    elif aggregation == "mean":
        agg = {code: sum(scores) / len(scores) for code, scores in filtered.items()}
    else:
        agg = {code: sum(scores) for code, scores in filtered.items()}

    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


# --- Batch runner ---

def run_multilabel_batch(
    retriever: TfidfRetriever,
    text_dir: Path,
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
    final_threshold: float = 0.0,  # set > 0 to cut low-scoring codes
) -> dict[str, set[str]]:
    """
    Returns {file_stem: predicted_code_set} for all .txt files in text_dir.
    """
    predictions: dict[str, set[str]] = {}
    for txt_file in sorted(text_dir.glob("*.txt")):
        note = txt_file.read_text(encoding="utf-8", errors="replace")
        scored = predict_codes_chunked(
            retriever, note,
            top_k_per_chunk=top_k_per_chunk,
            min_chunk_score=min_chunk_score,
            min_votes=min_votes,
            aggregation=aggregation,
        )
        predictions[txt_file.stem] = {
            code for code, score in scored if score >= final_threshold
        }
    return predictions
```

### End-to-end evaluation script

```python
# scripts/eval_multilabel.py
"""Evaluate multi-label ICD-10 retrieval on the CodiEsp dev set."""

from pathlib import Path
import polars as pl
from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever
from multilabel_retriever import run_multilabel_batch, evaluate_batch

KB_PATH   = Path("data/processed/icd10cm-codes-enriched-April-1-2026.csv")
GT_PATH   = Path("data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet")
TEXT_DIR  = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev/text_files_en")

kb       = KnowledgeBase(KB_PATH)
retriever = TfidfRetriever(kb)

gt = pl.read_parquet(GT_PATH)
gold_map: dict[str, set[str]] = {
    row["file_stem"]: set(row["codes"].split(";"))
    for row in gt.iter_rows(named=True)
}

predictions = run_multilabel_batch(
    retriever, TEXT_DIR,
    top_k_per_chunk=10,
    min_chunk_score=0.04,
    min_votes=1,
    aggregation="max",
    final_threshold=0.0,
)

pairs = [
    (predictions.get(stem, set()), gold_map.get(stem, set()))
    for stem in gold_map
]
metrics = evaluate_batch(pairs)
print(f"Macro-avg  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
```

---

## 7. Comparison Summary

| Option | Complexity | New deps | Recall potential | Notes |
|--------|-----------|----------|-----------------|-------|
| A — Full-doc TF-IDF threshold | Low | None | Low–Med | Best starting point |
| B — Sentence-level aggregation | Low | None | Med | Good balance |
| C — sklearn multi-label classifier | Med | sklearn (already present) | Med–High | Needs train split |
| D — Semantic embeddings | High | sentence-transformers, faiss | High | Best quality, high cost |
| E — Chunk + retrieve + aggregate | Low–Med | None | Med–High | **Recommended** |

---

## 8. Recommendation & Suggested Path

1. **Start with Option E** (chunk + retrieve + aggregate). It reuses `TfidfRetriever` directly, adds no dependencies, and is easy to iterate on. Tune `top_k_per_chunk`, `min_chunk_score`, and `min_votes` on the dev set.

2. **Baseline with Option A** first to get a floor metric — one function call, 10 lines of code.

3. **Add Option D** once Option E is working, using `sentence-transformers` as a second-stage reranker: first retrieve a large candidate set with TF-IDF, then rerank with the cross-encoder and cut below a semantic threshold.

4. **Option C** is worth pursuing only after obtaining the CodiEsp training split, where the labelled examples will provide actual signal for the classifier.

---

## 9. Code File Layout Suggestion

```
src/
  retriever.py            # existing single-label TF-IDF retriever
  multilabel_retriever.py # new: Options A, B, E (chunk-level retrieval)
  semantic_retriever.py   # new (later): Option D embedding-based retriever
scripts/
  eval_multilabel.py      # batch evaluation on codiesp dev set
```
