# sklearn TF-IDF — Step-by-Step Guide

A practical walkthrough of `TfidfVectorizer` for building text search, using a medical code dataset as the running example.

---

## The Math (briefly)

```
TF(term, doc)  = raw count of term in doc
                 (or 1 + log(count) if sublinear_tf=True)

IDF(term)      = log((1 + n_docs) / (1 + doc_freq)) + 1   # smooth_idf=True (default)

TF-IDF(term, doc) = TF * IDF

# Each doc vector is then L2-normalized:
vector_norm = vector / ||vector||
```

**Intuition**: a term scores high when it appears often in *this* document but rarely across *other* documents — making it a discriminating signal.

---

## Mock Dataset

All examples below use this dataset so you can run them end-to-end.

```python
DESCRIPTIONS = [
    "acute myocardial infarction of anterior wall",     # 0
    "acute myocardial infarction of inferior wall",     # 1
    "chronic ischemic heart disease",                   # 2
    "type 2 diabetes mellitus with diabetic nephropathy",  # 3
    "type 2 diabetes mellitus without complications",   # 4
    "essential primary hypertension",                   # 5
    "hypertensive chronic kidney disease",              # 6
    "chronic obstructive pulmonary disease unspecified",   # 7
    "pneumonia due to streptococcus pneumoniae",        # 8
    "asthma mild intermittent uncomplicated",           # 9
]

CODES = [
    "I21.0", "I21.1", "I25.9",
    "E11.65", "E11.9",
    "I10", "I12.9",
    "J44.1", "J13", "J45.20",
]
```

---

## Step 1 — Install

```bash
uv add scikit-learn scipy numpy pandas
```

---

## Step 2 — `TfidfVectorizer` vs `TfidfTransformer`

| Class | Input | Does tokenizing? | Use when |
|-------|-------|-----------------|----------|
| `TfidfVectorizer` | raw strings | Yes | Almost always |
| `TfidfTransformer` | count matrix | No | You already have counts from `CountVectorizer` |

`TfidfVectorizer` is `CountVectorizer` + `TfidfTransformer` in one object. Prefer it for search tasks.

---

## Step 3 — `fit`, `transform`, and `fit_transform`

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

# fit() — learn vocabulary and IDF weights from corpus; returns nothing useful
vectorizer.fit(DESCRIPTIONS)

# transform() — apply learned vocab+IDF to produce TF-IDF matrix
# Use on corpus AFTER fit, or on new queries
corpus_matrix = vectorizer.transform(DESCRIPTIONS)

# fit_transform() — fit + transform in one efficient pass
# Use this on the training corpus; use transform() for any new query
vectorizer2 = TfidfVectorizer()
corpus_matrix2 = vectorizer2.fit_transform(DESCRIPTIONS)
# corpus_matrix2 is identical to corpus_matrix

print(corpus_matrix2.shape)   # (10, N_unique_tokens)
print(type(corpus_matrix2))   # <class 'scipy.sparse._csr.csr_matrix'>
```

**Critical rule**: always call `fit_transform()` on the corpus, then `transform()` (never `fit_transform()`) on individual queries. Calling `fit_transform()` on a query relearns the vocabulary from that one string alone.

---

## Step 4 — Inspect the Output

The return value is a **sparse matrix** — most entries are zero (a document only uses a tiny fraction of the full vocabulary).

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(DESCRIPTIONS)

# --- vocabulary ---
features = vectorizer.get_feature_names_out()
print(features[:10])
# e.g. ['acute' 'anterior' 'asthma' 'chronic' 'complications' ...]

# --- IDF weights: higher = rarer = more discriminating ---
idf_series = pd.Series(vectorizer.idf_, index=features).sort_values()
print(idf_series.head())    # lowest IDF (most common terms)
print(idf_series.tail())    # highest IDF (rarest terms)

# --- dense view (only safe for small corpora) ---
dense = X.toarray()
df = pd.DataFrame(dense, columns=features, index=CODES)
print(df.round(3))

# --- sparsity ---
total = X.shape[0] * X.shape[1]
print(f"Sparsity: {1 - X.nnz / total:.1%}")   # e.g. 94.3%

# --- non-zero terms for one document ---
doc_idx = 0
vec = X[doc_idx]
for col in vec.nonzero()[1]:
    print(f"  {features[col]}: {vec[0, col]:.4f}")
```

---

## Step 5 — Key Parameters

### `ngram_range` — single words vs phrases

```python
corpus = ["type 2 diabetes mellitus", "diabetes mellitus type 1"]

# (1, 1) unigrams only — loses the distinction between "type 1" and "type 2"
v1 = TfidfVectorizer(ngram_range=(1, 1))
print(v1.fit(corpus).get_feature_names_out())
# ['diabetes' 'mellitus' 'type']

# (1, 2) unigrams + bigrams — "type 2" and "type 1" are now distinct features
v2 = TfidfVectorizer(ngram_range=(1, 2))
print(v2.fit(corpus).get_feature_names_out())
# ['diabetes' 'diabetes mellitus' 'mellitus' 'mellitus type' 'type' 'type 1' 'type 2']
```

**Recommendation**: `(1, 2)` for medical descriptions — preserves "myocardial infarction", "type 2", "kidney disease" as units.

### `analyzer` — word tokens vs character n-grams

```python
# 'word' (default): split on whitespace/punctuation
word_vec = TfidfVectorizer(analyzer='word')

# 'char_wb': character n-grams within word boundaries — handles typos and abbreviations
# "diabetes" → " diabetes " → extracts "dia", "iab", "abe", "bet", "ete", "tes"
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))

# Hybrid: stack both for typo-robustness + semantic meaning (see Step 7)
```

### `stop_words` — remove noise words

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Built-in English list (318 words): removes "the", "of", "with", "is", ...
v = TfidfVectorizer(stop_words='english')

# Custom: remove domain-specific noise while keeping clinically meaningful words
medical_noise = ['unspecified', 'elsewhere', 'classified', 'other']
combined = list(ENGLISH_STOP_WORDS.union(medical_noise))
v = TfidfVectorizer(stop_words=combined)
```

**Warning**: words like "not", "without", "without complications" carry clinical meaning in ICD-10 descriptions. Consider carefully before removing them.

### `min_df` and `max_df` — frequency filtering

```python
# Drop terms appearing in fewer than 2 documents (single-code-specific tokens)
# Drop terms appearing in more than 90% of documents (near-universal, uninformative)
v = TfidfVectorizer(min_df=2, max_df=0.9)

# As absolute counts instead of proportions:
v = TfidfVectorizer(min_df=2, max_df=5000)
```

### `sublinear_tf` — dampen term frequency

```python
# False (default): tf = raw count
# True:            tf = 1 + log(raw count)
# Prevents a term appearing 100x from being 100x more important than one appearing 1x

v = TfidfVectorizer(sublinear_tf=True)
```

For short ICD-10 descriptions (3–10 words each), no term repeats much within one doc, so this matters less. Still a good habit for general text.

### `max_features` — vocabulary size cap

```python
# None (default): use all terms found
# int: keep only the N most frequent terms across corpus

# For medical search, keep None — don't prune rare but specific terms
# like "takotsubo", "klebsiella", "pneumoniae"
v = TfidfVectorizer(max_features=None)
```

---

## Step 6 — Cosine Similarity Search

After fitting, convert a user query to a vector and score it against every document in the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Fit on corpus
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
corpus_matrix = vectorizer.fit_transform(DESCRIPTIONS)

def search(query: str, top_k: int = 5) -> list[dict]:
    # transform(), NOT fit_transform() — use the corpus vocabulary
    query_vec = vectorizer.transform([query])

    # linear_kernel == cosine_similarity when vectors are L2-normalized (default)
    # Faster because it skips recomputing norms
    scores = linear_kernel(query_vec, corpus_matrix).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    return [
        {
            "rank": rank + 1,
            "code": CODES[idx],
            "description": DESCRIPTIONS[idx],
            "score": round(float(scores[idx]), 4),
        }
        for rank, idx in enumerate(top_indices)
        if scores[idx] > 0
    ]

# Try it
for result in search("heart attack"):
    print(f"  {result['rank']}. [{result['code']}] {result['description']}  ({result['score']})")
```

**`linear_kernel` vs `cosine_similarity`**: mathematically identical when `norm='l2'` (the default). `linear_kernel` is faster for large corpora because it avoids recomputing norms that were already applied during `fit_transform`.

---

## Step 7 — Hybrid Word + Character Vectorizer (typo robustness)

For medical search where nurses may misspell terms, combine word-level and character-level features.

```python
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))

word_matrix = word_vec.fit_transform(DESCRIPTIONS)
char_matrix = char_vec.fit_transform(DESCRIPTIONS)
combined_matrix = hstack([word_matrix, char_matrix])   # horizontally stack features

def search_hybrid(query: str, top_k: int = 5) -> list[dict]:
    q = hstack([word_vec.transform([query]), char_vec.transform([query])])
    scores = linear_kernel(q, combined_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [
        {"code": CODES[i], "description": DESCRIPTIONS[i], "score": round(float(scores[i]), 4)}
        for i in top_indices if scores[i] > 0
    ]

# Handles typos via character overlap
print(search_hybrid("myocardal infarcton"))   # typos in both words
print(search_hybrid("diabtes kidney"))        # partial + misspelled
```

---

## Step 8 — Debugging: What Did the Vectorizer See?

Use this to understand why a query matched or didn't match.

```python
def explain_query(vectorizer, query: str):
    vec = vectorizer.transform([query])
    features = vectorizer.get_feature_names_out()
    nonzero = vec.nonzero()[1]
    print(f"Query: '{query}' → {len(nonzero)} matched features")
    for col in nonzero:
        print(f"  '{features[col]}':  tfidf={vec[0, col]:.4f}  idf={vectorizer.idf_[col]:.4f}")
    if len(nonzero) == 0:
        print("  (no terms in query match the vocabulary — all scores will be 0)")

explain_query(vectorizer, "heart attack")
# Query: 'heart attack' → 0 matched features
# (no terms in query match the vocabulary — all scores will be 0)
# → "heart attack" is not in any ICD-10 description; synonyms are needed

explain_query(vectorizer, "myocardial infarction anterior")
# Query: 'myocardial infarction anterior' → 3 matched features
#   'myocardial': tfidf=0.5774  idf=1.5108
#   'infarction':  tfidf=0.5774  idf=1.5108
#   'anterior':   tfidf=0.5774  idf=2.0986
```

---

## Step 9 — Persisting a Fitted Vectorizer

Re-fitting 74,719 descriptions on every startup is wasteful. Persist the fitted objects.

```python
import pickle
from pathlib import Path

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save
with open(CACHE_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(CACHE_DIR / "corpus_matrix.pkl", "wb") as f:
    pickle.dump(corpus_matrix, f)

# Load — no re-fitting needed
with open(CACHE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(CACHE_DIR / "corpus_matrix.pkl", "rb") as f:
    corpus_matrix = pickle.load(f)

# transform() works immediately after loading
query_vec = vectorizer.transform(["diabetes with kidney complications"])
```

---

## Step 10 — Recommended Parameters for ICD-10 Search

| Parameter | Value | Reason |
|-----------|-------|--------|
| `ngram_range` | `(1, 2)` | Captures multi-word clinical phrases |
| `analyzer` | `'word'` | Fast; use `'char_wb'` variant if typos are common |
| `stop_words` | `'english'` | Removes noise; audit for clinical words first |
| `min_df` | `1` | Keep all rare but specific medical terms |
| `max_df` | `0.9` | Drop near-universal terms like "unspecified" |
| `max_features` | `None` | Don't prune rare tokens |
| `sublinear_tf` | `True` | Good habit; minimal effect on short descriptions |
| `norm` | `'l2'` (default) | Required for `linear_kernel` to equal cosine similarity |
| `smooth_idf` | `True` (default) | Prevents zero-division on unseen query terms |

---

## Known Limitations

1. **No synonym handling**: "heart attack" scores 0 against a corpus that only contains "myocardial infarction". Solve with a synonym expansion layer before search, or use embeddings (e.g., SentenceTransformers) as a complement.

2. **Out-of-vocabulary tokens**: any query token not seen during `fit()` is silently ignored. Check with `explain_query()`.

3. **Short documents**: ICD-10 descriptions are 3–10 words. TF is almost always 1, so IDF dominates — this is fine and actually desirable for retrieval.

4. **No ranking context**: TF-IDF treats each code as isolated. It cannot reason that "I21.0" and "I21.1" are siblings under category "I21".
