"""Step 2: TfidfVectorizer vs CountVectorizer + TfidfTransformer.

Demonstrates that TfidfVectorizer is a convenience wrapper that combines
CountVectorizer and TfidfTransformer into a single object. Both approaches
produce identical TF-IDF matrices.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # --- Approach A: TfidfVectorizer (one step) ---
    print("=== Approach A: TfidfVectorizer (one step) ===")
    vectorizer = TfidfVectorizer()
    matrix_a = vectorizer.fit_transform(descriptions)
    print(f"Shape: {matrix_a.shape}")
    print(f"Type:  {type(matrix_a)}")

    # --- Approach B: CountVectorizer + TfidfTransformer (two steps) ---
    print("\n=== Approach B: CountVectorizer + TfidfTransformer (two steps) ===")
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(descriptions)
    print(f"Count matrix shape: {count_matrix.shape}")

    tfidf_transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=False)
    matrix_b = tfidf_transformer.fit_transform(count_matrix)
    print(f"TF-IDF matrix shape: {matrix_b.shape}")

    # --- Verify they produce the same result ---
    print("\n=== Comparison ===")
    diff = np.abs(matrix_a.toarray() - matrix_b.toarray()).max()
    print(f"Max difference between approaches: {diff:.2e}")
    print(f"Identical (within float tolerance): {diff < 1e-10}")


if __name__ == "__main__":
    main()
