"""Step 4: Inspecting the TF-IDF output.

Explores the sparse matrix, vocabulary, IDF weights, sparsity percentage,
and per-document non-zero terms.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(descriptions)

    # --- Vocabulary ---
    features = vectorizer.get_feature_names_out()
    print("=== Vocabulary (first 15 tokens) ===")
    print(features[:15])
    print(f"Total features: {len(features)}")

    # --- IDF weights ---
    print("\n=== IDF Weights (most common to rarest) ===")
    idf_series = pd.Series(vectorizer.idf_, index=features).sort_values()
    print("Lowest IDF (most common):")
    print(idf_series.head())
    print("\nHighest IDF (rarest):")
    print(idf_series.tail())

    # --- Dense view ---
    print("\n=== Dense TF-IDF Matrix ===")
    dense = X.toarray()
    df = pd.DataFrame(dense, columns=features, index=codes)
    print(df.round(3).to_string())

    # --- Sparsity ---
    total = X.shape[0] * X.shape[1]
    print("\n=== Sparsity ===")
    print(f"Shape: {X.shape}")
    print(f"Non-zero entries: {X.nnz} / {total}")
    print(f"Sparsity: {1 - X.nnz / total:.1%}")

    # --- Non-zero terms for first document ---
    print(f"\n=== Non-zero terms for doc 0: '{descriptions[0]}' ===")
    doc_vec = X[0]
    for col in doc_vec.nonzero()[1]:
        print(f"  {features[col]:>20s}: {doc_vec[0, col]:.4f}")


if __name__ == "__main__":
    main()
