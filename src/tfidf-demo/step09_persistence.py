"""Step 9: Persisting a fitted vectorizer.

Demonstrates saving and loading a fitted TfidfVectorizer and corpus matrix
with pickle, so you don't re-fit on every startup. Uses a temp directory
to keep the demo self-contained.
"""

import json
import pickle
import tempfile
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # --- Fit the vectorizer ---
    print("=== Fitting vectorizer ===")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    corpus_matrix = vectorizer.fit_transform(descriptions)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape:    {corpus_matrix.shape}")

    # --- Save to disk ---
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        vec_path = cache_dir / "vectorizer.pkl"
        mat_path = cache_dir / "corpus_matrix.pkl"

        print(f"\n=== Saving to {cache_dir} ===")
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(mat_path, "wb") as f:
            pickle.dump(corpus_matrix, f)

        print(f"vectorizer.pkl:    {vec_path.stat().st_size:,} bytes")
        print(f"corpus_matrix.pkl: {mat_path.stat().st_size:,} bytes")

        # --- Load from disk (simulating a fresh process) ---
        print("\n=== Loading from disk ===")
        with open(vec_path, "rb") as f:
            loaded_vec = pickle.load(f)
        with open(mat_path, "rb") as f:
            loaded_matrix = pickle.load(f)

        print(f"Loaded vocabulary size: {len(loaded_vec.vocabulary_)}")
        print(f"Loaded matrix shape:    {loaded_matrix.shape}")

        # --- Search using loaded objects (no re-fitting) ---
        print("\n=== Search with loaded vectorizer ===")
        query = "diabetes with kidney complications"
        query_vec = loaded_vec.transform([query])
        scores = linear_kernel(query_vec, loaded_matrix).flatten()
        top_indices = scores.argsort()[-3:][::-1]

        print(f"Query: '{query}'")
        for idx in top_indices:
            if scores[idx] > 0:
                print(
                    f"  [{codes[idx]}] {descriptions[idx]}  (score={scores[idx]:.4f})"
                )

    print("\n(temp directory cleaned up)")


if __name__ == "__main__":
    main()
