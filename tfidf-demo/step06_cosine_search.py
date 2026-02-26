"""Step 6: Cosine similarity search with linear_kernel.

Builds a search function that vectorizes a query and ranks all corpus
documents by cosine similarity. Uses linear_kernel (equivalent to
cosine_similarity when vectors are L2-normalized).
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # Fit on corpus
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    corpus_matrix = vectorizer.fit_transform(descriptions)
    print(f"Corpus: {corpus_matrix.shape[0]} docs, {corpus_matrix.shape[1]} features\n")

    def search(query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant codes for a query."""
        query_vec = vectorizer.transform([query])
        scores = linear_kernel(query_vec, corpus_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {
                "rank": rank + 1,
                "code": codes[idx],
                "description": descriptions[idx],
                "score": round(float(scores[idx]), 4),
            }
            for rank, idx in enumerate(top_indices)
            if scores[idx] > 0
        ]

    # --- Run several queries ---
    queries = [
        "myocardial infarction",
        "diabetes mellitus",
        "chronic kidney disease",
        "heart attack",  # synonym -- "attack" not in vocabulary, "heart" is
        "pulmonary disease",
        "hypertension",
        "broken leg fracture",  # completely outside vocabulary -- (no matches)
    ]

    for query in queries:
        results = search(query)
        print(f"Query: '{query}'")
        if results:
            for r in results[:3]:
                print(
                    f"  {r['rank']}. [{r['code']}] {r['description']}  (score={r['score']})"
                )
        else:
            print("  (no matches)")
        print()


if __name__ == "__main__":
    main()
