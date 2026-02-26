"""Step 7: Hybrid word + character n-gram search.

Combines a word-level TfidfVectorizer (semantic meaning) with a char_wb
vectorizer (typo robustness) by horizontally stacking their sparse matrices.
"""

import json
from pathlib import Path

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # Two vectorizers: word-level + character-level
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), stop_words="english"
    )
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    word_matrix = word_vec.fit_transform(descriptions)
    char_matrix = char_vec.fit_transform(descriptions)
    combined_matrix = hstack([word_matrix, char_matrix])

    print(f"Word features:     {word_matrix.shape[1]}")
    print(f"Char features:     {char_matrix.shape[1]}")
    print(f"Combined features: {combined_matrix.shape}\n")

    def search_hybrid(query: str, top_k: int = 5) -> list[dict]:
        q = hstack([word_vec.transform([query]), char_vec.transform([query])])
        scores = linear_kernel(q, combined_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {
                "code": codes[i],
                "description": descriptions[i],
                "score": round(float(scores[i]), 4),
            }
            for i in top_indices
            if scores[i] > 0
        ]

    # --- Compare: correct spelling vs typos ---
    test_queries = [
        ("myocardial infarction", "correct spelling"),
        ("myocardal infarcton", "typos in both words"),
        ("diabtes kidney", "partial + misspelled"),
        ("pnuemonia streptococcus", "misspelled pneumonia"),
        ("asthma", "exact match"),
    ]

    for query, note in test_queries:
        results = search_hybrid(query)
        print(f"Query: '{query}' ({note})")
        for r in results[:3]:
            print(f"  [{r['code']}] {r['description']}  (score={r['score']})")
        print()


if __name__ == "__main__":
    main()
