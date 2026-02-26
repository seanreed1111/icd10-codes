"""Step 8: Debugging -- what did the vectorizer see?

Uses explain_query() to show exactly which tokens in a query matched the
learned vocabulary, their TF-IDF weights, and IDF values. Helps diagnose
why a query did or didn't match.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def explain_query(vectorizer: TfidfVectorizer, query: str) -> None:
    """Print which query tokens matched the vocabulary and their weights."""
    vec = vectorizer.transform([query])
    features = vectorizer.get_feature_names_out()
    nonzero = vec.nonzero()[1]
    print(f"Query: '{query}' -> {len(nonzero)} matched feature(s)")
    for col in nonzero:
        print(
            f"  '{features[col]}':  tfidf={vec[0, col]:.4f}  idf={vectorizer.idf_[col]:.4f}"
        )
    if len(nonzero) == 0:
        print("  (no terms in query match the vocabulary -- all scores will be 0)")


def main() -> None:
    descriptions, codes = load_dataset()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectorizer.fit_transform(descriptions)

    print("=== Queries that MATCH vocabulary ===\n")
    explain_query(vectorizer, "myocardial infarction anterior")
    print()
    explain_query(vectorizer, "diabetes mellitus")
    print()
    explain_query(vectorizer, "chronic disease")
    print()

    print("=== Queries that DON'T match (synonyms/slang) ===\n")
    explain_query(vectorizer, "heart attack")
    print()
    explain_query(vectorizer, "sugar disease")
    print()
    explain_query(vectorizer, "high blood pressure")
    print()

    print("=== Partial matches ===\n")
    explain_query(vectorizer, "acute myocardial something unknown")
    print()
    explain_query(vectorizer, "type 2 diabetes with renal failure")


if __name__ == "__main__":
    main()
