"""Step 3: fit(), transform(), and fit_transform().

Shows the three core methods and why you must use transform() (not
fit_transform()) on new queries after fitting on the corpus.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # --- fit() then transform() separately ---
    print("=== fit() then transform() ===")
    v1 = TfidfVectorizer()
    v1.fit(descriptions)
    matrix1 = v1.transform(descriptions)
    print(f"Vocabulary size: {len(v1.vocabulary_)}")
    print(f"Matrix shape:    {matrix1.shape}")
    print(f"Matrix type:     {type(matrix1)}")

    # --- fit_transform() in one pass ---
    print("\n=== fit_transform() (one pass) ===")
    v2 = TfidfVectorizer()
    matrix2 = v2.fit_transform(descriptions)
    print(f"Vocabulary size: {len(v2.vocabulary_)}")
    print(f"Matrix shape:    {matrix2.shape}")

    # --- Verify identical ---
    diff = np.abs(matrix1.toarray() - matrix2.toarray()).max()
    print(f"\nMax difference: {diff:.2e} (should be ~0)")

    # --- Critical rule: transform() for queries ---
    print("\n=== Querying with transform() ===")
    query = "myocardial infarction"
    query_vec = v2.transform([query])
    print(
        f"Query vector shape: {query_vec.shape}  (1 doc x {query_vec.shape[1]} features)"
    )
    print(f"Non-zero features:  {query_vec.nnz}")

    # --- What goes wrong with fit_transform() on a query ---
    print("\n=== BAD: fit_transform() on a query (don't do this!) ===")
    v_bad = TfidfVectorizer()
    bad_vec = v_bad.fit_transform([query])
    print(f"Bad vector shape:  {bad_vec.shape}  (only {bad_vec.shape[1]} features!)")
    print(f"Corpus had {matrix2.shape[1]} features — the query vector is incompatible")


if __name__ == "__main__":
    main()
