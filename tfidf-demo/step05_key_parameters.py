"""Step 5: Key TfidfVectorizer parameters.

Demonstrates ngram_range, analyzer, stop_words, min_df/max_df,
sublinear_tf, and max_features with concrete before/after output.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


def load_dataset() -> tuple[list[str], list[str]]:
    data = json.loads((Path(__file__).parent / "dataset.json").read_text())
    return [d["description"] for d in data], [d["code"] for d in data]


def main() -> None:
    descriptions, codes = load_dataset()

    # =============================================
    # ngram_range
    # =============================================
    print("=" * 60)
    print("ngram_range - single words vs phrases")
    print("=" * 60)

    subset = ["type 2 diabetes mellitus", "diabetes mellitus type 1"]

    v1 = TfidfVectorizer(ngram_range=(1, 1))
    v1.fit(subset)
    print(f"\n(1,1) unigrams only:  {list(v1.get_feature_names_out())}")

    v2 = TfidfVectorizer(ngram_range=(1, 2))
    v2.fit(subset)
    print(f"(1,2) uni+bigrams:    {list(v2.get_feature_names_out())}")

    v3 = TfidfVectorizer(ngram_range=(1, 3))
    v3.fit(subset)
    print(f"(1,3) uni+bi+trigrams: {list(v3.get_feature_names_out())}")

    # Full dataset comparison
    v_uni = TfidfVectorizer(ngram_range=(1, 1))
    v_bi = TfidfVectorizer(ngram_range=(1, 2))
    v_uni.fit(descriptions)
    v_bi.fit(descriptions)
    print(f"\nFull dataset - unigrams: {len(v_uni.get_feature_names_out())} features")
    print(f"Full dataset - uni+bi:   {len(v_bi.get_feature_names_out())} features")

    # =============================================
    # analyzer
    # =============================================
    print("\n" + "=" * 60)
    print("analyzer - word vs char_wb")
    print("=" * 60)

    word_vec = TfidfVectorizer(analyzer="word")
    word_vec.fit(descriptions)
    print(f"\nanalyzer='word':    {len(word_vec.get_feature_names_out())} features")
    print(f"  Sample: {list(word_vec.get_feature_names_out()[:8])}")

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    char_vec.fit(descriptions)
    print(f"\nanalyzer='char_wb': {len(char_vec.get_feature_names_out())} features")
    print(f"  Sample: {list(char_vec.get_feature_names_out()[:8])}")

    # =============================================
    # stop_words
    # =============================================
    print("\n" + "=" * 60)
    print("stop_words - removing noise words")
    print("=" * 60)

    v_no_stop = TfidfVectorizer(stop_words=None)
    v_no_stop.fit(descriptions)
    feats_no_stop = set(v_no_stop.get_feature_names_out())

    v_en_stop = TfidfVectorizer(stop_words="english")
    v_en_stop.fit(descriptions)
    feats_en_stop = set(v_en_stop.get_feature_names_out())

    removed = feats_no_stop - feats_en_stop
    print(f"\nWithout stop_words: {len(feats_no_stop)} features")
    print(f"With stop_words='english': {len(feats_en_stop)} features")
    print(f"Removed: {sorted(removed)}")

    # Custom medical stop words
    medical_noise = ["unspecified", "elsewhere", "classified", "other"]
    combined = list(ENGLISH_STOP_WORDS.union(medical_noise))
    v_custom = TfidfVectorizer(stop_words=combined)
    v_custom.fit(descriptions)
    print(
        f"With custom medical stops: {len(v_custom.get_feature_names_out())} features"
    )

    # =============================================
    # min_df and max_df
    # =============================================
    print("\n" + "=" * 60)
    print("min_df / max_df - frequency filtering")
    print("=" * 60)

    for min_df_val in [1, 2, 3]:
        v = TfidfVectorizer(min_df=min_df_val)
        v.fit(descriptions)
        print(f"  min_df={min_df_val}: {len(v.get_feature_names_out())} features")

    for max_df_val in [1.0, 0.5, 0.3]:
        v = TfidfVectorizer(max_df=max_df_val)
        v.fit(descriptions)
        print(f"  max_df={max_df_val}: {len(v.get_feature_names_out())} features")

    # =============================================
    # sublinear_tf
    # =============================================
    print("\n" + "=" * 60)
    print("sublinear_tf - dampening term frequency")
    print("=" * 60)

    v_linear = TfidfVectorizer(sublinear_tf=False)
    v_sublin = TfidfVectorizer(sublinear_tf=True)
    X_lin = v_linear.fit_transform(descriptions)
    X_sub = v_sublin.fit_transform(descriptions)

    # Compare first document's vectors
    feats = v_linear.get_feature_names_out()
    doc0_lin = X_lin[0].toarray().flatten()
    doc0_sub = X_sub[0].toarray().flatten()

    print(f"\nDoc 0: '{descriptions[0]}'")
    for i, feat in enumerate(feats):
        if doc0_lin[i] > 0 or doc0_sub[i] > 0:
            print(
                f"  {feat:>20s}:  linear={doc0_lin[i]:.4f}  sublinear={doc0_sub[i]:.4f}"
            )

    # =============================================
    # max_features
    # =============================================
    print("\n" + "=" * 60)
    print("max_features - vocabulary size cap")
    print("=" * 60)

    for max_f in [5, 10, 20, None]:
        v = TfidfVectorizer(max_features=max_f)
        v.fit(descriptions)
        feats = v.get_feature_names_out()
        label = str(max_f) if max_f else "None (all)"
        print(
            f"  max_features={label:>10s}: {len(feats)} features -> {list(feats[:6])}..."
        )


if __name__ == "__main__":
    main()
