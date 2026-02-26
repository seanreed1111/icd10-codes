"""TF-IDF cosine-similarity retriever for ICD-10-CM codes."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from knowledge_base import KnowledgeBase


@dataclass
class SearchResult:
    """A single search hit."""

    rank: int
    code: str
    description: str
    score: float


@dataclass
class SearchResultWithSiblings:
    """A search hit enriched with sibling codes from the same category."""

    rank: int
    code: str
    description: str
    score: float
    category_code: str
    siblings: list[dict[str, str]] = field(default_factory=list)


class TfidfRetriever:
    """Build a TF-IDF index over KnowledgeBase descriptions and search by cosine similarity."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self._codes: list[str] = [e.code for e in kb.entries]
        self._descriptions: list[str] = [e.description for e in kb.entries]

        # Pre-compute category → sibling map for search_with_siblings
        self._category_siblings: dict[str, list[dict[str, str]]] = defaultdict(list)
        for entry in kb.entries:
            cat = entry.code[:3]
            self._category_siblings[cat].append(
                {"code": entry.code, "description": entry.description}
            )

        # Fit TF-IDF vectorizer on corpus
        # token_pattern=r"(?u)\b(?:\w\w+|\d)\b"  matches either:
        # - \w\w+ — two or more word characters (existing behavior)
        # - \d — a single digit character

        # So "type 1 diabetes" tokenizes as ["type", "1", "diabetes"],
        # enabling the word-level bigrams "type 1" and "1 diabetes".
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            token_pattern=r"(?u)\b(?:\w\w+|\d)\b",
        )
        self._corpus_matrix = self._vectorizer.fit_transform(self._descriptions)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the *top_k* codes most similar to *query* by cosine score.

        Results are ordered by descending score. Entries with zero similarity
        are excluded, so the returned list may be shorter than *top_k*.
        """
        if top_k <= 0:
            return []
        query_vec = self._vectorizer.transform([query])
        scores: np.ndarray = linear_kernel(query_vec, self._corpus_matrix).flatten()
        top_indices: np.ndarray = scores.argsort()[-top_k:][::-1]
        return [
            SearchResult(
                rank=rank + 1,
                code=self._codes[idx],
                description=self._descriptions[idx],
                score=round(float(scores[idx]), 4),
            )
            for rank, idx in enumerate(top_indices)
            if scores[idx] > 0
        ]

    def search_with_siblings(
        self, query: str, top_k: int = 5
    ) -> list[SearchResultWithSiblings]:
        """Like :meth:`search`, but each result includes sibling codes.

        Siblings are all codes sharing the same 3-character category prefix,
        so the nurse can confirm the correct specificity level.
        """
        base_results = self.search(query, top_k)
        return [
            SearchResultWithSiblings(
                rank=r.rank,
                code=r.code,
                description=r.description,
                score=r.score,
                category_code=r.code[:3],
                siblings=self._category_siblings[r.code[:3]],
            )
            for r in base_results
        ]
