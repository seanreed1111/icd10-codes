"""Tests for TfidfRetriever search methods."""

import json
from pathlib import Path

import polars as pl
import pytest

from knowledge_base import KnowledgeBase
from retriever import SearchResult, SearchResultWithSiblings, TfidfRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    ("A000", "Cholera due to Vibrio cholerae 01, biovar cholerae", "A00", 1),
    ("A001", "Cholera due to Vibrio cholerae 01, biovar eltor", "A00", 1),
    ("A009", "Cholera, unspecified", "A00", 1),
    ("A0100", "Typhoid fever, unspecified", "A01", 1),
    ("B0000", "Eczema herpeticum", "B00", 1),
    ("E1110", "Type 2 diabetes mellitus with ketoacidosis without coma", "E11", 4),
    ("E1165", "Type 2 diabetes mellitus with hyperglycemia", "E11", 4),
    (
        "I2510",
        "Atherosclerotic heart disease of native coronary artery without angina pectoris",
        "I25",
        9,
    ),
    ("J449", "Chronic obstructive pulmonary disease, unspecified", "J44", 10),
]


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Write a minimal enriched CSV that KnowledgeBase can parse."""
    csv_path = tmp_path / "sample.csv"
    pl.DataFrame(
        {
            "ICD10-CM-CODE": [r[0] for r in SAMPLE_ROWS],
            "description": [r[1] for r in SAMPLE_ROWS],
            "category_code": [r[2] for r in SAMPLE_ROWS],
            "chapter": [r[3] for r in SAMPLE_ROWS],
        }
    ).write_csv(csv_path)
    return csv_path


@pytest.fixture()
def kb(sample_csv: Path) -> KnowledgeBase:
    return KnowledgeBase(sample_csv)


@pytest.fixture()
def retriever(kb: KnowledgeBase) -> TfidfRetriever:
    return TfidfRetriever(kb)


# ---------------------------------------------------------------------------
# search() tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_returns_list_of_search_result(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_result_is_most_relevant(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera")
        assert len(results) > 0
        assert results[0].rank == 1
        # All cholera codes should be in the top results
        result_codes = {r.code for r in results}
        assert (
            "A000" in result_codes or "A001" in result_codes or "A009" in result_codes
        )

    def test_scores_are_descending(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera")
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_zero_score_results_excluded(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera")
        assert all(r.score > 0 for r in results)

    def test_top_k_limits_results(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("disease", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("xyznonexistentterm")
        assert results == []

    def test_empty_query_returns_empty(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("")
        assert results == []

    def test_top_k_zero_returns_empty(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("cholera", top_k=0)
        assert results == []

    def test_top_k_larger_than_corpus_returns_at_most_corpus_size(
        self, retriever: TfidfRetriever
    ) -> None:
        results = retriever.search("disease", top_k=100_000)
        assert len(results) <= len(retriever._descriptions)

    def test_diabetes_query(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("diabetes")
        result_codes = {r.code for r in results}
        assert "E1110" in result_codes or "E1165" in result_codes

    def test_pulmonary_disease_query(self, retriever: TfidfRetriever) -> None:
        results = retriever.search("pulmonary disease")
        assert results[0].code == "J449"


# ---------------------------------------------------------------------------
# search_with_siblings() tests
# ---------------------------------------------------------------------------


class TestSearchWithSiblings:
    def test_returns_search_result_with_siblings(
        self, retriever: TfidfRetriever
    ) -> None:
        results = retriever.search_with_siblings("cholera")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResultWithSiblings) for r in results)

    def test_siblings_share_category_prefix(self, retriever: TfidfRetriever) -> None:
        results = retriever.search_with_siblings("cholera")
        for r in results:
            assert r.category_code == r.code[:3]
            for sib in r.siblings:
                assert sib["code"][:3] == r.category_code

    def test_cholera_has_three_siblings(self, retriever: TfidfRetriever) -> None:
        results = retriever.search_with_siblings("cholera")
        # Find the result with category A00
        a00_results = [r for r in results if r.category_code == "A00"]
        assert len(a00_results) > 0
        # A00 category has 3 codes in our sample data
        assert len(a00_results[0].siblings) == 3

    def test_siblings_contain_code_and_description(
        self, retriever: TfidfRetriever
    ) -> None:
        results = retriever.search_with_siblings("cholera")
        for r in results:
            for sib in r.siblings:
                assert "code" in sib
                assert "description" in sib

    def test_scores_match_basic_search(self, retriever: TfidfRetriever) -> None:
        basic = retriever.search("diabetes", top_k=3)
        enriched = retriever.search_with_siblings("diabetes", top_k=3)
        assert len(basic) == len(enriched)
        for b, e in zip(basic, enriched, strict=True):
            assert b.code == e.code
            assert b.score == e.score


# ---------------------------------------------------------------------------
# Integration test against test_v1.json (uses full KB)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_retriever() -> TfidfRetriever:
    """Build retriever from the real enriched CSV (slow, cached per session)."""
    kb = KnowledgeBase()  # uses default CODES_FILE_PATH
    return TfidfRetriever(kb)


@pytest.fixture()
def test_cases() -> list[dict[str, str]]:
    path = Path(__file__).parent / "data" / "test_v1.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("idx", range(20), ids=[f"case-{i}" for i in range(20)])
def test_v1_expected_code_in_top_10(
    full_retriever: TfidfRetriever,
    test_cases: list[dict[str, str]],
    idx: int,
) -> None:
    """The expected code should appear somewhere in the top 10 results."""
    case = test_cases[idx]
    results = full_retriever.search(case["description"], top_k=10)
    result_codes = {r.code for r in results}
    assert case["expected_code"] in result_codes, (
        f"Expected {case['expected_code']} for '{case['description']}' "
        f"but got {[r.code for r in results[:5]]}"
    )
