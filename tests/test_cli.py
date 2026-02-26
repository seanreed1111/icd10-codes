"""Tests for the CLI (cli.py) using typer.testing.CliRunner."""

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from cli import app
from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever

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
    ("J449", "Chronic obstructive pulmonary disease, unspecified", "J44", 10),
]

runner = CliRunner()


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
def patched_retriever(
    sample_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> TfidfRetriever:
    """Replace _build_retriever with one backed by sample data."""
    retriever = TfidfRetriever(KnowledgeBase(sample_csv))
    monkeypatch.setattr("cli._build_retriever", lambda: retriever)
    return retriever


# ---------------------------------------------------------------------------
# search — basic output
# ---------------------------------------------------------------------------


class TestSearchBasic:
    def test_exits_zero(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera"])
        assert result.exit_code == 0

    def test_top_result_rank_present(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera"])
        assert "[1]" in result.output

    def test_score_present(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera"])
        assert "score=" in result.output

    def test_relevant_code_in_output(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera"])
        codes = {"A000", "A001", "A009"}
        assert any(c in result.output for c in codes)

    def test_description_in_output(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera"])
        assert "Cholera" in result.output

    def test_diabetes_query(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["diabetes"])
        assert "E1110" in result.output or "E1165" in result.output


# ---------------------------------------------------------------------------
# search — --top-k / -k flag
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top_k_limits_to_one(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["disease", "--top-k", "1"])
        assert "[1]" in result.output
        assert "[2]" not in result.output

    def test_short_flag_k_works(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["disease", "-k", "1"])
        assert result.exit_code == 0
        assert "[2]" not in result.output

    def test_top_k_two_returns_two_ranks(
        self, patched_retriever: TfidfRetriever
    ) -> None:
        result = runner.invoke(app, ["diabetes", "--top-k", "2"])
        assert "[1]" in result.output
        assert "[2]" in result.output
        assert "[3]" not in result.output


# ---------------------------------------------------------------------------
# search — no results
# ---------------------------------------------------------------------------


class TestNoResults:
    def test_no_results_message_in_output(
        self, patched_retriever: TfidfRetriever
    ) -> None:
        # stderr is mixed into output by default
        result = runner.invoke(app, ["xyznonexistentterm"])
        assert "No results found." in result.output

    def test_no_results_exits_zero(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["xyznonexistentterm"])
        assert result.exit_code == 0

    def test_no_results_no_rank_lines(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["xyznonexistentterm"])
        assert "[1]" not in result.output


# ---------------------------------------------------------------------------
# search — --siblings / -s flag
# ---------------------------------------------------------------------------


class TestSiblings:
    def test_siblings_flag_exits_zero(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera", "--siblings"])
        assert result.exit_code == 0

    def test_short_flag_s_exits_zero(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera", "-s"])
        assert result.exit_code == 0

    def test_category_line_present(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera", "--siblings"])
        assert "Category:" in result.output

    def test_sibling_count_present(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera", "--siblings"])
        assert "sibling(s)" in result.output

    def test_all_category_siblings_listed(
        self, patched_retriever: TfidfRetriever
    ) -> None:
        result = runner.invoke(app, ["cholera", "--siblings"])
        # A00 has 3 codes; all should appear in the sibling list
        assert "A000" in result.output
        assert "A001" in result.output
        assert "A009" in result.output

    def test_matched_code_has_arrow_marker(
        self, patched_retriever: TfidfRetriever
    ) -> None:
        result = runner.invoke(app, ["cholera unspecified", "--siblings"])
        # "Cholera, unspecified" → A009 should get the ">" marker
        assert ">" in result.output

    def test_siblings_top_k_respected(self, patched_retriever: TfidfRetriever) -> None:
        result = runner.invoke(app, ["cholera", "--siblings", "--top-k", "1"])
        assert "[1]" in result.output
        assert "[2]" not in result.output

    def test_siblings_scores_match_basic_search(
        self, patched_retriever: TfidfRetriever
    ) -> None:
        basic = runner.invoke(app, ["diabetes", "--top-k", "2"])
        enriched = runner.invoke(app, ["diabetes", "--siblings", "--top-k", "2"])
        # Both should list the same top code
        assert "[1]" in basic.output
        assert "[1]" in enriched.output
