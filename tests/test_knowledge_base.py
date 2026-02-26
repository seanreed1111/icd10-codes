"""Tests for KnowledgeBase.save / KnowledgeBase.load round-trip."""

from pathlib import Path

import polars as pl
import pytest

from knowledge_base import ICD10Code, KnowledgeBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    ("A000", "Cholera due to Vibrio cholerae 01, biovar cholerae", "A00", 1),
    ("A001", "Cholera due to Vibrio cholerae 01, biovar eltor", "A00", 1),
    ("A009", "Cholera, unspecified", "A00", 1),
    ("A0100", "Typhoid fever, unspecified", "A01", 1),
    ("B0000", "Eczema herpeticum", "B00", 1),
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
            # category_description / section omitted — not consumed by _construct_entries
        }
    ).write_csv(csv_path)
    return csv_path


@pytest.fixture()
def kb(sample_csv: Path) -> KnowledgeBase:
    return KnowledgeBase(sample_csv)


@pytest.fixture()
def saved_parquet(kb: KnowledgeBase, tmp_path: Path) -> Path:
    path = tmp_path / "kb.parquet"
    kb.save(path)
    return path


# ---------------------------------------------------------------------------
# save() tests
# ---------------------------------------------------------------------------


def test_save_creates_file(saved_parquet: Path) -> None:
    assert saved_parquet.exists()
    assert saved_parquet.stat().st_size > 0


def test_save_produces_readable_parquet(saved_parquet: Path) -> None:
    df = pl.read_parquet(saved_parquet)
    assert len(df) == len(SAMPLE_ROWS)


def test_save_parquet_has_expected_columns(saved_parquet: Path) -> None:
    df = pl.read_parquet(saved_parquet)
    assert set(df.columns) == {
        "ICD10-CM-CODE",
        "description",
        "description_aliases",
        "category_code",
        "chapter",
    }


# ---------------------------------------------------------------------------
# load() tests
# ---------------------------------------------------------------------------


def test_load_returns_knowledge_base(saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    assert isinstance(loaded, KnowledgeBase)


def test_load_file_path_set(saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    assert loaded.file_path == saved_parquet


# ---------------------------------------------------------------------------
# Round-trip integrity tests
# ---------------------------------------------------------------------------


def test_roundtrip_entry_count(kb: KnowledgeBase, saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    assert len(loaded.entries) == len(kb.entries)


def test_roundtrip_entries_are_icd10code(saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    assert all(isinstance(e, ICD10Code) for e in loaded.entries)


def test_roundtrip_codes_preserved(kb: KnowledgeBase, saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    original_codes = [e.code for e in kb.entries]
    loaded_codes = [e.code for e in loaded.entries]
    assert loaded_codes == original_codes


def test_roundtrip_descriptions_preserved(
    kb: KnowledgeBase, saved_parquet: Path
) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    for orig, reloaded in zip(kb.entries, loaded.entries, strict=True):
        assert reloaded.description == orig.description


def test_roundtrip_category_preserved(kb: KnowledgeBase, saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    for orig, reloaded in zip(kb.entries, loaded.entries, strict=True):
        assert reloaded.category == orig.category


def test_roundtrip_chapter_preserved(kb: KnowledgeBase, saved_parquet: Path) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    for orig, reloaded in zip(kb.entries, loaded.entries, strict=True):
        assert reloaded.chapter == orig.chapter


def test_roundtrip_description_aliases_preserved(
    kb: KnowledgeBase, saved_parquet: Path
) -> None:
    loaded = KnowledgeBase.load(saved_parquet)
    for orig, reloaded in zip(kb.entries, loaded.entries, strict=True):
        assert reloaded.description_aliases == orig.description_aliases


def test_roundtrip_specific_entry(saved_parquet: Path) -> None:
    """Spot-check a specific known row end-to-end."""
    loaded = KnowledgeBase.load(saved_parquet)
    entry = loaded.entries[2]  # A009
    assert entry.code == "A009"
    assert entry.description == "Cholera, unspecified"
    assert entry.category == "A00"
    assert entry.chapter == 1
