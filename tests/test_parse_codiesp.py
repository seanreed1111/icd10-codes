"""Tests for CodiEsp parser save/load functionality."""

from pathlib import Path

import polars as pl
import pytest

from scripts.parse_codiesp import load_from_parquet, save_to_parquet

SAMPLE_RESULTS: list[tuple[str, str]] = [
    ("S0010", "M545;M791"),
    ("S0024", "J00;J209"),
    ("S0033", "N390"),
]


@pytest.fixture()
def saved_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "codiesp_ground_truth.parquet"
    save_to_parquet(SAMPLE_RESULTS, path)
    return path


def test_save_creates_file(saved_parquet: Path) -> None:
    assert saved_parquet.exists()
    assert saved_parquet.stat().st_size > 0


def test_save_produces_readable_parquet(saved_parquet: Path) -> None:
    df = pl.read_parquet(saved_parquet)
    assert len(df) == len(SAMPLE_RESULTS)


def test_save_has_expected_columns(saved_parquet: Path) -> None:
    df = pl.read_parquet(saved_parquet)
    assert set(df.columns) == {"file_stem", "codes"}


def test_roundtrip_preserves_count(saved_parquet: Path) -> None:
    loaded = load_from_parquet(saved_parquet)
    assert len(loaded) == len(SAMPLE_RESULTS)


def test_roundtrip_preserves_data(saved_parquet: Path) -> None:
    loaded = load_from_parquet(saved_parquet)
    assert loaded == SAMPLE_RESULTS


def test_roundtrip_preserves_types(saved_parquet: Path) -> None:
    loaded = load_from_parquet(saved_parquet)
    for stem, codes in loaded:
        assert isinstance(stem, str)
        assert isinstance(codes, str)


def test_roundtrip_specific_entry(saved_parquet: Path) -> None:
    loaded = load_from_parquet(saved_parquet)
    stem, codes = loaded[0]
    assert stem == "S0010"
    assert codes == "M545;M791"


def test_save_empty_results(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    save_to_parquet([], path)
    loaded = load_from_parquet(path)
    assert loaded == []


def test_load_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_parquet(tmp_path / "does_not_exist.parquet")
