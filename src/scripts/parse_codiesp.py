"""Extract diagnosis codes from CodiEsp devX.tsv for each clinical text file."""

from pathlib import Path

import polars as pl

DEV_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev")
TSV_PATH = DEV_DIR / "devX.tsv"
TEXT_DIR = DEV_DIR / "text_files_en"

OUTPUT_PATH = Path(
    "data/test-datasets/codiesp/data-pipeline/processed/gold/codiesp_ground_truth.parquet"
)

COLUMN_NAMES = ["file_stem", "type", "code", "description", "span"]


def parse_codiesp_diagnostics() -> list[tuple[str, str]]:
    """Parse devX.tsv and return (file_stem, semicolon_joined_codes) for each text file.

    Only rows with type == "DIAGNOSTICO" are included. Files with no
    DIAGNOSTICO rows are omitted from the result.
    """
    text_stems = {p.stem for p in TEXT_DIR.glob("*.txt")}

    df = (
        pl.read_csv(
            TSV_PATH,
            separator="\t",
            has_header=False,
            new_columns=COLUMN_NAMES,
        )
        .filter(
            pl.col("type").eq("DIAGNOSTICO")
            & pl.col("file_stem").is_in(list(text_stems))
        )
        .with_columns(pl.col("code").str.to_uppercase().str.replace_all(r"[^\w]", ""))
        .group_by("file_stem", maintain_order=True)
        .agg(pl.col("code").str.join(";").alias("codes"))
    )

    return list(zip(df["file_stem"].to_list(), df["codes"].to_list(), strict=True))


def save_to_parquet(results: list[tuple[str, str]], path: Path = OUTPUT_PATH) -> None:
    """Write parsed results to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "file_stem": [r[0] for r in results],
            "codes": [r[1] for r in results],
        }
    ).write_parquet(path)


def load_from_parquet(path: Path = OUTPUT_PATH) -> list[tuple[str, str]]:
    """Read a previously saved Parquet file back into a list of (file_stem, codes) tuples."""
    df = pl.read_parquet(path)
    return list(zip(df["file_stem"].to_list(), df["codes"].to_list(), strict=True))


if __name__ == "__main__":
    results = parse_codiesp_diagnostics()
    for stem, codes in results[0:2]:
        print(f"({stem},{codes!r})")
    save_to_parquet(results)
    print(f"\nSaved {len(results)} entries to {OUTPUT_PATH}")
