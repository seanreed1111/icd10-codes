"""Extract diagnosis codes from CodiEsp devX.tsv for each clinical text file."""

from pathlib import Path

import polars as pl

DEV_DIR = Path("data/test-datasets/codiesp/gold/final_dataset_v4_to_publish/dev")
TSV_PATH = DEV_DIR / "devX.tsv"
TEXT_DIR = DEV_DIR / "text_files_en"

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


if __name__ == "__main__":
    results = parse_codiesp_diagnostics()
    for stem, codes in results[0:2]:
        print(f"({stem},{codes!r})")
