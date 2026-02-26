from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
CODES_FILE_NAME = "icd10cm-codes-enriched-April-1-2026.csv"
CODES_FILE_PATH = PROCESSED / CODES_FILE_NAME


@dataclass
class Category:
    code: str
    description: str


@dataclass
class Chapter:
    code: str
    description: str


@dataclass
class ICD10Code:
    code: str
    description: str
    description_aliases: list[str] = field(default_factory=list)
    category: Category | None = field(default=None)
    chapter: Chapter | None = field(default=None)


class KnowledgeBase:
    def __init__(self, file_path: Path = CODES_FILE_PATH) -> None:
        self.file_path: Path = file_path
        self.entries: list[ICD10Code] = self._construct_entries()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    @staticmethod
    def _entries_from_df(df: pl.DataFrame) -> list[ICD10Code]:
        if (
            "description_aliases" not in df.columns
        ):  # no aliases yet; mirror description
            df = df.with_columns(pl.col("description").alias("description_aliases"))
        return [
            ICD10Code(code, description, description_aliases, category, chapter)
            for code, description, description_aliases, category, chapter in zip(
                df["ICD10-CM-CODE"],
                df["description"],
                df["description_aliases"],
                df["category_code"],
                df["chapter"],
                strict=True,
            )
        ]

    def _construct_entries(self) -> list[ICD10Code]:
        df = pl.read_csv(self.file_path)
        return KnowledgeBase._entries_from_df(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise entries to a Parquet file for fast subsequent loads."""
        pl.DataFrame(
            {
                "ICD10-CM-CODE": [e.code for e in self.entries],
                "description": [e.description for e in self.entries],
                "description_aliases": [e.description_aliases for e in self.entries],
                "category_code": [e.category for e in self.entries],
                "chapter": [e.chapter for e in self.entries],
            }
        ).write_parquet(path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        """Load a KnowledgeBase from a Parquet file written by :meth:`save`."""
        obj = cls.__new__(cls)
        obj.file_path = path
        df = pl.read_parquet(path)
        obj.entries = cls._entries_from_df(df)
        return obj


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.entries[0])
