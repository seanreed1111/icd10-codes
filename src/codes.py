from dataclasses import dataclass
from dataclasses import field
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
    def __init__(self, file_path: Path = CODES_FILE_PATH):
        self.file_path: Path = file_path
        self.entries: list[ICD10Code] = self._construct_entries()
    
    def _construct_entries(self):
        df = pl.read_csv(self.file_path)
        if "description_aliases" not in df.columns:
            df = df.with_columns(pl.col("description").alias("description_aliases")) #no description aliases exist yet
        return [ICD10Code(code, description, description_aliases, category, chapter) for code, description, description_aliases, category, chapter in zip(df["ICD10-CM-CODE"], df["description"], df["description_aliases"], df["category_code"], df["chapter"])]




if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.entries[0])


