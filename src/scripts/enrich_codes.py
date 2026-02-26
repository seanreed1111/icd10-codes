from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def main():
    codes = pd.read_csv(RAW / "icd10cm-codes-April-1-2026.csv")
    categories = pd.read_csv(RAW / "icd10cm-categories-April-1-2026.csv")

    # Rename to avoid collision with codes 'description'
    categories = categories.rename(
        columns={
            "ICD10-CM-CODE": "category_code",
            "description": "category_description",
        }
    )

    # Derive join key from first 3 chars of full code
    codes["category_code"] = codes["ICD10-CM-CODE"].str[:3]

    enriched = codes.merge(categories, on="category_code", how="left")

    # Reorder columns
    enriched = enriched[
        [
            "ICD10-CM-CODE",
            "description",
            "category_code",
            "category_description",
            "section",
            "chapter",
        ]
    ]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(PROCESSED / "icd10cm-codes-enriched-April-1-2026.csv", index=False)
    print(
        f"Wrote {len(enriched):,} rows to {PROCESSED / 'icd10cm-codes-enriched-April-1-2026.csv'}"
    )


if __name__ == "__main__":
    main()
