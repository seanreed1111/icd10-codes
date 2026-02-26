import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

SRC = Path("data/COT-SYMP-ICD10-2024/train/data-00000-of-00001.arrow")
DEST = Path("data/COT-SYMP-ICD10-2024/train/data.json")


def extract_code(answer: str) -> str:
    """Extract the ICD-10 code after the #### delimiter, removing the decimal point."""
    raw = answer.split("####")[-1].strip()
    return raw.replace(".", "")


def main() -> None:
    with pa.memory_map(str(SRC), "r") as source:
        table = ipc.open_stream(source).read_all()

    records = []
    for batch in table.to_batches():
        questions = batch.column("question").to_pylist()
        answers = batch.column("answer").to_pylist()
        for question, answer in zip(questions, answers, strict=True):
            records.append(
                {
                    "question": question,
                    "answer": answer,
                    "code": extract_code(answer),
                }
            )

    DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {DEST}")


if __name__ == "__main__":
    main()
