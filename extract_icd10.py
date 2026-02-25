"""
ICD-10 Code CSV Dataset Extractor
Pulls ICD-10 codes and descriptions from:
  1. https://www.definitivehc.com/resources/healthcare-insights/top-snf-diagnoses
  2. C:/Users/sqr99/Downloads/ITSN21-optum-coding.pdf
  3. C:/Users/sqr99/Downloads/LTC-ICD-10-essentials.pdf
Outputs: data/icd10_codes.csv
"""

import csv
import os
import re

import pdfplumber
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEBSITE_URL = (
    "https://www.definitivehc.com/resources/healthcare-insights/top-snf-diagnoses"
)
OPTUM_PDF = "C:/Users/sqr99/Downloads/ITSN21-optum-coding.pdf"
LTC_PDF = "C:/Users/sqr99/Downloads/LTC-ICD-10-essentials.pdf"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "icd10_codes.csv")

# ICD-10 code: letter + 2–5 digits + optional decimal + 1–4 digits + optional letter
ICD10_RE = re.compile(r"^([A-Z][0-9]{2,5}(?:\.[0-9]{1,4})?[A-Z]?)$")

# Lines to discard from PDFs
SKIP_PATTERNS = [
    re.compile(r"RIC\s*Excl:", re.IGNORECASE),    # with or without space
    re.compile(r"^\d+$"),                          # bare page numbers
    re.compile(r"copyright", re.IGNORECASE),
    re.compile(r"^Page\s+\d+", re.IGNORECASE),
    re.compile(r"optum", re.IGNORECASE),
    re.compile(r"^ICD-?10", re.IGNORECASE),
    re.compile(r"^©"),
    re.compile(r"^\s*$"),                          # blank lines
]

# PDF encoding noise like %A, %B …
ARTIFACT_PATTERN = re.compile(r"%[A-Z]")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ICD10Scraper/1.0)"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def clean_description(text: str) -> str:
    """Strip PDF artifacts, collapse whitespace, strip edge punctuation."""
    text = ARTIFACT_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,;:-")
    return text


def normalize_code_for_comparison(code: str) -> str:
    """
    Normalise an ICD-10 code so website codes without decimals compare
    equal to PDF codes with decimals.
      G9341  -> G93.41
      I60.3  -> I60.3  (unchanged)
      G20    -> G20    (length ≤ 3, no change)
    """
    if "." in code or len(code) <= 3:
        return code
    # Insert decimal after position 3 (letter + 2 digits)
    return code[:3] + "." + code[3:]


def _should_skip(line: str) -> bool:
    """Return True if the line should be discarded."""
    for pat in SKIP_PATTERNS:
        if pat.search(line):
            return True
    return False


# ---------------------------------------------------------------------------
# State machine for parsing lines from a PDF column
# ---------------------------------------------------------------------------

def _parse_lines(lines: list[str]) -> list[dict]:
    """
    State machine: accumulate (code, description) pairs.

    Each line may be "CODE description..." or just continuation text.
    A single-letter section indicator (like 'c', 'd') before the code is skipped.
    %A artifacts are stripped before any check.
    """
    records: list[dict] = []
    current_code: str | None = None
    desc_parts: list[str] = []

    def _flush():
        if current_code:
            desc = clean_description(" ".join(desc_parts))
            if desc:
                records.append({"code": current_code, "description": desc})

    for raw_line in lines:
        # Strip encoding artifacts before any processing
        line = ARTIFACT_PATTERN.sub("", raw_line).strip()
        if not line:
            continue
        if _should_skip(line):
            continue

        tokens = line.split()
        if not tokens:
            continue

        # Skip single-letter section indicator (e.g. 'c', 'd', 'b') before code
        idx = 0
        if len(tokens[0]) == 1 and tokens[0].isalpha() and len(tokens) > 1:
            idx = 1

        first_tok = tokens[idx]
        rest = " ".join(tokens[idx + 1:])

        m = ICD10_RE.match(first_tok)
        if m:
            _flush()
            current_code = m.group(1)
            desc_parts = [rest] if rest else []
        else:
            if current_code is not None:
                desc_parts.append(line)

    _flush()
    return records


# ---------------------------------------------------------------------------
# Source 1: website scraper
# ---------------------------------------------------------------------------

def scrape_website(url: str) -> list[dict]:
    """
    Scrape ICD-10 codes from the Definitive HC SNF diagnoses table.
    Table columns: [0]=Rank, [1]=ICD-10 code, [2]=Description
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="dhs-table")
    if table is None:
        # Fallback: try the first table on the page
        table = soup.find("table")
    if table is None:
        raise ValueError("No table found on website page")

    records: list[dict] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        code = cells[1].get_text(strip=True)
        description = cells[2].get_text(strip=True)
        if ICD10_RE.match(code):
            records.append(
                {
                    "icd10_code": code,
                    "description": clean_description(description),
                    "source": "website",
                }
            )
    return records


# ---------------------------------------------------------------------------
# Source 2 & 3: PDF extractor
# ---------------------------------------------------------------------------

def _words_to_lines(words: list[dict], col_tolerance: float = 3.0) -> list[str]:
    """
    Group pdfplumber word dicts into text lines by proximity in y-coordinate.
    Words are sorted by x within each line.
    Returns list of joined line strings.
    """
    if not words:
        return []

    # Sort by vertical position first
    words_sorted = sorted(words, key=lambda w: (round(w["top"] / col_tolerance), w["x0"]))

    lines: list[list[dict]] = []
    current_group: list[dict] = [words_sorted[0]]
    current_top = words_sorted[0]["top"]

    for word in words_sorted[1:]:
        if abs(word["top"] - current_top) <= col_tolerance:
            current_group.append(word)
        else:
            lines.append(sorted(current_group, key=lambda w: w["x0"]))
            current_group = [word]
            current_top = word["top"]
    lines.append(sorted(current_group, key=lambda w: w["x0"]))

    return [" ".join(w["text"] for w in group) for group in lines]


def extract_from_pdf(pdf_path: str, source_label: str) -> list[dict]:
    """
    Extract ICD-10 codes + descriptions from a multi-column PDF.
    Strategy: split each page at the horizontal midpoint into left/right
    columns, then group words into lines and run the state machine.
    """
    all_records: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            mid = page.width / 2
            words = page.extract_words()
            if not words:
                continue

            left_words = [w for w in words if w["x0"] < mid]
            right_words = [w for w in words if w["x0"] >= mid]

            for col_words in (left_words, right_words):
                lines = _words_to_lines(col_words)
                col_records = _parse_lines(lines)
                all_records.extend(col_records)

    # Attach source label
    return [
        {
            "icd10_code": r["code"],
            "description": r["description"],
            "source": source_label,
        }
        for r in all_records
    ]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    """
    Deduplicate by normalised ICD-10 code, keeping the first occurrence.
    Source priority (by insertion order): website → optum → ltc.
    """
    seen: dict[str, bool] = {}
    unique: list[dict] = []
    for rec in records:
        key = normalize_code_for_comparison(rec["icd10_code"])
        if key not in seen:
            seen[key] = True
            unique.append(rec)
    return unique


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["icd10_code", "description", "source"])
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_records: list[dict] = []

    # --- Website ---
    try:
        website_records = scrape_website(WEBSITE_URL)
        print(f"[website] Extracted {len(website_records)} codes")
        all_records.extend(website_records)
    except Exception as exc:
        print(f"[website] ERROR: {exc}")

    # --- Optum PDF ---
    try:
        optum_records = extract_from_pdf(OPTUM_PDF, "optum")
        print(f"[optum]   Extracted {len(optum_records)} codes")
        all_records.extend(optum_records)
    except FileNotFoundError:
        print(f"[optum]   ERROR: PDF not found at {OPTUM_PDF}")
    except Exception as exc:
        print(f"[optum]   ERROR: {exc}")

    # --- LTC PDF ---
    try:
        ltc_records = extract_from_pdf(LTC_PDF, "ltc")
        print(f"[ltc]     Extracted {len(ltc_records)} codes")
        all_records.extend(ltc_records)
    except FileNotFoundError:
        print(f"[ltc]     ERROR: PDF not found at {LTC_PDF}")
    except Exception as exc:
        print(f"[ltc]     ERROR: {exc}")

    if not all_records:
        print("No records extracted. Exiting without writing CSV.")
        return

    # --- Deduplicate ---
    deduped = deduplicate(all_records)
    print(f"\nTotal after deduplication: {len(deduped)} unique codes")

    # --- Write CSV ---
    write_csv(deduped, OUTPUT_CSV)
    print(f"CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
