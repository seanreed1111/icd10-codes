"""Multi-label ICD-10 retrieval pipeline (4–7 char codes only).

Uses a sliding-window chunking approach (Option E from v2) to handle
long clinical notes. Three-character category codes are discarded
throughout — see IGNORE_THREE_CHARACTER_CODE flag on each function.

Note on naming convention: ``IGNORE_THREE_CHARACTER_CODE`` parameter
names are intentionally ALL_CAPS to signal a global configuration flag
rather than a regular function argument. This is a deliberate deviation
from PEP 8 to make the flag's special status visually distinct.

Run from project root:
    uv run python src/pipeline_v5.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))  # makes bare imports work

from knowledge_base import KnowledgeBase
from retriever import TfidfRetriever

if TYPE_CHECKING:
    from types import NotImplementedType

# ===================================================================
# Project paths (relative to project root)
# ===================================================================

PROJECT_ROOT = Path(__file__).parent.parent
KB_PATH = (
    PROJECT_ROOT / "data" / "processed" / "icd10cm-codes-enriched-April-1-2026.csv"
)
GT_PATH = (
    PROJECT_ROOT
    / "data"
    / "test-datasets"
    / "codiesp"
    / "data-pipeline"
    / "processed"
    / "gold"
    / "codiesp_ground_truth.parquet"
)
TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "test-datasets"
    / "codiesp"
    / "gold"
    / "final_dataset_v4_to_publish"
    / "dev"
    / "text_files_en"
)


# ===================================================================
# Step 0: KB expansion — add category-only codes
# ===================================================================


def expand_kb_with_categories(
    kb: KnowledgeBase,
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> KnowledgeBase | NotImplementedType:
    """Add category-level entries to the KnowledgeBase.

    Parameters
    ----------
    kb : KnowledgeBase
        Original KB (74,719 billable code entries).
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), 3-char category codes are discarded and the
        original KB is returned unchanged.

    Returns
    -------
    KnowledgeBase
        Original KB unchanged (3-char codes are discarded when flag is True).
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    # 3-char category codes are discarded; return the original KB as-is
    return kb


def get_category_only_codes(
    kb: KnowledgeBase,
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> set[str] | NotImplementedType:
    """Return 3-char codes that are ICD-10-CM categories but NOT billable.

    Parameters
    ----------
    kb : KnowledgeBase
        Knowledge base (used to identify billable codes).
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), all category-only codes are 3-char and are
        therefore discarded; returns an empty set.

    Returns
    -------
    set[str]
        Empty set (all category-only codes are 3-char, so all are discarded).
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    # All category-only codes are 3-char; discard them all
    return set()


# ===================================================================
# Pipeline step 1: Load text from disk
# ===================================================================


def load_notes(stems: list[str], text_dir: Path = TEXT_DIR) -> list[str]:
    """Load clinical note text files from disk.

    Parameters
    ----------
    stems : list[str]
        File stems (e.g., 'S0004-06142005000900016-1').
    text_dir : Path
        Directory containing <stem>.txt files.

    Returns
    -------
    list[str]
        UTF-8 text contents, one per stem.
    """
    return [
        (text_dir / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")
        for stem in stems
    ]


# ===================================================================
# Pipeline step 2: Normalize text
# ===================================================================


def clean_texts(texts: list[str], *, lowercase: bool = True) -> list[str]:
    """Collapse whitespace and optionally lowercase."""
    result: list[str] = []
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        if lowercase:
            text = text.lower()
        result.append(text)
    return result


# ===================================================================
# Pipeline step 3: Sliding-window chunker
# ===================================================================


def chunk_texts(
    texts: list[str],
    *,
    window: int = 3,
    stride: int = 2,
    min_words: int = 3,
    include_full_text: bool = True,
) -> list[list[str]]:
    """Split each text into overlapping sentence-window chunks.

    Parameters
    ----------
    texts : list[str]
        One text per document.
    window : int
        Number of sentences per chunk.
    stride : int
        Step between chunk starts.  stride < window means overlap.
    min_words : int
        Minimum tokens for a sentence to be included.
    include_full_text : bool
        Append the full document as an extra chunk (preserves global context).

    Returns
    -------
    list[list[str]]
        One list-of-chunks per document.
    """
    result: list[list[str]] = []
    for text in texts:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.split()) >= min_words
        ]
        chunks: list[str] = []
        for i in range(0, max(1, len(sentences) - window + 1), stride):
            chunk = " ".join(sentences[i : i + window])
            if chunk:
                chunks.append(chunk)
        if include_full_text:
            chunks.append(text)
        result.append(chunks)
    return result


# ===================================================================
# Pipeline step 4: Chunked retrieval predictor
# ===================================================================


def retrieve_codes(
    chunks_per_doc: list[list[str]],
    retriever: TfidfRetriever,
    *,
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
    final_threshold: float = 0.04,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> list[set[str]] | NotImplementedType:
    """Multi-label prediction: TF-IDF search per chunk, aggregate, threshold.

    For each document's chunks, retrieves top_k codes per chunk,
    aggregates scores across chunks, filters by votes + threshold.

    Parameters
    ----------
    chunks_per_doc : list[list[str]]
        One list-of-chunks per document.
    retriever : TfidfRetriever
        Pre-built TF-IDF retriever (built once over the KB).
    top_k_per_chunk : int
        How many candidate codes to retrieve per chunk.
    min_chunk_score : float
        Per-chunk cosine score floor.
    min_votes : int
        A code must appear in at least this many chunks to survive.
    aggregation : str
        "max", "mean", or "sum".
    final_threshold : float
        Aggregated score must exceed this to appear in the prediction.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), any 3-char codes in retrieval results are
        discarded before returning.

    Returns
    -------
    list[set[str]]
        One set of predicted ICD-10 codes per document (4–7 char only).
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    predictions: list[set[str]] = []
    for chunks in chunks_per_doc:
        code_scores: dict[str, list[float]] = defaultdict(list)
        for chunk in chunks:
            hits = retriever.search(chunk, top_k=top_k_per_chunk)
            for h in hits:
                if h.score >= min_chunk_score:
                    code_scores[h.code].append(h.score)

        filtered = {
            code: scores
            for code, scores in code_scores.items()
            if len(scores) >= min_votes
        }

        aggregated: dict[str, float] = {}
        for code, scores in filtered.items():
            if aggregation == "max":
                aggregated[code] = max(scores)
            elif aggregation == "mean":
                aggregated[code] = sum(scores) / len(scores)
            else:  # "sum"
                aggregated[code] = sum(scores)

        predictions.append(
            {
                code
                for code, score in aggregated.items()
                if score >= final_threshold and len(code) > 3  # discard 3-char codes
            }
        )
    return predictions


# ===================================================================
# Pipeline step 5: Category roll-up (Strategy 2b)
# ===================================================================


def rollup_categories(
    predictions: list[set[str]],
    category_only_codes: set[str],
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> list[set[str]] | NotImplementedType:
    """Post-prediction: drop any 3-char codes from predictions.

    With IGNORE_THREE_CHARACTER_CODE=True (default), 3-char codes are
    discarded from the prediction sets and no new 3-char category codes
    are added. The category_only_codes argument is accepted for API
    compatibility but is not used when the flag is True.

    Parameters
    ----------
    predictions : list[set[str]]
        One set of predicted codes per document.
    category_only_codes : set[str]
        Unused when IGNORE_THREE_CHARACTER_CODE=True.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), drops any 3-char codes from predictions
        and does not add new category codes.

    Returns
    -------
    list[set[str]]
        Predictions with all 3-char codes removed.
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    # Drop any 3-char codes already in the prediction sets; don't add new ones
    return [{code for code in pred_set if len(code) > 3} for pred_set in predictions]


# ===================================================================
# Evaluation helpers
# ===================================================================


def precision_recall_f1(
    predicted: set[str],
    gold: set[str],
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> dict[str, float] | NotImplementedType:
    """Per-document precision, recall, F1.

    Parameters
    ----------
    predicted : set[str]
        Predicted ICD-10 codes for one document.
    gold : set[str]
        Ground-truth ICD-10 codes for one document.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), any 3-char codes in predicted or gold are
        discarded before computing metrics so they cannot inflate or
        deflate performance.
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    # Discard 3-char codes from both sides before scoring
    predicted = {c for c in predicted if len(c) > 3}
    gold = {c for c in gold if len(c) > 3}
    # Both empty after filtering → vacuous truth: perfect score (intentional;
    # no codes on either side means there is nothing to penalize).
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_average(
    pairs: list[tuple[set[str], set[str]]],
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> dict[str, float] | NotImplementedType:
    """Macro-average P/R/F1 over (predicted, gold) pairs.

    Parameters
    ----------
    pairs : list[tuple[set[str], set[str]]]
        List of (predicted, gold) set pairs, one per document.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), passed through to precision_recall_f1 so
        3-char codes are stripped from every pair before scoring.

    Returns
    -------
    dict[str, float]
        Macro-averaged metrics with keys: ``precision``, ``recall``, ``f1``.
        All values in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    agg: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, gold in pairs:
        m = precision_recall_f1(
            pred, gold, IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE
        )
        for k in agg:
            agg[k] += m[k]
    n = len(pairs)
    return {k: round(v / n, 4) for k, v in agg.items()}


def load_gold_map(
    gt_path: Path = GT_PATH,
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> dict[str, set[str]] | NotImplementedType:
    """Load ground truth -> {file_stem: set_of_unique_codes}.

    Parameters
    ----------
    gt_path : Path
        Path to the ground truth parquet file.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), any 3-char codes in the ground truth are
        discarded before returning.

    Returns
    -------
    dict[str, set[str]]
        Mapping from file stem to set of codes (4–7 char only).
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    df = pl.read_parquet(gt_path)
    return {
        str(row["file_stem"]): {
            c
            for c in str(row["codes"]).split(";")
            if len(c) > 3  # discard 3-char codes
        }
        for row in df.iter_rows(named=True)
    }


# ===================================================================
# Pipeline config and runner
# ===================================================================


@dataclass
class PipelineConfig:
    """All tunable parameters for the v5 pipeline."""

    retriever: TfidfRetriever
    text_dir: Path = TEXT_DIR
    # Chunker params
    window: int = 3
    stride: int = 2
    # Retrieval params
    top_k_per_chunk: int = 10
    min_chunk_score: float = 0.04
    min_votes: int = 1
    aggregation: str = "max"
    final_threshold: float = 0.04
    # Category handling (unused when IGNORE_THREE_CHARACTER_CODE=True)
    category_only_codes: set[str] | None = None


def run_pipeline(
    cfg: PipelineConfig,
    stems: list[str],
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> list[set[str]] | NotImplementedType:
    """Execute the full pipeline in a single pass.

    Steps:
        1. Load clinical note texts from disk
        2. Clean/normalize texts
        3. Chunk texts into sliding windows
        4. Retrieve candidate ICD-10 codes per chunk (3-char dropped)
        5. Optionally roll up to category codes (no-op with flag=True)

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration (retriever, thresholds, etc.).
    stems : list[str]
        File stems identifying the clinical notes.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), 3-char codes are discarded at each step.

    Returns
    -------
    list[set[str]]
        One set of predicted ICD-10 codes per document (4–7 char only).
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    texts = load_notes(stems, cfg.text_dir)
    cleaned = clean_texts(texts)
    chunks = chunk_texts(cleaned, window=cfg.window, stride=cfg.stride)
    predictions = retrieve_codes(
        chunks,
        cfg.retriever,
        top_k_per_chunk=cfg.top_k_per_chunk,
        min_chunk_score=cfg.min_chunk_score,
        min_votes=cfg.min_votes,
        aggregation=cfg.aggregation,
        final_threshold=cfg.final_threshold,
        IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE,
    )
    if cfg.category_only_codes is not None:
        predictions = rollup_categories(
            predictions,
            cfg.category_only_codes,
            IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE,
        )
    return predictions


def build_pipeline(
    kb: KnowledgeBase,
    text_dir: Path = TEXT_DIR,
    *,
    window: int = 3,
    stride: int = 2,
    top_k_per_chunk: int = 10,
    min_chunk_score: float = 0.04,
    min_votes: int = 1,
    aggregation: str = "max",
    final_threshold: float = 0.04,
    expand_kb: bool = False,
    enable_rollup: bool = False,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> PipelineConfig | NotImplementedType:
    """Build the full end-to-end v5 pipeline config.

    Constructs a TfidfRetriever from ``kb`` and returns a PipelineConfig.
    Use this convenience factory when you need a single self-contained
    config. For parameter sweeps where the KB is constant, prefer
    constructing PipelineConfig directly with a shared TfidfRetriever to
    avoid rebuilding the TF-IDF index on each call.

    Input:  list[str]       file stems (passed to run_pipeline)
    Output: list[set[str]]  predicted ICD-10 code sets (4–7 char only)

    Parameters
    ----------
    kb : KnowledgeBase
        Original KB.
    expand_kb : bool
        If True, calls expand_kb_with_categories (no-op when flag=True).
    enable_rollup : bool
        If True, applies post-prediction rollup (no-op when flag=True).
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), passed through to all sub-functions; 3-char
        codes are discarded at every step.
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    working_kb = kb
    if expand_kb:
        working_kb = expand_kb_with_categories(
            kb, IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE
        )

    # Build TF-IDF index once (TfidfRetriever builds in __init__)
    retriever = TfidfRetriever(working_kb)

    cat_only = (
        get_category_only_codes(
            kb, IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE
        )
        if enable_rollup
        else None
    )

    return PipelineConfig(
        retriever=retriever,
        text_dir=text_dir,
        window=window,
        stride=stride,
        top_k_per_chunk=top_k_per_chunk,
        min_chunk_score=min_chunk_score,
        min_votes=min_votes,
        aggregation=aggregation,
        final_threshold=final_threshold,
        category_only_codes=cat_only,
    )


# ===================================================================
# End-to-end evaluation
# ===================================================================


def evaluate(
    cfg: PipelineConfig,
    stems: list[str],
    gold_map: dict[str, set[str]],
    label: str = "",
    *,
    IGNORE_THREE_CHARACTER_CODE: bool = True,
) -> dict[str, float] | NotImplementedType:
    """Run pipeline over ``stems`` and compute macro-averaged P/R/F1.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully constructed pipeline configuration (from ``build_pipeline``
        or constructed directly).
    stems : list[str]
        File stems to evaluate; must all be keys in ``gold_map``.
    gold_map : dict[str, set[str]]
        Mapping from file stem to ground-truth code set (4–7 char codes only,
        as returned by ``load_gold_map``).
    label : str
        Optional display label printed next to the metrics.  Pass ``""``
        to suppress output.
    IGNORE_THREE_CHARACTER_CODE : bool
        If False, returns NotImplemented immediately (not implemented).
        If True (default), passed through to ``run_pipeline`` and
        ``macro_average``; 3-char codes are stripped at every layer.

    Returns
    -------
    dict[str, float]
        Macro-averaged metrics with keys: ``precision``, ``recall``, ``f1``.
        All values in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not IGNORE_THREE_CHARACTER_CODE:
        return NotImplemented
    predictions = run_pipeline(
        cfg, stems, IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE
    )
    pairs = [
        (pred, gold_map[stem]) for pred, stem in zip(predictions, stems, strict=True)
    ]
    metrics = macro_average(
        pairs, IGNORE_THREE_CHARACTER_CODE=IGNORE_THREE_CHARACTER_CODE
    )
    if label:
        print(
            f"{label:<40}  "
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )
    return metrics


# ===================================================================
# Results collection and markdown report
# ===================================================================


@dataclass
class EvalResult:
    """One row of evaluation output (baseline or sweep config)."""

    label: str
    is_baseline: bool
    window: int
    top_k: int
    threshold: float
    precision: float
    recall: float
    f1: float


def save_results_markdown(
    results: list[EvalResult],
    n_docs: int,
    kb_size: int,
    output_path: Path,
) -> None:
    """Write all evaluation results to a formatted markdown file.

    Parameters
    ----------
    results : list[EvalResult]
        All evaluation rows (baseline first, then sweep).
    n_docs : int
        Number of documents evaluated.
    kb_size : int
        Number of entries in the knowledge base.
    output_path : Path
        Destination file path (parent directory must exist or will be created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baseline_rows = [r for r in results if r.is_baseline]
    sweep_rows = sorted(
        [r for r in results if not r.is_baseline], key=lambda r: r.f1, reverse=True
    )
    best = (
        sweep_rows[0] if sweep_rows else (baseline_rows[0] if baseline_rows else None)
    )

    lines: list[str] = []

    lines += [
        "# Multi-Label ICD-10 Retrieval v5 — Evaluation Results",
        "",
        f"**Run date:** {run_ts}  ",
        f"**Documents evaluated:** {n_docs}  ",
        f"**KB size:** {kb_size:,}  ",
        "",
    ]

    # --- Baseline ---
    lines += [
        "## Baseline",
        "",
        "| Config | Window | top_k | Threshold | Precision | Recall | F1 |",
        "|--------|--------|-------|-----------|-----------|--------|----|",
    ]
    for r in baseline_rows:
        lines.append(
            f"| {r.label} | {r.window} | {r.top_k} | {r.threshold:.2f}"
            f" | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |"
        )
    lines.append("")

    # --- Parameter sweep (sorted by F1 desc) ---
    lines += [
        "## Parameter Sweep",
        "",
        "_Sorted by F1 descending._",
        "",
        "| Window | top_k | Threshold | Precision | Recall | F1 |",
        "|--------|-------|-----------|-----------|--------|----|",
    ]
    for r in sweep_rows:
        lines.append(
            f"| {r.window} | {r.top_k} | {r.threshold:.2f}"
            f" | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |"
        )
    lines.append("")

    # --- Best config ---
    if best is not None:
        lines += [
            "## Best Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| window | {best.window} |",
            f"| top_k | {best.top_k} |",
            f"| threshold | {best.threshold:.2f} |",
            f"| **Precision** | **{best.precision:.4f}** |",
            f"| **Recall** | **{best.recall:.4f}** |",
            f"| **F1** | **{best.f1:.4f}** |",
            "",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved -> {output_path}")


# ===================================================================
# Main: baseline + parameter sweep
# ===================================================================

if __name__ == "__main__":
    import itertools

    kb_original = KnowledgeBase(KB_PATH)
    gold_map = load_gold_map(IGNORE_THREE_CHARACTER_CODE=True)
    stems = sorted(gold_map.keys())

    print("=" * 72)
    print("Multi-Label ICD-10 Retrieval - v5 Evaluation")
    print("=" * 72)
    print(f"Documents: {len(stems)}")
    print(f"KB size: {len(kb_original.entries):,}")
    print()

    # Build TF-IDF index once — reused for baseline and all sweep configs.
    # All configs use the same KB (no expand_kb), so rebuilding per config
    # would be wasteful (28× slower).
    print("Building TF-IDF index...")
    shared_retriever = TfidfRetriever(kb_original)

    all_results: list[EvalResult] = []

    # --- Baseline ---
    print()
    print("--- Baseline ---")
    cfg_baseline = PipelineConfig(retriever=shared_retriever)
    m_baseline = evaluate(
        cfg_baseline,
        stems,
        gold_map,
        label="Baseline",
        IGNORE_THREE_CHARACTER_CODE=True,
    )
    all_results.append(
        EvalResult(
            label="Baseline",
            is_baseline=True,
            window=cfg_baseline.window,
            top_k=cfg_baseline.top_k_per_chunk,
            threshold=cfg_baseline.final_threshold,
            precision=m_baseline["precision"],
            recall=m_baseline["recall"],
            f1=m_baseline["f1"],
        )
    )

    # --- Parameter sweep ---
    print()
    print("--- Parameter Sweep ---")

    for window, top_k, threshold in itertools.product(
        [2, 3, 5],
        [5, 10, 20],
        [0.03, 0.05, 0.08],
    ):
        cfg = PipelineConfig(
            retriever=shared_retriever,
            window=window,
            top_k_per_chunk=top_k,
            final_threshold=threshold,
        )
        m = evaluate(
            cfg,
            stems,
            gold_map,
            label=f"  w={window} k={top_k} t={threshold:.2f}",
            IGNORE_THREE_CHARACTER_CODE=True,
        )
        all_results.append(
            EvalResult(
                label=f"w={window} k={top_k} t={threshold:.2f}",
                is_baseline=False,
                window=window,
                top_k=top_k,
                threshold=threshold,
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
            )
        )

    best = max((r for r in all_results if not r.is_baseline), key=lambda r: r.f1)
    print()
    print(
        f"Best: w={best.window} k={best.top_k} t={best.threshold:.2f}"
        f"  P={best.precision:.4f}  R={best.recall:.4f}  F1={best.f1:.4f}"
    )

    # Save markdown report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PROJECT_ROOT / "results" / f"pipeline_v5_{ts}.md"
    save_results_markdown(
        all_results,
        n_docs=len(stems),
        kb_size=len(kb_original.entries),
        output_path=report_path,
    )
