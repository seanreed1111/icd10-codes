# Plan Review: TF-IDF Demo Scripts Implementation Plan

**Review Date:** 2026-02-25
**Reviewer:** Claude Code Review Agent
**Plan Location:** `plan/future-plans/2026-02-25-tfidf-demo-scripts.md`

---

## Executive Summary

**Executability Score: 91/100 - Excellent**

**Overall Assessment:**

This is a well-structured, highly detailed implementation plan. Its greatest strength is that it provides the complete source code for every file to be created, eliminating the primary source of ambiguity in most plans. The dataset is faithfully extracted from the guide, all script code follows consistent patterns, and the dependency ordering is correct. An executing agent can follow this plan with minimal interpretation.

The few issues identified are minor: a small inconsistency in the file inventory table, a missing explicit step to create the `docs/tfidf/demo/` directory, the lack of an explicit step to verify that required Python packages are installed, and the "NOTE FOR REVIEWER" footer that references summarized code sections that don't actually exist in the plan (since the full code IS present). None of these are blockers.

**Recommendation:**
- [x] Ready for execution
- [ ] Ready with minor clarifications
- [ ] Requires improvements before execution
- [ ] Requires major revisions

---

## Detailed Analysis

### 1. Accuracy (19/20)

| Sub-criterion | Score | Max | Notes |
|---|---|---|---|
| Technical correctness | 5 | 5 | All scikit-learn API usage is correct. `TfidfVectorizer`, `CountVectorizer`, `TfidfTransformer`, `linear_kernel`, `hstack`, and `pickle` are used correctly. The assertion that `linear_kernel` equals `cosine_similarity` for L2-normalized vectors is accurate. |
| File path validity | 5 | 5 | All paths are relative to project root and consistent. `Path(__file__).parent / "dataset.json"` is the correct approach for self-contained scripts. |
| Codebase understanding | 4 | 5 | The plan correctly identifies `uv` as the package manager and existing project structure. However, the "Current State Analysis" says `docs/tfidf/demo/` does not exist yet, but it actually already does exist (as an empty directory). This is a minor inaccuracy — it doesn't affect execution since the scripts just need to be placed there. |
| Dependency accuracy | 5 | 5 | All four required packages (`scikit-learn`, `scipy`, `numpy`, `pandas`) are already in `pyproject.toml`. The plan says "add if missing" which is appropriately cautious. |

**Deduction (-1):** The plan states `docs/tfidf/demo/` "does not exist yet" (line 25), but it already exists as an empty directory. Minor factual inaccuracy.

---

### 2. Consistency (14/15)

| Sub-criterion | Score | Max | Notes |
|---|---|---|---|
| Internal consistency | 4 | 5 | See minor issue below regarding file inventory table. |
| Naming conventions | 5 | 5 | `stepNN_descriptive_name.py` naming is consistent throughout. `load_dataset()` function signature `tuple[list[str], list[str]]` is identical in all scripts. |
| Pattern adherence | 5 | 5 | All scripts follow the same structure: docstring, imports, `load_dataset()`, `main()`, `if __name__ == "__main__"` guard. This matches the project's existing code style. |

**Deduction (-1):** In the File Inventory table (line 74), `step03_fit_transform.py` has the purpose "Compare TfidfVectorizer vs CountVectorizer+TfidfTransformer" which is copy-pasted from step02. It should say "Demonstrate fit, transform, fit_transform" or similar. The actual code and the "What this does" section are correct — only the inventory table cell is wrong.

---

### 3. Clarity (23/25)

| Sub-criterion | Score | Max | Notes |
|---|---|---|---|
| Instruction clarity | 7 | 7 | Instructions are extremely clear. Full source code is provided for every file, removing interpretation. Each section has "What this does" summaries. |
| Success criteria clarity | 6 | 7 | Success criteria are concrete and include bash commands for verification. However, Phase 1's automated verification only checks JSON validity and item count, not the actual content of each item against the guide. |
| Minimal ambiguity | 10 | 11 | The plan is highly explicit; minimal ambiguity. Minor gap: the verification commands don't confirm output correctness, only error-free execution. |

**Deduction (-2):** Success criteria for Phases 2 and 3 only say "run without error." A more robust criterion would check for expected output patterns (e.g., step02 should print "Identical (within float tolerance): True"). Also, Phase 1's second checkbox ("Contains 10 items, each with `description` and `code` keys") has no corresponding automated command.

---

### 4. Completeness (21/25)

| Sub-criterion | Score | Max | Notes |
|---|---|---|---|
| All steps present | 9 | 11 | Two steps are missing (detailed below). |
| Context adequate | 6 | 6 | Context sections reference exact guide line numbers and clearly describe what to read before each phase. |
| Edge cases covered | 4 | 6 | The code handles empty search results (`if scores[idx] > 0`) and demonstrates failure modes (step03 "BAD" example, step08 no-match queries). However, the plan doesn't address what happens if `uv sync` hasn't been run, or if the scripts are run from a different working directory. |
| Testing comprehensive | 2 | 2 | Testing strategy is appropriate for demo scripts: run-all-scripts loop plus manual verification checklist. Correctly decided against pytest for self-demonstrating scripts. |

**Deductions (-4):**

1. **Missing step: Create directory** (-1). The plan assumes `docs/tfidf/demo/` exists. It happens to exist in the current repo, but the plan doesn't mention creating it. If this directory were missing (or if the plan is re-executed in a clean checkout), Phase 1 would fail. The plan should include `mkdir -p docs/tfidf/demo/` or note that it already exists.

2. **Missing step: Verify dependencies are installed** (-1). The plan says "add if missing" for scikit-learn, scipy, numpy, pandas, but never provides a concrete command (`uv sync` or `uv add ...`). While these packages are already in `pyproject.toml`, the plan should include an explicit `uv sync` step before running any scripts.

3. **No handling of pre-commit hooks** (-1). The project has ruff pre-commit hooks. If an executing agent commits these files, the ruff hooks will run. The plan's code appears to be ruff-compliant, but this isn't mentioned or verified.

4. **step05 implicit assumption about aligned feature names** (-1). The `sublinear_tf` comparison section compares feature-by-feature values across two independently-fitted vectorizers (`v_linear` and `v_sublin`). This is fine since both use the same data and default settings, producing the same vocabulary, but the plan doesn't call this out as a requirement. If someone modified the dataset, this could silently break.

---

### 5. Executability (18/20)

| Sub-criterion | Score | Max | Notes |
|---|---|---|---|
| Agent-executable | 7 | 8 | Full code is provided inline, making this highly agent-executable. The one gap is the missing directory creation and dependency sync steps. |
| Dependencies ordered | 6 | 6 | Phase ordering is correct. Phases 2 and 3 depend only on Phase 1, and are independent of each other. No circular dependencies. |
| Success criteria verifiable | 5 | 6 | All verification is automated via bash commands. The for-loop at the end catches runtime errors. However, the criteria don't verify correctness of output, only absence of errors. |

**Deduction (-2):** An agent following the plan literally might not run `uv sync` first, causing import errors. Also, the verification commands run from the project root, which is correct, but this isn't explicitly stated as a precondition for the verification step.

---

## Identified Pain Points

### Critical Blockers

None. The plan is executable as-is given the current state of the repository.

### Major Concerns

1. **File Inventory table error (line 74):** `step03_fit_transform.py` has the wrong description ("Compare TfidfVectorizer vs CountVectorizer+TfidfTransformer" instead of describing fit/transform mechanics). An agent reading the inventory for context before implementation could be misled, though the actual code section (2.2) is correct.

### Minor Issues

1. **Directory creation not mentioned:** The plan should explicitly state that `docs/tfidf/demo/` needs to be created (or verified to exist) before writing files into it.

2. **No `uv sync` step:** The plan should include a prerequisite step to ensure dependencies are installed.

3. **"Current State Analysis" says directory doesn't exist (line 25):** The `docs/tfidf/demo/` directory already exists in the repo.

4. **Verification only checks for errors, not correctness:** The success criteria check that scripts run without error but don't validate output content (e.g., step02 should print "Identical (within float tolerance): True").

---

## Specific Recommendations

### High Priority

1. **Add a Phase 0 or prerequisites section** that includes:
   - Verify `docs/tfidf/demo/` directory exists, or create it: `mkdir -p docs/tfidf/demo/`
   - Run `uv sync` to ensure all dependencies are installed
   - Note: run all commands from the project root

2. **Fix the File Inventory table (line 74):** Change `step03_fit_transform.py` purpose from "Compare TfidfVectorizer vs CountVectorizer+TfidfTransformer" to "Demonstrate fit, transform, fit_transform".

### Medium Priority

3. **Add output-content verification to success criteria.** For example:
   ```bash
   uv run python docs/tfidf/demo/step02_vectorizer_vs_transformer.py 2>&1 | grep -q "Identical (within float tolerance): True"
   ```

4. **Update "Current State Analysis" line 25** to reflect that `docs/tfidf/demo/` already exists as an empty directory.

### Low Priority

5. **Add a note about ruff compliance** to reassure executing agents that the provided code is expected to pass pre-commit hooks without modification.

---

## Phase-by-Phase Analysis

### Phase 1: Dataset and Shared Loader

**Rating: Strong**

The dataset is an exact match to the guide's mock data (lines 29–48 of `tfidf-guide.md`). All 10 items are present with correct descriptions and codes. The JSON structure is clean and valid. Automated verification is provided.

**Gap:** The second success criterion checkbox ("Contains 10 items, each with `description` and `code` keys") has no corresponding automated verification command. Consider adding:
```bash
uv run python -c "import json, pathlib; d=json.loads(pathlib.Path('docs/tfidf/demo/dataset.json').read_text()); assert len(d)==10; assert all('description' in x and 'code' in x for x in d), 'missing keys'"
```

### Phase 2: Core TF-IDF Demos (Steps 2–5)

**Rating: Strong**

All four scripts faithfully implement the guide's examples. Notable strengths:
- step02 correctly verifies matrix equivalence with `np.abs().max() < 1e-10`
- step03 demonstrates the critical "don't fit_transform on queries" anti-pattern with a clear "BAD" label
- step04 uses pandas for readable output (already a project dependency)
- step05 covers all 6 parameters with before/after comparisons, including custom medical stop words

**Dependencies:** Correctly depends on Phase 1; no inter-dependencies within Phase 2.

**Success Criteria:** Adequate (scripts run without error), but output content is not verified.

### Phase 3: Search and Advanced Demos (Steps 6–9)

**Rating: Strong**

- step06 includes the "heart attack" synonym failure case as a teaching demonstration
- step07 correctly uses `hstack` from `scipy.sparse` (not numpy) to combine sparse matrices
- step08's `explain_query()` function includes the proper type hint and covers both match and no-match cases
- step09 uses `tempfile.TemporaryDirectory()` as a context manager with auto-cleanup — correct design for a demo

**Dependencies:** Correctly depends on Phase 1 only; no inter-dependencies within Phase 3.

---

## Testing Strategy Assessment

**Coverage: Good**

The testing strategy is appropriate for the scope of this plan:

- **Automated:** A single bash for-loop runs all scripts and surfaces any runtime errors. Correct level of automation for demo scripts.
- **Manual checklist:** Five specific items to visually verify, including the synonym limitation demo and typo-robustness demo.

**Gaps:**
- No automated check for output correctness. If step02 prints "Identical: False" due to a floating-point edge case, the automated test would still pass.
- No test for running scripts from a non-root directory (they use `Path(__file__).parent` which should work from anywhere, but this isn't tested).

---

## Dependency Graph Validation

**Graph Correctness: Valid**

```
Phase 1 (dataset.json) ── no dependencies
    |
    ├── Phase 2 (steps 02-05) ── depends on Phase 1 only
    |
    └── Phase 3 (steps 06-09) ── depends on Phase 1 only
```

- Execution order is: **clear and correct**
- Parallelization opportunities are: **well-identified** (Phases 2 and 3 in parallel)
- Blocking dependencies are: **properly documented**
- No circular dependencies exist
- External package dependencies (`scikit-learn`, `scipy`, `numpy`, `pandas`) are already in `pyproject.toml`

---

## Summary of Changes Needed

**Before execution, address:**

1. **Critical (Must Fix):**
   - [ ] Fix `step03_fit_transform.py` description in File Inventory table (line 74) — wrong purpose copied from step02

2. **Important (Should Fix):**
   - [ ] Add prerequisites section with `mkdir -p docs/tfidf/demo/` and `uv sync` before Phase 1
   - [ ] Update "Current State Analysis" (line 25) — directory already exists
   - [ ] Add automated verification command for Phase 1 item count/key check

3. **Optional (Nice to Have):**
   - [ ] Add output-content grep checks to Phase 2 and Phase 3 success criteria
   - [ ] Add note confirming code is ruff-compliant for pre-commit hooks

---

## Reviewer Notes

This is one of the better implementation plans reviewed. The decision to include complete, runnable source code for every file eliminates the most common source of plan failure: ambiguous implementation instructions. The code is clean, follows consistent patterns, and faithfully translates the guide's examples into standalone scripts.

The plan's weakest area is missing "setup" steps (directory creation, dependency sync) that experienced developers take for granted but an automated agent might not infer. Adding a small prerequisites section would bring this plan to near-perfect executability.

The Decision Log table is a particularly strong element — it documents trade-offs (JSON vs CSV, temp dir vs real dir, single file vs split for step 5) with clear rationale, which helps any future maintainer understand design choices.

**Final score breakdown:**

| Dimension | Score | Max |
|---|---|---|
| Accuracy | 19 | 20 |
| Consistency | 14 | 15 |
| Clarity | 23 | 25 |
| Completeness | 21 | 25 |
| Executability | 18 | 20 |
| **Total** | **91** | **100** |

---

**Note:** This review is advisory only. No changes have been made to the original plan. All suggestions require explicit approval before implementation.
