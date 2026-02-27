# Plan Review: CodiEsp Parser Save/Load Implementation Plan

**Review Date:** 2026-02-26
**Reviewer:** Claude Code Review Agent
**Plan Location:** `plan/future-plans/2026-02-26-codiesp-save-load.md`

---

## Executive Summary

**Executability Score:** 78/100 - Good

**Overall Assessment:**
This is a well-structured, mostly executable plan. It follows existing project patterns, provides full file contents for all MODIFY and CREATE actions, and has clear verification commands. The Before/After format for Phase 1 is unambiguous and the test file is complete and copy-pasteable.

However, there is one critical blocker (a corrupted docstring in the Phase 1 "After" block that would introduce an incorrect edit if executed literally), one major concern (the test import path is unverified and could stall an agent), and a missing defensive `mkdir` call that could fail in clean environments.

With the critical and major fixes applied, this plan would score in the high 80s and be fully agent-executable without further human input.

**Recommendation:**
- [ ] Ready for execution
- [x] Ready with minor clarifications
- [ ] Requires improvements before execution
- [ ] Requires major revisions

---

## Detailed Analysis

### 1. Accuracy (16/20)

**Score Breakdown:**
- Technical correctness: 3/5
- File path validity: 5/5
- Codebase understanding: 4/5
- Dependency accuracy: 4/5

**Findings:**
- ✅ Strength: All file paths are consistent and reference real locations in the project (Phase 1.1, Phase 2.1, Phase 3.1).
- ✅ Strength: Code logic is technically correct — `pl.DataFrame.write_parquet()` / `pl.read_parquet()` usage matches the existing `knowledge_base.py` pattern.
- ❌ Critical: The "After" block in Phase 1.1 contains a corrupted docstring — "Only rows with no DIAGNOSTICO rows are omitted" instead of the original "Only rows with type == 'DIAGNOSTICO' are included. Files with no DIAGNOSTICO rows are omitted." A literal copy-paste will introduce a defective docstring.
- ⚠️ Issue: The plan states the output directory "already exists" but `save_to_parquet` never calls `path.parent.mkdir(parents=True, exist_ok=True)`. This is fragile in CI or clean checkouts.

**Suggestions:**
1. Restore the full original docstring in the Phase 1.1 "After" block.
2. Add `path.parent.mkdir(parents=True, exist_ok=True)` before `write_parquet` in `save_to_parquet`.

### 2. Consistency (13/15)

**Score Breakdown:**
- Internal consistency: 4/5
- Naming conventions: 5/5
- Pattern adherence: 4/5

**Findings:**
- ✅ Strength: `save_to_parquet` / `load_from_parquet` naming matches the existing `knowledge_base.py` pattern perfectly.
- ✅ Strength: Column names `file_stem` and `codes` match the variable names used inside `parse_codiesp_diagnostics`.
- ⚠️ Issue: The Phase 1 "Dependencies" section only says "Required by: Phase 2" but the mermaid diagram and Dependencies section both show Phase 3 also depends on Phase 1. The phase-level dependency block is incomplete.

**Suggestions:**
1. Update Phase 1 "Required by" to: "Phase 2, Phase 3".

### 3. Clarity (21/25)

**Score Breakdown:**
- Instruction clarity: 6/7
- Success criteria clarity: 6/7
- Minimal ambiguity: 9/11

**Findings:**
- ✅ Strength: Full file contents are provided for every CREATE and MODIFY action — no partial snippets or pseudocode.
- ✅ Strength: Verification commands are explicit and runnable.
- ⚠️ Issue: Phase 3's success criteria automated check ("File exists") has no runnable command — only a prose checkbox. An agent cannot execute a prose assertion.
- ⚠️ Issue: The import path `from scripts.parse_codiesp import ...` in the test file is asserted without showing the pyproject.toml configuration that makes it work.

**Suggestions:**
1. Replace Phase 3 automated verification prose with a runnable command, e.g.: `test -f data/test-datasets/codiesp/data-pipeline/processed/gold/README.md && echo OK`
2. Add an explicit note in Phase 2 context: "Read `pyproject.toml` to confirm `pythonpath` or package config before writing the import line."

### 4. Completeness (18/25)

**Score Breakdown:**
- All steps present: 8/11
- Context adequate: 5/6
- Edge cases covered: 3/6
- Testing comprehensive: 2/2

**Findings:**
- ✅ Strength: Empty input edge case is covered in `test_save_empty_results`.
- ✅ Strength: Spot-check test on a known entry is included.
- ⚠️ Issue: No test for `load_from_parquet` with a non-existent file path — the `FileNotFoundError` behavior is undocumented and untested.
- ⚠️ Issue: No full regression test run (`uv run pytest`) in any phase's success criteria — only the new test file is checked.
- ⚠️ Issue: No step to verify `pre-commit` / `ruff-format` compliance (the project uses `ruff-format` as well as `ruff-check`).

**Suggestions:**
1. Add `test_load_nonexistent_file` test: `with pytest.raises(Exception): load_from_parquet(tmp_path / "nonexistent.parquet")`.
2. Add `uv run pytest` (full suite) to final success criteria.
3. Add `uv run ruff format --check src/scripts/parse_codiesp.py tests/test_parse_codiesp.py` to automated verification.

### 5. Executability (10/20)

**Score Breakdown:**
- Agent-executable: 4/8
- Dependencies ordered: 4/6
- Success criteria verifiable: 2/6

**Findings:**
- ❌ Critical: The corrupted docstring in Phase 1.1 "After" block means a literal copy-paste introduces a defect. An agent executing the plan will produce incorrect output.
- ⚠️ Issue: The import path `from scripts.parse_codiesp import ...` is unverified. If incorrect (e.g., should be `from src.scripts.parse_codiesp import ...`), all Phase 2 tests fail with `ModuleNotFoundError` and the agent has no fallback instruction.
- ⚠️ Issue: Phase ordering is correct but no working-directory instruction is given. All commands assume the project root, but this is stated only in the manual testing steps, not in automated verification.

**Suggestions:**
1. Fix the corrupted docstring (Critical).
2. Add to Phase 2 context: "Verify the import prefix by checking `tests/test_knowledge_base.py` line 1–10 for its import pattern."
3. Add a note that all commands must be run from the project root.

---

## Identified Pain Points

### Critical Blockers
1. **Corrupted docstring in Phase 1.1 "After" block** — The sentence "Only rows with type == 'DIAGNOSTICO' are included. Files with no" has been truncated to "Only rows with no". Executing the plan as written will produce an incorrect docstring in the script.

### Major Concerns
1. **Unverified import path in Phase 2** — `from scripts.parse_codiesp import ...` assumes `src/` is the Python root in pytest configuration. If incorrect, all tests fail immediately with `ModuleNotFoundError` and the agent has no recovery path.
2. **Missing `mkdir` before `write_parquet`** — `save_to_parquet` will raise `FileNotFoundError` if the parent directory does not exist (e.g., fresh clone, CI). A one-line guard prevents this.

### Minor Issues
1. **Phase 1 "Required by" is incomplete** — says "Phase 2" only; should also list "Phase 3".
2. **Phase 3 automated verification has no runnable command** — the checkbox is prose only.
3. **No `FileNotFoundError` test for `load_from_parquet`**.
4. **No full test suite run** (`uv run pytest`) in success criteria.
5. **No `ruff format --check`** in automated verification (only `ruff check`).
6. **README dataset citation** ("IberLEF 2020") is unverified — the executing agent should be instructed to cross-check against the dataset's own documentation.

---

## Specific Recommendations

### High Priority
1. **Fix corrupted docstring**
   - Location: Phase 1.1, "After" block, `parse_codiesp_diagnostics` docstring
   - Issue: Text truncated — "Only rows with no" instead of "Only rows with type == 'DIAGNOSTICO' are included. Files with no"
   - Fix: Restore to: `Only rows with type == "DIAGNOSTICO" are included. Files with no DIAGNOSTICO rows are omitted from the result.`
   - Impact: Without this fix, the executed plan produces an incorrect docstring that misrepresents the function's behavior.

2. **Add `path.parent.mkdir(parents=True, exist_ok=True)` to `save_to_parquet`**
   - Location: Phase 1.1, "After" block, `save_to_parquet` function body, before `write_parquet`
   - Issue: Directory may not exist in CI or fresh checkouts
   - Fix: Insert `path.parent.mkdir(parents=True, exist_ok=True)` as the first line of `save_to_parquet`
   - Impact: Prevents `FileNotFoundError` in clean environments

3. **Verify import path before Phase 2**
   - Location: Phase 2, Context section
   - Issue: `from scripts.parse_codiesp import ...` is assumed correct but not verified
   - Fix: Add "Read `tests/test_knowledge_base.py` lines 1–5 to confirm the import prefix pattern used by existing tests" to Phase 2 Context
   - Impact: Prevents test failure with no recovery path

### Medium Priority
4. **Add runnable command to Phase 3 automated verification**
   - Location: Phase 3, Success Criteria, Automated Verification
   - Fix: `python -c "from pathlib import Path; assert Path('data/test-datasets/codiesp/data-pipeline/processed/gold/README.md').exists()"`

5. **Add full regression run to success criteria**
   - Location: Phase 2 (or a final overall success criteria section)
   - Fix: Add `uv run pytest` as the last verification step

6. **Fix Phase 1 "Required by"**
   - Location: Phase 1, Dependencies block
   - Fix: Change "Required by: Phase 2" to "Required by: Phase 2, Phase 3"

### Low Priority
7. **Add `FileNotFoundError` test** for `load_from_parquet` in Phase 2 test file
8. **Add `ruff format --check`** to automated verification in Phase 2
9. **Add a note on IberLEF 2020 citation** in Phase 3 — instruct executing agent to verify against dataset's own README

---

## Phase-by-Phase Analysis

### Phase 1: Add save and load
- **Score:** 19/25
- **Readiness:** Needs Work (Critical: corrupted docstring; Major: missing mkdir)
- **Key Issues:**
  - Corrupted docstring in "After" block will produce incorrect output
  - `save_to_parquet` has no directory guard
  - "Required by: Phase 2" is incomplete (Phase 3 also depends on this)
- **Dependencies:** Correctly identified as having no upstream dependencies
- **Success Criteria:** Automated commands are correct but `ruff format --check` is missing

### Phase 2: Tests
- **Score:** 18/25
- **Readiness:** Needs Work (Major: import path unverified)
- **Key Issues:**
  - Import path assumption could cause immediate `ModuleNotFoundError`
  - No `FileNotFoundError` test for `load_from_parquet`
  - Full regression run not included
- **Dependencies:** Correctly depends on Phase 1
- **Success Criteria:** Commands are specific and runnable if import path is correct

### Phase 3: README
- **Score:** 20/25
- **Readiness:** Ready (minor issues only)
- **Key Issues:**
  - Automated verification check has no runnable command
  - IberLEF 2020 citation should be flagged for verification
- **Dependencies:** Correctly depends on Phase 1; correctly parallel with Phase 2
- **Success Criteria:** Manual verification is clear and actionable

---

## Testing Strategy Assessment

**Coverage:** Good

**Unit Testing:**
- Save creates file with correct size: ✅
- Parquet is readable: ✅
- Expected columns present: ✅
- Round-trip count, data, types: ✅
- Spot-check on known entry: ✅
- Empty input: ✅

**Integration Testing:**
- None specified — appropriate for this scope (pure file I/O, no external services)

**Manual Testing:**
- Run `__main__` and verify file creation: ✅ Clear and specific

**Gaps:**
- No test for `load_from_parquet` on a non-existent path
- No test for row ordering guarantee (minor; Parquet preserves order, but undocumented)
- No test for `save_to_parquet` when parent directory doesn't exist (would be addressed by adding `mkdir`)

---

## Dependency Graph Validation

**Graph Correctness:** Mostly Valid (one inconsistency)

**Analysis:**
- Execution order is: clear and correct (1 → 2 and 1 → 3)
- Parallelization opportunities are: correctly identified (Phases 2 and 3 parallel)
- Blocking dependencies are: correctly identified in the global Dependencies section

**Issues:**
- Phase 1's own "Required by" block says only "Phase 2" — missing Phase 3. This is inconsistent with the mermaid diagram and the global Dependencies section. An agent reading only the phase-level block would not know Phase 3 waits on Phase 1.

---

## Summary of Changes Needed

**Before execution, address:**

1. **Critical (Must Fix):**
   - [ ] Restore corrupted docstring in Phase 1.1 "After" block (`parse_codiesp_diagnostics` — restore "Only rows with type == 'DIAGNOSTICO' are included. Files with no")

2. **Important (Should Fix):**
   - [ ] Add `path.parent.mkdir(parents=True, exist_ok=True)` to `save_to_parquet` before `write_parquet`
   - [ ] Add instruction to Phase 2 Context to verify import path by reading existing test files
   - [ ] Fix Phase 1 "Required by" to list both Phase 2 and Phase 3
   - [ ] Add runnable automated verification command to Phase 3

3. **Optional (Nice to Have):**
   - [ ] Add `test_load_nonexistent_file` test in Phase 2
   - [ ] Add `uv run pytest` (full suite) to final success criteria
   - [ ] Add `ruff format --check` to Phase 2 automated verification
   - [ ] Add note to Phase 3 instructing executing agent to verify IberLEF 2020 citation

---

## Reviewer Notes

The plan is well-executed overall. The Before/After full-file approach for Phase 1 is exactly the right format — it eliminates ambiguity entirely. The test file is complete, idiomatic, and follows the project's existing fixture-chain pattern.

The critical docstring issue appears to be a copy-paste truncation artifact introduced during plan editing. It is easy to fix and should not delay execution once corrected.

The import path concern is the most likely real-world failure point: if `src/` is not on the pytest Python path, the test file won't import. The fix is simple — check the existing `test_knowledge_base.py` import line before writing the new test file. Once that is confirmed, the plan is straightforwardly executable.

---

**Note:** This review is advisory only. No changes have been made to the original plan. All suggestions require explicit approval before implementation.
