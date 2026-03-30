"""
server/graders/__init__.py

Deterministic graders for all three tasks.

Each grader returns a RewardBreakdown with total in [0.0, 1.0].
All graders are:
  - Deterministic: same inputs → same score
  - Partial-credit: non-zero signal on every trajectory step
  - Interpretable: reward_breakdown.explanation tells the agent what happened

Grading weights:
  fix_syntax_error  → correctness only (step penalty applied)
  fix_logic_error   → row-level correctness (partial credit per matched row)
  optimize_query    → 60% correctness + 40% efficiency (speedup vs baseline)
"""

from __future__ import annotations

import math
from typing import Any, Optional

from models import RewardBreakdown
from server.tasks import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_penalty(step_count: int, max_steps: int) -> float:
    """
    Smooth decreasing multiplier: 1.0 at step 1, ~0.7 at max_steps.
    Provides gradient signal throughout the trajectory.
    """
    if max_steps <= 1:
        return 1.0
    # Decay: 1 - 0.3 * (steps_used / max_steps)
    fraction_used = (step_count - 1) / max_steps
    return max(0.7, 1.0 - 0.3 * fraction_used)


def _normalise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise row values for comparison:
      - Round floats to 2 decimal places
      - Convert all values to string-comparable form
    """
    normalised = []
    for row in rows:
        normalised.append({
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in row.items()
        })
    return normalised


def _row_match_score(
    result: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    ordered: bool = True,
) -> float:
    """
    Compute row-level correctness score.

    ordered=True  → checks both content and position (for ORDER BY tasks)
    ordered=False → checks only content membership
    """
    if not expected:
        return 1.0 if not result else 0.0

    norm_result   = _normalise_rows(result)
    norm_expected = _normalise_rows(expected)

    if ordered:
        # Score each position: full credit for exact match, half for wrong position
        matches = 0.0
        for i, exp_row in enumerate(norm_expected):
            if i < len(norm_result) and norm_result[i] == exp_row:
                matches += 1.0
            elif exp_row in norm_result:
                matches += 0.5   # partial credit: right row, wrong position
        return matches / len(norm_expected)
    else:
        matched = sum(1 for row in norm_expected if row in norm_result)
        return matched / len(norm_expected)


# ---------------------------------------------------------------------------
# Task 1 grader — Easy
# ---------------------------------------------------------------------------

def grade_easy(
    task: Task,
    error: Optional[str],
    result: Optional[list[dict[str, Any]]],
    step_count: int,
) -> RewardBreakdown:
    """
    fix_syntax_error grader.

    Scoring:
      - If query fails with error           → 0.05 (tiny signal: query was tried)
      - If query succeeds but result wrong  → 0.40 * step_penalty
      - If query succeeds and result right  → 1.0  * step_penalty
    """
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        # Still give tiny positive reward so the agent has a gradient
        return RewardBreakdown(
            correctness=0.0,
            step_penalty=penalty,
            total=0.05,
            explanation=f"Query failed: {error[:120]}",
        )

    if result is None:
        return RewardBreakdown(
            correctness=0.0,
            step_penalty=penalty,
            total=0.1,
            explanation="Query returned no result object.",
        )

    correctness = _row_match_score(result, task.expected_result, ordered=True)
    # Execution bonus: query ran (even if wrong) → signal above error level
    execution_bonus = 0.25 if correctness < 0.1 else 0.0
    raw = max(correctness * penalty, execution_bonus)
    total = round(min(raw, 1.0), 4)

    return RewardBreakdown(
        correctness=round(correctness, 4),
        step_penalty=round(penalty, 4),
        total=total,
        explanation=(
            f"Execution OK. Correctness={correctness:.2f} "
            f"(matched {int(correctness * len(task.expected_result))}"
            f"/{len(task.expected_result)} rows). "
            f"Step penalty={penalty:.2f}."
        ),
    )


# ---------------------------------------------------------------------------
# Task 2 grader — Medium
# ---------------------------------------------------------------------------

def grade_medium(
    task: Task,
    error: Optional[str],
    result: Optional[list[dict[str, Any]]],
    step_count: int,
) -> RewardBreakdown:
    """
    fix_logic_error grader.

    Scoring:
      - Execution error                 → 0.05
      - Partial row match (unordered)   → correctness_fraction * step_penalty
      - All rows correct (unordered)    → 1.0 * step_penalty

    We use unordered matching because the agent might use a different but valid
    ORDER BY. The grader checks row membership, not position.
    """
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            correctness=0.0,
            step_penalty=penalty,
            total=0.05,
            explanation=f"Execution error: {error[:120]}",
        )

    if not result:
        return RewardBreakdown(
            correctness=0.0,
            step_penalty=penalty,
            total=0.05,
            explanation="Query returned empty result set.",
        )

    correctness = _row_match_score(result, task.expected_result, ordered=False)
    total = round(min(correctness * penalty, 1.0), 4)
    matched_n = round(correctness * len(task.expected_result))

    return RewardBreakdown(
        correctness=round(correctness, 4),
        step_penalty=round(penalty, 4),
        total=total,
        explanation=(
            f"Row-level correctness={correctness:.2f} "
            f"({matched_n}/{len(task.expected_result)} expected rows matched). "
            f"Step penalty={penalty:.2f}."
        ),
    )


# ---------------------------------------------------------------------------
# Task 3 grader — Hard
# ---------------------------------------------------------------------------

def grade_hard(
    task: Task,
    error: Optional[str],
    result: Optional[list[dict[str, Any]]],
    exec_ms: Optional[float],
    step_count: int,
) -> RewardBreakdown:
    """
    optimize_query grader.

    Scoring = 0.6 × correctness + 0.4 × efficiency, then × step_penalty.

    Correctness (0.0–1.0):
      Row-level match (unordered). Checks id, name, region, tier,
      total_revenue (±0.05 tolerance), order_count.

    Efficiency (0.0–1.0):
      speedup = baseline_ms / exec_ms
      efficiency = min(speedup / 5.0, 1.0)   # full score at 5× speedup
      If exec_ms is None or ≥ baseline → efficiency = 0.0
    """
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            correctness=0.0,
            efficiency=0.0,
            step_penalty=penalty,
            total=0.0,
            explanation=f"Execution error: {error[:120]}",
        )

    if not result:
        return RewardBreakdown(
            correctness=0.0,
            efficiency=0.0,
            step_penalty=penalty,
            total=0.05,
            explanation="Query returned empty result.",
        )

    # ── Correctness with numeric tolerance ──────────────────────────────────
    norm_result   = _normalise_rows(result)
    norm_expected = _normalise_rows(task.expected_result)

    matched = 0
    for exp in norm_expected:
        for got in norm_result:
            if _hard_row_match(exp, got):
                matched += 1
                break
    correctness = matched / len(norm_expected)

    # ── Efficiency ───────────────────────────────────────────────────────────
    if exec_ms and exec_ms > 0 and task.baseline_exec_ms > 0 and exec_ms < task.baseline_exec_ms:
        speedup    = task.baseline_exec_ms / exec_ms
        efficiency = min(speedup / 5.0, 1.0)
    else:
        efficiency = 0.0

    combined = 0.6 * correctness + 0.4 * efficiency
    total    = round(min(combined * penalty, 1.0), 4)

    return RewardBreakdown(
        correctness=round(correctness, 4),
        efficiency=round(efficiency, 4),
        step_penalty=round(penalty, 4),
        total=total,
        explanation=(
            f"Correctness={correctness:.2f} ({matched}/{len(norm_expected)} rows). "
            f"Efficiency={efficiency:.2f} "
            f"(exec={exec_ms:.2f}ms, baseline={task.baseline_exec_ms:.2f}ms). "
            f"Combined={combined:.2f} × step_penalty={penalty:.2f} = {total}."
        ),
    )


def _hard_row_match(exp: dict, got: dict) -> bool:
    """
    Flexible row comparison for optimize_query grader.
    Allows ±0.05 tolerance on float columns (total_revenue).
    """
    for key in exp:
        if key not in got:
            return False
        ev, gv = exp[key], got[key]
        if isinstance(ev, float) or isinstance(gv, float):
            try:
                if abs(float(ev) - float(gv)) > 0.05:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if ev != gv:
                return False
    return True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade(
    task: Task,
    error: Optional[str],
    result: Optional[list[dict[str, Any]]],
    exec_ms: Optional[float],
    step_count: int,
) -> RewardBreakdown:
    """Top-level dispatch — routes to the correct grader by task_id."""
    if task.task_id == "fix_syntax_error":
        return grade_easy(task, error, result, step_count)
    elif task.task_id == "fix_logic_error":
        return grade_medium(task, error, result, step_count)
    elif task.task_id == "optimize_query":
        return grade_hard(task, error, result, exec_ms, step_count)
    else:
        return RewardBreakdown(
            total=0.0,
            explanation=f"Unknown task_id: {task.task_id}",
        )


__all__ = ["grade", "grade_easy", "grade_medium", "grade_hard"]
