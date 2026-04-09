"""
server/graders/__init__.py -- SQL Debug Environment v3.0

RULE: Every reward total must be STRICTLY between 0 and 1.
      0.0 and 1.0 are both REJECTED by the Phase 2 validator.
      All totals clamped to [0.01, 0.99] via _strict().

Grader tiers:
  easy   : ordered row match + column coverage
  medium : unordered row match + column coverage + count
  hard   : correctness 60% + efficiency 40%
  expert : strict row+column+count match
"""
from __future__ import annotations

import re
from typing import Any, Optional

from models import QueryComplexity, RewardBreakdown, PerformanceMetrics
from server.tasks import Task


# ---------------------------------------------------------------------------
# Core: clamp every reward to strictly (0, 1)
# ---------------------------------------------------------------------------

def _strict(score: float) -> float:
    """Ensure score is strictly (0.0, 1.0) exclusive. Never 0.0, never 1.0."""
    try:
        v = float(score)
    except (TypeError, ValueError):
        v = 0.05
    return round(max(0.01, min(0.99, v)), 4)


# ---------------------------------------------------------------------------
# Step penalty: smooth decay 0.99 -> 0.70 across steps
# ---------------------------------------------------------------------------

def _step_penalty(step: int, max_steps: int) -> float:
    if max_steps <= 1:
        return 0.99
    return max(0.70, 1.0 - 0.3 * ((step - 1) / max_steps))


# ---------------------------------------------------------------------------
# Row / column comparison helpers
# ---------------------------------------------------------------------------

def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _rows_equal(a: dict, b: dict, tol: float = 0.05) -> bool:
    for k in b:
        if k not in a:
            return False
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if abs(float(av) - float(bv)) > tol:
                return False
        else:
            if _norm(av) != _norm(bv):
                return False
    return True


def _row_coverage(result: list, expected: list) -> float:
    """Fraction of expected rows found in result (unordered)."""
    if not expected:
        return 0.98 if not result else 0.02
    if not result:
        return 0.02
    matched = sum(1 for exp in expected
                  if any(_rows_equal(got, exp) for got in result))
    return _strict(matched / len(expected))


def _ordered_row_score(result: list, expected: list) -> float:
    """Positional match: exact position=1.0, wrong position=0.5."""
    if not expected:
        return 0.98 if not result else 0.02
    if not result:
        return 0.02
    score = 0.0
    for i, exp in enumerate(expected):
        if i < len(result) and _rows_equal(result[i], exp):
            score += 1.0
        elif any(_rows_equal(r, exp) for r in result):
            score += 0.5
    return _strict(score / len(expected))


def _column_coverage(result: list, expected: list) -> float:
    """Fraction of expected column names present in result."""
    if not expected or not result:
        return 0.02
    exp_cols = set(expected[0].keys())
    got_cols = set(result[0].keys())
    if not exp_cols:
        return 0.98
    return _strict(len(exp_cols & got_cols) / len(exp_cols))


def _count_score(result: list, expected: list) -> float:
    """Score based on row count closeness."""
    if not expected:
        return 0.98
    diff = abs(len(result) - len(expected))
    return _strict(max(0.01, 1.0 - diff / max(len(expected), 1)))


# ---------------------------------------------------------------------------
# Query complexity classifier
# ---------------------------------------------------------------------------

def classify_query(sql: str) -> QueryComplexity:
    s = sql.upper()
    has_join = bool(re.search(r"\bJOIN\b", s))
    has_subquery = s.count("SELECT") > 1
    has_agg = bool(re.search(r"\b(SUM|AVG|COUNT|MAX|MIN)\s*\(", s))
    has_window = bool(re.search(r"\bOVER\s*\(", s))
    has_cte = bool(re.search(r"\bWITH\b", s))
    has_group = bool(re.search(r"\bGROUP\s+BY\b", s))
    has_order = bool(re.search(r"\bORDER\s+BY\b", s))
    has_where = bool(re.search(r"\bWHERE\b", s))
    join_count = len(re.findall(r"\bJOIN\b", s))
    subquery_depth = s.count("SELECT") - 1

    score = (
        0.10 * has_join + 0.15 * has_subquery + 0.10 * has_agg +
        0.25 * has_window + 0.20 * has_cte + 0.05 * has_group +
        0.05 * has_order + min(0.10 * join_count, 0.20)
    )
    score = min(round(score, 3), 0.98)
    label = (
        "simple" if score < 0.15 else
        "moderate" if score < 0.35 else
        "complex" if score < 0.60 else "advanced"
    )
    return QueryComplexity(
        has_join=has_join, has_subquery=has_subquery, has_aggregation=has_agg,
        has_window_function=has_window, has_cte=has_cte, has_group_by=has_group,
        has_order_by=has_order, has_where=has_where, join_count=join_count,
        subquery_depth=subquery_depth, complexity_score=score, label=label,
    )


# ---------------------------------------------------------------------------
# EXPLAIN query plan analysis
# ---------------------------------------------------------------------------

def analyse_query_plan(conn, sql: str, baseline_ms: float = 0.0,
                       exec_ms: float = 0.0) -> PerformanceMetrics:
    try:
        rows = [dict(r) for r in
                conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()]
        details = [str(r.get("detail", "")) for r in rows]
        scan_count = sum(1 for d in details if "SCAN" in d.upper())
        index_count = sum(1 for d in details if "INDEX" in d.upper())
        speedup = (round(baseline_ms / exec_ms, 2)
                   if exec_ms > 0 and baseline_ms > 0 else 0.0)
        efficiency = 0.0
        if exec_ms > 0 and baseline_ms > 0 and exec_ms < baseline_ms:
            efficiency = _strict(speedup / 5.0)
        suggestion = (
            "Excellent — index only." if scan_count == 0 and index_count > 0
            else f"Scans: {scan_count}."
        )
        return PerformanceMetrics(
            execution_ms=round(exec_ms, 3), baseline_ms=baseline_ms,
            speedup_ratio=speedup, scan_count=scan_count,
            index_count=index_count, uses_index=index_count > 0,
            plan_steps=len(rows), tables_scanned=[],
            suggestion=suggestion, efficiency_score=efficiency,
        )
    except Exception:
        return PerformanceMetrics(
            suggestion="Plan unavailable.", execution_ms=exec_ms)


# ---------------------------------------------------------------------------
# GRADER: Easy (ordered row match)
# ---------------------------------------------------------------------------

def grade_easy(task: Task, error: Optional[str],
               result: Optional[list], step_count: int,
               sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation="Empty result.")

    col_cov = _column_coverage(result, task.expected_result)
    row_cov = _ordered_row_score(result, task.expected_result)

    if row_cov < 0.05:
        return RewardBreakdown(
            total=0.25, correctness=0.25,
            column_coverage=col_cov, row_coverage=0.05,
            step_penalty=penalty,
            explanation=f"Runs but wrong. cols={col_cov:.2f}")

    correctness = 0.7 * row_cov + 0.3 * col_cov
    total = _strict(correctness * penalty)
    return RewardBreakdown(
        total=total, correctness=_strict(correctness),
        step_penalty=penalty, row_coverage=row_cov,
        column_coverage=col_cov,
        explanation=f"rows={row_cov:.2f} cols={col_cov:.2f} pen={penalty:.2f}")


# ---------------------------------------------------------------------------
# GRADER: Medium (unordered row match)
# ---------------------------------------------------------------------------

def grade_medium(task: Task, error: Optional[str],
                 result: Optional[list], step_count: int,
                 sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    cnt = _count_score(result, task.expected_result)
    correctness = 0.6 * row_cov + 0.25 * col_cov + 0.15 * cnt
    total = _strict(correctness * penalty) if correctness >= 0.05 else 0.25
    matched = round(row_cov * len(task.expected_result))
    return RewardBreakdown(
        total=total, correctness=_strict(correctness),
        step_penalty=penalty, row_coverage=row_cov,
        column_coverage=col_cov,
        explanation=(f"{matched}/{len(task.expected_result)} rows "
                     f"cols={col_cov:.2f} pen={penalty:.2f}"))


# ---------------------------------------------------------------------------
# GRADER: Hard (correctness 60% + efficiency 40%)
# ---------------------------------------------------------------------------

def grade_hard(task: Task, error: Optional[str],
               result: Optional[list], exec_ms: Optional[float],
               step_count: int, perf: Optional[PerformanceMetrics] = None,
               sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.05, efficiency=0.05,
            step_penalty=penalty, explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.05, efficiency=0.05,
            step_penalty=penalty, explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    correctness = 0.7 * row_cov + 0.3 * col_cov

    efficiency = 0.05
    speedup = 0.0
    if perf and perf.efficiency_score > 0:
        efficiency = _strict(perf.efficiency_score)
        speedup = perf.speedup_ratio
    elif (exec_ms and exec_ms > 0 and task.baseline_exec_ms > 0
          and exec_ms < task.baseline_exec_ms):
        speedup = round(task.baseline_exec_ms / exec_ms, 2)
        efficiency = _strict(speedup / 5.0)

    combined = 0.6 * correctness + 0.4 * efficiency
    total = _strict(combined * penalty)
    return RewardBreakdown(
        total=total, correctness=_strict(correctness),
        efficiency=efficiency, step_penalty=penalty,
        row_coverage=row_cov, column_coverage=col_cov,
        explanation=(f"corr={correctness:.2f} eff={efficiency:.2f} "
                     f"spd={speedup:.1f}x pen={penalty:.2f}"))


# ---------------------------------------------------------------------------
# GRADER: Expert (strict row + column + count)
# ---------------------------------------------------------------------------

def grade_expert(task: Task, error: Optional[str],
                 result: Optional[list], step_count: int,
                 sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.05, step_penalty=penalty,
            explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    cnt_exact = 0.98 if len(result) == len(task.expected_result) else 0.05
    correctness = 0.5 * row_cov + 0.3 * col_cov + 0.2 * cnt_exact
    total = _strict(correctness * penalty) if correctness >= 0.05 else 0.25
    return RewardBreakdown(
        total=total, correctness=_strict(correctness),
        step_penalty=penalty, row_coverage=row_cov,
        column_coverage=col_cov,
        explanation=(f"rows={row_cov:.2f} cols={col_cov:.2f} "
                     f"cnt={cnt_exact:.2f} pen={penalty:.2f}"))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade(task: Task, error: Optional[str], result: Optional[list],
          exec_ms: Optional[float], step_count: int,
          perf: Optional[PerformanceMetrics] = None,
          sql: str = "") -> RewardBreakdown:
    if task.difficulty == "easy":
        return grade_easy(task, error, result, step_count, sql)
    elif task.difficulty == "medium":
        return grade_medium(task, error, result, step_count, sql)
    elif task.difficulty == "hard":
        return grade_hard(task, error, result, exec_ms, step_count, perf, sql)
    elif task.difficulty == "expert":
        return grade_expert(task, error, result, step_count, sql)
    return RewardBreakdown(
        total=0.05, explanation=f"Unknown difficulty: {task.difficulty}")


__all__ = [
    "grade", "grade_easy", "grade_medium", "grade_hard", "grade_expert",
    "classify_query", "analyse_query_plan", "_strict",
]
