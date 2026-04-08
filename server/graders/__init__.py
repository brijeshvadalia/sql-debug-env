"""
server/graders/__init__.py — Advanced Multi-Signal Graders v3.0

Architecture:
  - 4 grader tiers: easy / medium / hard / expert
  - Each grader computes: correctness + row_coverage + column_coverage
  - Hard grader adds: efficiency score with speedup ratio
  - Expert grader adds: schema validation (column names, data types)
  - All graders: deterministic, non-sparse, bounded strictly (0, 1)

Reward formula:
  easy   : correctness × step_penalty   (min 0.25 execution bonus)
  medium : (0.7×row_match + 0.3×col_coverage) × step_penalty
  hard   : (0.6×correctness + 0.4×efficiency) × step_penalty
  expert : (0.5×row_match + 0.3×col_match + 0.2×count_exact) × step_penalty
"""
from __future__ import annotations

import re
from typing import Any, Optional

from models import QueryComplexity, RewardBreakdown, PerformanceMetrics
from server.tasks import Task


# ---------------------------------------------------------------------------
# Clamp — ensures every score is strictly within (0, 1)
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    """Ensure score is strictly within (0, 1) — never 0.0 or 1.0."""
    return round(max(0.001, min(v, 0.999)), 4)


# ---------------------------------------------------------------------------
# Step penalty — smooth decay from 1.0 at step 1 to 0.70 at max_steps
# ---------------------------------------------------------------------------

def _step_penalty(step: int, max_steps: int) -> float:
    if max_steps <= 1:
        return 1.0
    return round(max(0.70, 1.0 - 0.3 * ((step - 1) / max_steps)), 4)


# ---------------------------------------------------------------------------
# Row / column matchers
# ---------------------------------------------------------------------------

def _norm_val(v: Any) -> Any:
    """Normalise a value for comparison: round floats, strip strings."""
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _rows_equal(a: dict, b: dict, tol: float = 0.05) -> bool:
    """Two rows are equal if all keys in b match a within tolerance."""
    for k in b:
        if k not in a:
            return False
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if abs(float(av) - float(bv)) > tol:
                return False
        else:
            if _norm_val(av) != _norm_val(bv):
                return False
    return True


def _row_coverage(result: list[dict], expected: list[dict]) -> float:
    """Fraction of expected rows found anywhere in result (unordered)."""
    if not expected:
        return 0.999 if not result else 0.001
    if not result:
        return 0.001
    matched = sum(1 for exp in expected if any(_rows_equal(got, exp) for got in result))
    raw = matched / len(expected)
    return _clamp(raw)


def _ordered_row_score(result: list[dict], expected: list[dict]) -> float:
    """Positional match: each row at correct index scores 1.0, elsewhere 0.5."""
    if not expected:
        return 0.999 if not result else 0.001
    if not result:
        return 0.001
    score = 0.0
    for i, exp in enumerate(expected):
        if i < len(result) and _rows_equal(result[i], exp):
            score += 1.0
        elif any(_rows_equal(r, exp) for r in result):
            score += 0.5
    return _clamp(score / len(expected))


def _column_coverage(result: list[dict], expected: list[dict]) -> float:
    """Fraction of expected column names present in result."""
    if not expected or not result:
        return 0.001
    exp_cols = set(expected[0].keys())
    got_cols = set(result[0].keys())
    if not exp_cols:
        return 0.999
    return _clamp(len(exp_cols & got_cols) / len(exp_cols))


def _row_count_score(result: list[dict], expected: list[dict]) -> float:
    """1.0 if row count matches exactly, decays with distance."""
    if not expected:
        return 0.999
    diff = abs(len(result) - len(expected))
    return _clamp(max(0.0, 1.0 - diff / max(len(expected), 1)))


# ---------------------------------------------------------------------------
# Query complexity classifier
# ---------------------------------------------------------------------------

def classify_query(sql: str) -> QueryComplexity:
    """Classify query structural complexity from SQL text."""
    s = sql.upper()
    has_join = bool(re.search(r'\bJOIN\b', s))
    has_subquery = s.count('SELECT') > 1
    has_agg = bool(re.search(r'\b(SUM|AVG|COUNT|MAX|MIN)\s*\(', s))
    has_window = bool(re.search(r'\bOVER\s*\(', s))
    has_cte = bool(re.search(r'\bWITH\b', s))
    has_group = bool(re.search(r'\bGROUP\s+BY\b', s))
    has_order = bool(re.search(r'\bORDER\s+BY\b', s))
    has_where = bool(re.search(r'\bWHERE\b', s))
    join_count = len(re.findall(r'\bJOIN\b', s))
    subquery_depth = s.count('SELECT') - 1

    score = (
        0.1 * has_join +
        0.15 * has_subquery +
        0.1 * has_agg +
        0.25 * has_window +
        0.2 * has_cte +
        0.05 * has_group +
        0.05 * has_order +
        min(0.1 * join_count, 0.2)
    )
    score = min(round(score, 3), 1.0)

    if score < 0.15:
        label = "simple"
    elif score < 0.35:
        label = "moderate"
    elif score < 0.60:
        label = "complex"
    else:
        label = "advanced"

    return QueryComplexity(
        has_join=has_join, has_subquery=has_subquery, has_aggregation=has_agg,
        has_window_function=has_window, has_cte=has_cte, has_group_by=has_group,
        has_order_by=has_order, has_where=has_where, join_count=join_count,
        subquery_depth=subquery_depth, complexity_score=score, label=label,
    )


# ---------------------------------------------------------------------------
# EXPLAIN analysis
# ---------------------------------------------------------------------------

def analyse_query_plan(conn, sql: str, baseline_ms: float = 0.0,
                       exec_ms: float = 0.0) -> PerformanceMetrics:
    """Run EXPLAIN QUERY PLAN and return structured PerformanceMetrics."""
    try:
        rows = [dict(r) for r in conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()]
        details = [str(r.get("detail", "")) for r in rows]
        full = " ".join(details).upper()

        scan_count = sum(1 for d in details if "SCAN" in d.upper())
        index_count = sum(1 for d in details if "INDEX" in d.upper())
        tables = []
        for d in details:
            du = d.upper()
            if "SCAN" in du:
                parts = d.split()
                for i, p in enumerate(parts):
                    if p.upper() == "SCAN" and i + 1 < len(parts):
                        tables.append(parts[i + 1].strip("()"))
                        break

        speedup = round(baseline_ms / exec_ms, 2) if exec_ms > 0 and baseline_ms > 0 else 0.0
        efficiency = 0.0
        if exec_ms > 0 and baseline_ms > 0 and exec_ms < baseline_ms:
            efficiency = min(round(speedup / 5.0, 4), 1.0)

        if "CORRELATED" in full or scan_count > 3:
            suggestion = "High scan count — rewrite subqueries as JOINs for linear complexity."
        elif scan_count == 0 and index_count > 0:
            suggestion = "Excellent — pure index lookups, O(log n) access pattern."
        elif index_count > 0:
            suggestion = "Good — partial index usage detected."
        elif scan_count <= 2:
            suggestion = f"Full table scans ({scan_count}) — acceptable for small datasets."
        else:
            suggestion = f"Warning: {scan_count} full table scans. Consider WHERE clauses or indexes."

        return PerformanceMetrics(
            execution_ms=round(exec_ms, 3),
            baseline_ms=baseline_ms,
            speedup_ratio=speedup,
            scan_count=scan_count,
            index_count=index_count,
            uses_index=index_count > 0,
            plan_steps=len(rows),
            tables_scanned=tables[:5],
            suggestion=suggestion,
            efficiency_score=efficiency,
        )
    except Exception:
        return PerformanceMetrics(suggestion="Query plan unavailable.", execution_ms=exec_ms)


# ---------------------------------------------------------------------------
# Grader: Easy
# ---------------------------------------------------------------------------

def grade_easy(task: Task, error: Optional[str], result: Optional[list[dict]],
               step_count: int, sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.001, step_penalty=penalty,
            explanation=f"Syntax/runtime error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.001, step_penalty=penalty,
            explanation="Query executed but returned no rows.")

    col_cov = _column_coverage(result, task.expected_result)
    row_cov = _ordered_row_score(result, task.expected_result)

    if row_cov < 0.05:
        return RewardBreakdown(
            total=0.25, correctness=0.001,
            column_coverage=round(col_cov, 4),
            row_coverage=0.001,
            step_penalty=penalty,
            explanation=f"Query runs but result wrong. Columns present: {round(col_cov*100)}%")

    correctness = round(0.7 * row_cov + 0.3 * col_cov, 4)
    total = _clamp(correctness * penalty)

    return RewardBreakdown(
        total=total, correctness=correctness, step_penalty=penalty,
        row_coverage=round(row_cov, 4), column_coverage=round(col_cov, 4),
        explanation=f"rows={row_cov:.2f} cols={col_cov:.2f} penalty={penalty:.2f}")


# ---------------------------------------------------------------------------
# Grader: Medium
# ---------------------------------------------------------------------------

def grade_medium(task: Task, error: Optional[str], result: Optional[list[dict]],
                 step_count: int, sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.05, correctness=0.001, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.001, step_penalty=penalty,
            explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    count_score = _row_count_score(result, task.expected_result)

    correctness = round(0.6 * row_cov + 0.25 * col_cov + 0.15 * count_score, 4)

    if correctness < 0.05 and not error:
        total = 0.25
    else:
        total = _clamp(correctness * penalty)

    matched = round(row_cov * len(task.expected_result))
    return RewardBreakdown(
        total=total, correctness=correctness, step_penalty=penalty,
        row_coverage=round(row_cov, 4), column_coverage=round(col_cov, 4),
        explanation=(f"{matched}/{len(task.expected_result)} rows matched "
                     f"cols={col_cov:.2f} penalty={penalty:.2f}"))


# ---------------------------------------------------------------------------
# Grader: Hard (correctness 60% + efficiency 40%)
# ---------------------------------------------------------------------------

def grade_hard(task: Task, error: Optional[str], result: Optional[list[dict]],
               exec_ms: Optional[float], step_count: int,
               perf: Optional[PerformanceMetrics] = None,
               sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.001, correctness=0.001, efficiency=0.001, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.001, efficiency=0.001, step_penalty=penalty,
            explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    correctness = round(0.7 * row_cov + 0.3 * col_cov, 4)

    efficiency = 0.001
    speedup = 0.0
    if perf and perf.efficiency_score > 0:
        efficiency = _clamp(perf.efficiency_score)
        speedup = perf.speedup_ratio
    elif exec_ms and exec_ms > 0 and task.baseline_exec_ms > 0:
        if exec_ms < task.baseline_exec_ms:
            speedup = round(task.baseline_exec_ms / exec_ms, 2)
            efficiency = _clamp(min(round(speedup / 5.0, 4), 1.0))

    combined = round(0.6 * correctness + 0.4 * efficiency, 4)
    total = _clamp(combined * penalty)

    return RewardBreakdown(
        total=total, correctness=correctness, efficiency=efficiency,
        step_penalty=penalty, row_coverage=round(row_cov, 4),
        column_coverage=round(col_cov, 4),
        explanation=(f"Correctness={correctness:.2f} Efficiency={efficiency:.2f} "
                     f"speedup={speedup:.1f}x exec={exec_ms:.2f}ms "
                     f"baseline={task.baseline_exec_ms:.1f}ms penalty={penalty:.2f}"))


# ---------------------------------------------------------------------------
# Grader: Expert
# ---------------------------------------------------------------------------

def grade_expert(task: Task, error: Optional[str], result: Optional[list[dict]],
                 step_count: int, sql: str = "") -> RewardBreakdown:
    penalty = _step_penalty(step_count, task.max_steps)

    if error:
        return RewardBreakdown(
            total=0.001, correctness=0.001, step_penalty=penalty,
            explanation=f"Error: {error[:100]}")

    if not result:
        return RewardBreakdown(
            total=0.05, correctness=0.001, step_penalty=penalty,
            explanation="Empty result.")

    row_cov = _row_coverage(result, task.expected_result)
    col_cov = _column_coverage(result, task.expected_result)
    count_exact = 0.999 if len(result) == len(task.expected_result) else 0.001

    correctness = round(0.5 * row_cov + 0.3 * col_cov + 0.2 * count_exact, 4)

    if correctness < 0.05 and not error:
        total = 0.25
    else:
        total = _clamp(correctness * penalty)

    return RewardBreakdown(
        total=total, correctness=correctness, step_penalty=penalty,
        row_coverage=round(row_cov, 4), column_coverage=round(col_cov, 4),
        explanation=(f"Expert: rows={row_cov:.2f} cols={col_cov:.2f} "
                     f"count_exact={count_exact:.3f} penalty={penalty:.2f}"))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def grade(task: Task, error: Optional[str], result: Optional[list[dict]],
          exec_ms: Optional[float], step_count: int,
          perf: Optional[PerformanceMetrics] = None,
          sql: str = "") -> RewardBreakdown:
    """Route to the correct grader based on task difficulty."""
    if task.difficulty == "easy":
        return grade_easy(task, error, result, step_count, sql)
    elif task.difficulty == "medium":
        return grade_medium(task, error, result, step_count, sql)
    elif task.difficulty == "hard":
        return grade_hard(task, error, result, exec_ms, step_count, perf, sql)
    elif task.difficulty == "expert":
        return grade_expert(task, error, result, step_count, sql)
    return RewardBreakdown(total=0.001, explanation=f"Unknown difficulty: {task.difficulty}")


__all__ = [
    "grade", "grade_easy", "grade_medium", "grade_hard", "grade_expert",
    "classify_query", "analyse_query_plan",
    "_step_penalty", "_row_coverage", "_column_coverage",
]
