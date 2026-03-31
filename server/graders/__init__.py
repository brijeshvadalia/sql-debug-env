"""
server/graders/__init__.py — Deterministic graders for all 8 tasks.

All graders return scores in [0.0, 1.0].
All graders are deterministic: same inputs -> same score.
All graders provide partial credit (non-sparse reward signal).

Reward structure:
  error     (0.05) < executes_wrong (0.25) < partial (0.3-0.7) < correct (0.8+)

Step penalty: 1.0 - 0.3*(steps/max_steps) -> range [0.70, 1.00]
"""
from __future__ import annotations
from typing import Any, Optional
from models import RewardBreakdown
from server.tasks import Task


def _step_penalty(step_count: int, max_steps: int) -> float:
    if max_steps <= 1:
        return 1.0
    return max(0.70, 1.0 - 0.3 * ((step_count - 1) / max_steps))


def _normalise(rows: list[dict]) -> list[dict]:
    return [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in rows]


def _ordered_score(result: list[dict], expected: list[dict]) -> float:
    if not expected:
        return 1.0 if not result else 0.0
    nr, ne = _normalise(result), _normalise(expected)
    matches = 0.0
    for i, exp in enumerate(ne):
        if i < len(nr) and nr[i] == exp:
            matches += 1.0
        elif exp in nr:
            matches += 0.5
    return matches / len(ne)


def _unordered_score(result: list[dict], expected: list[dict]) -> float:
    if not expected:
        return 1.0 if not result else 0.0
    nr, ne = _normalise(result), _normalise(expected)
    matched = sum(1 for e in ne if e in nr)
    return matched / len(ne)


def _float_row_match(exp: dict, got: dict, tol: float = 0.05) -> bool:
    for k in exp:
        if k not in got:
            return False
        ev, gv = exp[k], got[k]
        if isinstance(ev, float) or isinstance(gv, float):
            try:
                if abs(float(ev) - float(gv)) > tol:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if ev != gv:
                return False
    return True


def _float_unordered_score(result: list[dict], expected: list[dict]) -> float:
    if not expected:
        return 1.0 if not result else 0.0
    nr = _normalise(result)
    ne = _normalise(expected)
    matched = sum(1 for e in ne if any(_float_row_match(e, r) for r in nr))
    return matched / len(ne)


# ── Easy grader ─────────────────────────────────────────────────────────────

def grade_easy(task, error, result, step_count):
    penalty = _step_penalty(step_count, task.max_steps)
    if error:
        return RewardBreakdown(total=0.05, correctness=0.0, step_penalty=penalty,
                               explanation=f"Query failed: {error[:120]}")
    if result is None:
        return RewardBreakdown(total=0.05, correctness=0.0, step_penalty=penalty,
                               explanation="No result returned.")
    correctness = _ordered_score(result, task.expected_result)
    execution_bonus = 0.25 if correctness < 0.1 else 0.0
    total = round(min(max(correctness * penalty, execution_bonus), 1.0), 4)
    return RewardBreakdown(total=total, correctness=round(correctness, 4),
                           step_penalty=round(penalty, 4),
                           explanation=f"Correctness={correctness:.2f} step_penalty={penalty:.2f}")


# ── Medium grader (unordered row match) ─────────────────────────────────────

def grade_medium(task, error, result, step_count):
    penalty = _step_penalty(step_count, task.max_steps)
    if error:
        return RewardBreakdown(total=0.05, correctness=0.0, step_penalty=penalty,
                               explanation=f"Error: {error[:120]}")
    if not result:
        return RewardBreakdown(total=0.05, correctness=0.0, step_penalty=penalty,
                               explanation="Empty result set.")
    correctness = _float_unordered_score(result, task.expected_result)
    total = round(min(correctness * penalty, 1.0), 4)
    matched = round(correctness * len(task.expected_result))
    return RewardBreakdown(total=total, correctness=round(correctness, 4),
                           step_penalty=round(penalty, 4),
                           explanation=f"{matched}/{len(task.expected_result)} rows matched. step_penalty={penalty:.2f}")


# ── Hard grader (correctness 60% + efficiency 40%) ──────────────────────────

def grade_hard(task, error, result, exec_ms, step_count):
    penalty = _step_penalty(step_count, task.max_steps)
    if error:
        return RewardBreakdown(total=0.0, correctness=0.0, efficiency=0.0,
                               step_penalty=penalty, explanation=f"Error: {error[:120]}")
    if not result:
        return RewardBreakdown(total=0.05, correctness=0.0, efficiency=0.0,
                               step_penalty=penalty, explanation="Empty result.")
    correctness = _float_unordered_score(result, task.expected_result)
    if exec_ms and exec_ms > 0 and task.baseline_exec_ms > 0 and exec_ms < task.baseline_exec_ms:
        efficiency = min((task.baseline_exec_ms / exec_ms) / 5.0, 1.0)
    else:
        efficiency = 0.0
    combined = 0.6 * correctness + 0.4 * efficiency
    total = round(min(combined * penalty, 1.0), 4)
    return RewardBreakdown(total=total, correctness=round(correctness, 4),
                           efficiency=round(efficiency, 4), step_penalty=round(penalty, 4),
                           explanation=(f"Correctness={correctness:.2f} Efficiency={efficiency:.2f} "
                                        f"exec={exec_ms:.2f}ms baseline={task.baseline_exec_ms:.2f}ms "
                                        f"step_penalty={penalty:.2f}"))


# ── Expert grader (strict multi-column match) ────────────────────────────────

def grade_expert(task, error, result, step_count):
    penalty = _step_penalty(step_count, task.max_steps)
    if error:
        return RewardBreakdown(total=0.0, correctness=0.0, step_penalty=penalty,
                               explanation=f"Error: {error[:120]}")
    if not result:
        return RewardBreakdown(total=0.05, correctness=0.0, step_penalty=penalty,
                               explanation="Empty result.")
    correctness = _float_unordered_score(result, task.expected_result)
    # Bonus for getting exact row count
    count_bonus = 0.1 if len(result) == len(task.expected_result) else 0.0
    total = round(min((correctness + count_bonus) * penalty, 1.0), 4)
    matched = round(correctness * len(task.expected_result))
    return RewardBreakdown(total=total, correctness=round(correctness, 4),
                           step_penalty=round(penalty, 4),
                           explanation=(f"Expert: {matched}/{len(task.expected_result)} rows. "
                                        f"count_bonus={count_bonus:.1f} step_penalty={penalty:.2f}"))


# ── Dispatch ─────────────────────────────────────────────────────────────────

def grade(task: Task, error, result, exec_ms, step_count) -> RewardBreakdown:
    if task.difficulty == "easy":
        return grade_easy(task, error, result, step_count)
    elif task.difficulty == "medium":
        return grade_medium(task, error, result, step_count)
    elif task.difficulty == "hard":
        return grade_hard(task, error, result, exec_ms, step_count)
    elif task.difficulty == "expert":
        return grade_expert(task, error, result, step_count)
    return RewardBreakdown(total=0.0, explanation=f"Unknown difficulty: {task.difficulty}")


__all__ = ["grade", "grade_easy", "grade_medium", "grade_hard", "grade_expert"]
