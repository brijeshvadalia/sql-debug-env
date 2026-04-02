"""
server/environment.py — SQLDebugEnvironment

Advanced OpenEnv-compatible environment featuring:
  - 8 real-world SQL debugging tasks (easy/medium/hard/expert)
  - Multi-turn conversation memory (full trajectory history)
  - Dynamic hint system (3 levels, reward-penalised)
  - EXPLAIN query analysis (scan count, index usage hints)
  - Curriculum auto-advancement (easy->medium->hard->expert)
  - WebSocket-ready state management
  - Thread-safe SQLite execution
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4
from typing import Any, Optional

from models import RewardBreakdown, SQLAction, SQLObservation, SQLState
from server.graders import grade
from server.tasks import TASKS, Task

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent / "db" / "seed.sql"
_MAX_RESULT_ROWS = 20

# ---------------------------------------------------------------------------
# Curriculum order
# ---------------------------------------------------------------------------

CURRICULUM_ORDER = [
    "fix_syntax_error",       # easy
    "fix_logic_error",        # medium
    "fix_null_handling",      # medium
    "fix_subquery_bug",       # medium
    "optimize_query",         # hard
    "fix_window_function",    # hard
    "fix_cte",                # hard
    "multi_step_aggregation", # expert
]

# ---------------------------------------------------------------------------
# Hint system — 3 levels per task
# ---------------------------------------------------------------------------

HINTS: dict[str, list[str]] = {
    "fix_syntax_error": [
        "Level 1: Look carefully at the SQL keywords — some are misspelled.",
        "Level 2: Three keywords are wrong: the SELECT keyword, the FROM keyword, and the ORDER BY keyword.",
        "Level 3: Fix: 'SELEC' -> 'SELECT', 'FORM' -> 'FROM', 'ORDR BY' -> 'ORDER BY'",
    ],
    "fix_logic_error": [
        "Level 1: The JOIN type causes some orders to be excluded from results.",
        "Level 2: Change INNER JOIN order_items to LEFT JOIN, and fix the SUM column.",
        "Level 3: Use LEFT JOIN order_items and SUM(oi.quantity * oi.unit_price) instead of SUM(o.total_amount)",
    ],
    "fix_null_handling": [
        "Level 1: Products with no reviews are being excluded entirely.",
        "Level 2: Change INNER JOIN to LEFT JOIN, and handle NULL from AVG().",
        "Level 3: Use LEFT JOIN reviews, and wrap AVG with COALESCE(AVG(r.rating), 0.0)",
    ],
    "fix_subquery_bug": [
        "Level 1: There is an alias conflict between the outer and inner query.",
        "Level 2: The inner subquery reuses alias 'oi' which shadows the outer query's 'oi'.",
        "Level 3: Rename the inner JOIN alias from 'oi' to 'oi2': JOIN order_items oi2 ON oi2.order_id = ord.id",
    ],
    "optimize_query": [
        "Level 1: The query uses correlated subqueries — one for each customer row.",
        "Level 2: Replace both subqueries with LEFT JOINs and GROUP BY.",
        "Level 3: Use LEFT JOIN orders ON ... AND o.status != 'cancelled', LEFT JOIN order_items, GROUP BY c.id, use COALESCE(SUM(...),0) and COUNT(DISTINCT o.id)",
    ],
    "fix_window_function": [
        "Level 1: The window function uses the wrong ranking function and wrong partition column.",
        "Level 2: ROW_NUMBER() should be RANK(), and PARTITION BY should use region not tier.",
        "Level 3: Change to RANK() OVER (PARTITION BY c.region ORDER BY SUM(...) DESC)",
    ],
    "fix_cte": [
        "Level 1: The percentage calculation always returns 100% for every row.",
        "Level 2: You are dividing total_revenue by itself instead of the grand total.",
        "Level 3: Change the divisor to: (SELECT SUM(total_revenue) FROM customer_revenue)",
    ],
    "multi_step_aggregation": [
        "Level 1: The query is missing a GROUP BY clause and the average order value is hardcoded as 0.",
        "Level 2: Add GROUP BY p.category, c.tier and compute the real avg_order_value.",
        "Level 3: GROUP BY p.category, c.tier and use ROUND(SUM(oi.quantity*oi.unit_price)/COUNT(DISTINCT o.id),2) AS avg_order_value",
    ],
}


# ---------------------------------------------------------------------------
# EXPLAIN analysis
# ---------------------------------------------------------------------------

def _analyse_query(conn: sqlite3.Connection, sql: str) -> dict[str, Any]:
    """
    Run EXPLAIN QUERY PLAN and return structured analysis.
    Returns dict with: scan_count, uses_index, tables_scanned, suggestion.
    """
    try:
        plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        plans = [dict(r) for r in plan_rows]
        details = [str(p.get("detail","")) for p in plans]
        full = " ".join(details).upper()

        scan_count = sum(1 for d in details if "SCAN" in d.upper())
        index_count = sum(1 for d in details if "INDEX" in d.upper() or "USING INDEX" in d.upper())
        tables = [d.split("SCAN")[-1].strip().split()[0] if "SCAN" in d.upper() else "" for d in details]
        tables = [t for t in tables if t]

        if "CORRELATED" in full or scan_count > 3:
            suggestion = "High scan count detected. Consider rewriting subqueries as JOINs."
        elif scan_count == 0:
            suggestion = "Query plan looks efficient — using index lookups."
        elif index_count > 0:
            suggestion = "Some index usage detected. Query looks reasonable."
        else:
            suggestion = f"Full table scans on {scan_count} table(s). Consider adding WHERE clauses."

        return {
            "scan_count": scan_count,
            "uses_index": index_count > 0,
            "tables_scanned": tables[:5],
            "plan_steps": len(plans),
            "suggestion": suggestion,
        }
    except Exception:
        return {
            "scan_count": 0,
            "uses_index": False,
            "tables_scanned": [],
            "plan_steps": 0,
            "suggestion": "Could not analyse query plan.",
        }


# ---------------------------------------------------------------------------
# Conversation turn (for multi-turn memory)
# ---------------------------------------------------------------------------

class ConversationTurn:
    def __init__(self, step: int, sql: str, reasoning: Optional[str],
                 error: Optional[str], reward: float, done: bool,
                 result_count: int, exec_ms: Optional[float]):
        self.step = step
        self.sql = sql
        self.reasoning = reasoning
        self.error = error
        self.reward = reward
        self.done = done
        self.result_count = result_count
        self.exec_ms = exec_ms

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "sql_query": self.sql,
            "reasoning": self.reasoning,
            "error": self.error,
            "reward": self.reward,
            "done": self.done,
            "result_count": self.result_count,
            "exec_ms": round(self.exec_ms, 3) if self.exec_ms else None,
        }

    def to_context_str(self) -> str:
        """Human-readable summary for including in next observation."""
        parts = [f"Step {self.step}: reward={self.reward:.3f}"]
        if self.error:
            parts.append(f"error='{self.error[:80]}'")
        else:
            parts.append(f"rows_returned={self.result_count}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLDebugEnvironment:
    """
    Advanced OpenEnv-compatible SQL debugging environment.

    Features:
      - 8 tasks across easy/medium/hard/expert difficulties
      - Multi-turn memory: agent sees full trajectory history
      - 3-level hint system with reward penalty
      - EXPLAIN query plan analysis
      - Curriculum auto-advancement
      - Thread-safe SQLite execution
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = self._init_db()
        self._state = SQLState()
        self._current_task: Optional[Task] = None

        # Multi-turn memory
        self._conversation: list[ConversationTurn] = []

        # Hint tracking
        self._hints_used: int = 0
        self._hint_penalty: float = 0.0

        # Curriculum tracking
        self._curriculum_scores: dict[str, list[float]] = {t: [] for t in CURRICULUM_ORDER}
        self._curriculum_index: int = 0

        # Baseline exec times for optimize tasks (populated on first run)
        self._baseline_times: dict[str, float] = {}

        logger.info("SQLDebugEnvironment initialised.")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SEED_PATH.read_text())
        conn.commit()
        logger.info("Database seeded from %s", _SEED_PATH)
        return conn

    def _execute_query(self, sql: str) -> tuple[Optional[list[dict]], Optional[str], Optional[float]]:
        with self._lock:
            try:
                t0 = time.perf_counter()
                cursor = self._conn.execute(sql)
                rows = [dict(r) for r in cursor.fetchmany(_MAX_RESULT_ROWS)]
                exec_ms = (time.perf_counter() - t0) * 1000
                return rows, None, exec_ms
            except sqlite3.Error as exc:
                return None, str(exc), None
            except Exception as exc:
                return None, f"Unexpected error: {exc}", None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "fix_syntax_error") -> SQLObservation:
        """
        Reset environment for a new episode.
        Clears conversation history and hint state.
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")

        self._current_task = TASKS[task_id]
        self._conversation = []
        self._hints_used = 0
        self._hint_penalty = 0.0

        episode_id = str(uuid4())
        self._state = SQLState(
            episode_id=episode_id,
            step_count=0,
            task_id=task_id,
            max_steps=self._current_task.max_steps,
            done=False,
            last_reward=0.0,
            cumulative_reward=0.0,
        )

        logger.info("Episode %s started — task=%s", episode_id[:8], task_id)

        return SQLObservation(
            task_id=task_id,
            task_description=self._current_task.description,
            broken_query=self._current_task.broken_query,
            schema_hint=self._current_task.schema_hint,
            conversation_history=[],
            query_analysis=None,
            hint_available=True,
            hints_used=0,
            max_steps=self._current_task.max_steps,
        )

    def step(self, action: SQLAction) -> SQLObservation:
        """Execute one agent action (a SQL query)."""
        if self._current_task is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        step = self._state.step_count

        # Execute query
        rows, error, exec_ms = self._execute_query(action.sql_query)

        # Update baseline exec time for hard tasks
        task_id = self._current_task.task_id
        if self._current_task.baseline_exec_ms > 0 and exec_ms:
            if task_id not in self._baseline_times:
                self._baseline_times[task_id] = self._current_task.baseline_exec_ms

        # Grade
        reward_bd = grade(
            task=self._current_task,
            error=error,
            result=rows,
            exec_ms=exec_ms,
            step_count=step,
        )

        # Apply hint penalty (max 30% penalty)
        final_reward = max(0.0, round(reward_bd.total * (1.0 - self._hint_penalty), 4))
        reward_bd.total = final_reward
        if self._hint_penalty > 0:
            reward_bd.explanation += f" | hint_penalty={self._hint_penalty:.0%}"

        # Query analysis (EXPLAIN)
        query_analysis = None
        if not error:
            query_analysis = _analyse_query(self._conn, action.sql_query)

        # Update conversation memory
        turn = ConversationTurn(
            step=step,
            sql=action.sql_query,
            reasoning=action.reasoning,
            error=error,
            reward=final_reward,
            done=False,
            result_count=len(rows) if rows else 0,
            exec_ms=exec_ms,
        )

        # Episode termination
        done = (
            final_reward >= self._current_task.reward_threshold
            or step >= self._current_task.max_steps
        )
        turn.done = done
        self._conversation.append(turn)

        # Update state
        self._state.done = done
        self._state.last_reward = final_reward
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + final_reward, 4
        )

        # Update curriculum scores
        if done and task_id in self._curriculum_scores:
            self._curriculum_scores[task_id].append(final_reward)

        # Build conversation history for observation (last 5 turns)
        conv_history = [t.to_dict() for t in self._conversation[-5:]]

        logger.info("Step %d: reward=%.4f done=%s", step, final_reward, done)

        return SQLObservation(
            task_id=task_id,
            task_description=self._current_task.description,
            broken_query=self._current_task.broken_query,
            schema_hint=self._current_task.schema_hint,
            error_message=error,
            query_result=rows,
            execution_time_ms=exec_ms,
            reward=final_reward,
            reward_breakdown=reward_bd,
            conversation_history=conv_history,
            query_analysis=query_analysis,
            hint_available=self._hints_used < 3,
            hints_used=self._hints_used,
            step_count=step,
            max_steps=self._current_task.max_steps,
            done=done,
        )

    @property
    def state(self) -> SQLState:
        return self._state

    # ------------------------------------------------------------------
    # Hint system
    # ------------------------------------------------------------------

    def get_hint(self) -> dict[str, Any]:
        """
        Return the next hint for the current task.
        Each hint costs a 10% reward penalty (max 3 hints = 30% penalty).
        """
        if self._current_task is None:
            raise RuntimeError("Call reset() before get_hint().")

        task_id = self._current_task.task_id
        hints = HINTS.get(task_id, [])

        if self._hints_used >= len(hints):
            return {
                "hint": "No more hints available for this task.",
                "level": self._hints_used,
                "penalty_applied": 0.0,
                "total_penalty": self._hint_penalty,
                "hints_remaining": 0,
            }

        hint_text = hints[self._hints_used]
        self._hints_used += 1
        self._hint_penalty = min(0.30, self._hints_used * 0.10)

        return {
            "hint": hint_text,
            "level": self._hints_used,
            "penalty_applied": 0.10,
            "total_penalty": self._hint_penalty,
            "hints_remaining": max(0, 3 - self._hints_used),
            "message": f"Hint {self._hints_used}/3 used. Future rewards penalised by {self._hint_penalty:.0%}.",
        }

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def curriculum_next(self) -> dict[str, Any]:
        """
        Return the next recommended task based on mastery scores.
        Advances when avg score >= 0.8 over last 3 episodes for current task.
        """
        if self._curriculum_index >= len(CURRICULUM_ORDER):
            return {
                "status": "complete",
                "message": "All curriculum tasks mastered!",
                "recommended_task": CURRICULUM_ORDER[-1],
                "progress": f"{len(CURRICULUM_ORDER)}/{len(CURRICULUM_ORDER)}",
            }

        current_tid = CURRICULUM_ORDER[self._curriculum_index]
        scores = self._curriculum_scores.get(current_tid, [])
        recent = scores[-3:]
        avg = sum(recent) / len(recent) if recent else 0.0

        # Check if should advance
        should_advance = len(recent) >= 3 and avg >= 0.8
        if should_advance and self._curriculum_index < len(CURRICULUM_ORDER) - 1:
            self._curriculum_index += 1
            next_tid = CURRICULUM_ORDER[self._curriculum_index]
            return {
                "status": "advanced",
                "message": f"Mastered '{current_tid}' (avg={avg:.2f}). Advancing!",
                "recommended_task": next_tid,
                "previous_task": current_tid,
                "progress": f"{self._curriculum_index+1}/{len(CURRICULUM_ORDER)}",
                "mastery_score": round(avg, 3),
            }

        # Recommend same task
        task = TASKS[current_tid]
        return {
            "status": "in_progress",
            "recommended_task": current_tid,
            "difficulty": task.difficulty,
            "episodes_on_task": len(scores),
            "recent_avg": round(avg, 3),
            "needed_avg": 0.8,
            "progress": f"{self._curriculum_index+1}/{len(CURRICULUM_ORDER)}",
            "message": (
                f"Keep practising '{current_tid}'. "
                f"Recent avg: {avg:.2f}/0.80 needed. "
                f"({len(recent)}/3 recent episodes counted)"
            ),
        }

    def curriculum_status(self) -> dict[str, Any]:
        """Full curriculum progress report."""
        result = []
        for i, tid in enumerate(CURRICULUM_ORDER):
            scores = self._curriculum_scores.get(tid, [])
            recent = scores[-3:]
            avg = round(sum(recent)/len(recent), 3) if recent else 0.0
            mastered = len(recent) >= 3 and avg >= 0.8
            result.append({
                "task_id": tid,
                "difficulty": TASKS[tid].difficulty,
                "episodes": len(scores),
                "recent_avg": avg,
                "mastered": mastered,
                "current": i == self._curriculum_index,
            })
        return {
            "progress": f"{self._curriculum_index+1}/{len(CURRICULUM_ORDER)}",
            "current_task": CURRICULUM_ORDER[self._curriculum_index],
            "tasks": result,
        }

    # ------------------------------------------------------------------
    # History & metadata
    # ------------------------------------------------------------------

    def get_history(self) -> dict[str, Any]:
        """Full step-by-step conversation history for current episode."""
        return {
            "episode_id": self._state.episode_id,
            "task_id": self._state.task_id,
            "steps": [t.to_dict() for t in self._conversation],
            "total_steps": len(self._conversation),
            "cumulative_reward": self._state.cumulative_reward,
            "hints_used": self._hints_used,
            "hint_penalty": self._hint_penalty,
        }

    def get_metadata(self) -> dict:
        return {
            "name": "sql-debug-env",
            "version": "3.0.0",
            "description": "Advanced real-world SQL debugging and optimisation environment.",
            "tasks": list(TASKS.keys()),
            "features": [
                "multi_turn_memory",
                "hint_system",
                "query_analysis",
                "curriculum_learning",
                "8_tasks",
                "4_difficulty_levels",
            ],
            "action_fields": ["sql_query", "reasoning"],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("SQLDebugEnvironment closed.")
