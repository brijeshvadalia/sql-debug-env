"""
server/environment.py — SQLDebugEnvironment

Implements the OpenEnv Environment interface:
  reset()  → SQLObservation
  step()   → SQLObservation
  state    → SQLState (property)

Uses an in-memory SQLite database seeded from server/db/seed.sql.
Thread-safe: uses a threading.Lock around all DB operations.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

from models import RewardBreakdown, SQLAction, SQLObservation, SQLState
from server.graders import grade
from server.tasks import TASKS, Task

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent / "db" / "seed.sql"
_MAX_RESULT_ROWS = 20


class SQLDebugEnvironment:
    """
    OpenEnv-compatible environment for SQL debugging and optimisation.

    An agent interacts with this environment by submitting SQL queries.
    The environment executes the queries against an in-memory SQLite database
    seeded with a realistic e-commerce schema and returns structured
    observations with reward signals.

    Tasks:
      fix_syntax_error  (easy)   — fix broken SQL keywords
      fix_logic_error   (medium) — fix wrong JOIN / aggregation
      optimize_query    (hard)   — rewrite N+1 subquery for speed
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = self._init_db()
        self._state = SQLState()
        self._current_task: Task | None = None
        self._last_reward = 0.0
        logger.info("SQLDebugEnvironment initialised.")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        """Create and seed in-memory SQLite database."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        seed_sql = _SEED_PATH.read_text()
        conn.executescript(seed_sql)
        conn.commit()
        logger.info("Database seeded from %s", _SEED_PATH)
        return conn

    def _execute_query(
        self, sql: str
    ) -> tuple[list[dict] | None, str | None, float | None]:
        """
        Execute a SQL query safely.
        Returns (rows, error_message, execution_time_ms).
        """
        with self._lock:
            try:
                t0 = time.perf_counter()
                cursor = self._conn.execute(sql)
                rows_raw = cursor.fetchmany(_MAX_RESULT_ROWS)
                exec_ms = (time.perf_counter() - t0) * 1000
                rows = [dict(row) for row in rows_raw]
                return rows, None, exec_ms
            except sqlite3.Error as exc:
                return None, str(exc), None
            except Exception as exc:  # noqa: BLE001
                return None, f"Unexpected error: {exc}", None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "fix_syntax_error") -> SQLObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: One of "fix_syntax_error", "fix_logic_error",
                     "optimize_query". Defaults to easy task.

        Returns:
            Initial SQLObservation with no execution results yet.
        """
        if task_id not in TASKS:
            valid = list(TASKS.keys())
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid options: {valid}"
            )

        self._current_task = TASKS[task_id]
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

        logger.info(
            "Episode %s started — task=%s (max_steps=%d)",
            episode_id,
            task_id,
            self._current_task.max_steps,
        )

        return SQLObservation(
            task_id=task_id,
            task_description=self._current_task.description,
            broken_query=self._current_task.broken_query,
            schema_hint=self._current_task.schema_hint,
            error_message=None,
            query_result=None,
            execution_time_ms=None,
            reward=0.0,
            reward_breakdown=None,
            step_count=0,
            max_steps=self._current_task.max_steps,
            done=False,
        )

    def step(self, action: SQLAction) -> SQLObservation:
        """
        Execute one agent action (a SQL query).

        Args:
            action: SQLAction with the SQL query to execute.

        Returns:
            SQLObservation with execution results and reward signal.

        Raises:
            RuntimeError: If reset() has not been called first.
        """
        if self._current_task is None:
            raise RuntimeError(
                "Environment not initialised. Call reset() before step()."
            )

        self._state.step_count += 1
        step = self._state.step_count

        logger.info(
            "Episode %s step %d — query: %.80s…",
            self._state.episode_id,
            step,
            action.sql_query.replace("\n", " "),
        )

        # Execute
        rows, error, exec_ms = self._execute_query(action.sql_query)

        # Grade
        reward_bd = grade(
            task=self._current_task,
            error=error,
            result=rows,
            exec_ms=exec_ms,
            step_count=step,
        )

        # Episode termination
        done = (
            reward_bd.total >= self._current_task.reward_threshold
            or step >= self._current_task.max_steps
        )

        # Update state
        self._state.done = done
        self._state.last_reward = reward_bd.total
        self._state.cumulative_reward += reward_bd.total

        logger.info(
            "Step %d: reward=%.4f done=%s error=%s",
            step,
            reward_bd.total,
            done,
            bool(error),
        )

        return SQLObservation(
            task_id=self._current_task.task_id,
            task_description=self._current_task.description,
            broken_query=self._current_task.broken_query,
            schema_hint=self._current_task.schema_hint,
            error_message=error,
            query_result=rows,
            execution_time_ms=exec_ms,
            reward=reward_bd.total,
            reward_breakdown=reward_bd,
            step_count=step,
            max_steps=self._current_task.max_steps,
            done=done,
        )

    @property
    def state(self) -> SQLState:
        """Return current episode metadata."""
        return self._state

    def get_metadata(self) -> dict:
        """Return environment metadata (used by create_fastapi_app)."""
        return {
            "name": "sql-debug-env",
            "version": "1.0.0",
            "description": (
                "Real-world SQL debugging and optimisation environment. "
                "AI agents fix broken queries and rewrite slow queries."
            ),
            "tasks": list(TASKS.keys()),
            "action_fields": ["sql_query", "reasoning"],
        }

    def close(self) -> None:
        """Release database resources."""
        with self._lock:
            self._conn.close()
        logger.info("SQLDebugEnvironment closed.")
