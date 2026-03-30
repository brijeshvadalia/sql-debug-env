"""
models.py — Typed Pydantic models for the SQL Debug Environment.

Defines the three core model types required by OpenEnv:
  - SQLAction     → what the agent sends
  - SQLObservation → what the environment returns
  - SQLState      → episode metadata (returned by state())

All fields are explicitly typed and documented for OpenEnv spec compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SQLAction(BaseModel):
    """
    Action submitted by the agent each step.

    The agent must provide a SQL query to execute against the environment's
    SQLite database. Optionally, it may provide chain-of-thought reasoning
    which is logged but not evaluated.
    """

    sql_query: str = Field(
        ...,
        description="SQL query to execute against the environment database.",
        examples=[
            "SELECT id, name FROM customers WHERE region = 'EU' ORDER BY name;",
            "SELECT o.id, SUM(oi.quantity * oi.unit_price) AS total FROM orders o "
            "JOIN order_items oi ON o.id = oi.order_id GROUP BY o.id;",
        ],
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought reasoning (logged, not evaluated).",
    )

    def __init__(self, **data) -> None:
        data.setdefault("reasoning", None)
        super().__init__(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "sql_query": "SELECT id, name FROM customers ORDER BY name;",
                "reasoning": "The original query had a missing semicolon.",
            }
        }


# ---------------------------------------------------------------------------
# Reward breakdown (structured for interpretability)
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed decomposition of the reward signal."""

    correctness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of expected rows correctly returned (0.0–1.0).",
    )
    efficiency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Query efficiency score (only non-zero for optimize_query task).",
    )
    step_penalty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Multiplier penalising wasted steps (1.0 = no penalty).",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the reward components.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SQLObservation(BaseModel):
    """
    Observation returned by the environment after reset() and each step().

    Contains everything the agent needs to reason about its next action:
    the task description, the broken query, schema context, the result of
    its last SQL execution, and the current reward.
    """

    # ── Task context ────────────────────────────────────────────────────────
    task_id: str = Field(
        ...,
        description="Identifier of the active task.",
        examples=["fix_syntax_error", "fix_logic_error", "optimize_query"],
    )
    task_description: str = Field(
        ...,
        description="Full description of what the agent must accomplish.",
    )
    broken_query: str = Field(
        ...,
        description="The original broken or slow query the agent must fix.",
    )
    schema_hint: str = Field(
        ...,
        description="DDL snippet + example rows for relevant tables.",
    )

    # ── Execution feedback ──────────────────────────────────────────────────
    error_message: Optional[str] = Field(
        default=None,
        description="SQLite error from the last query execution, if any.",
    )
    query_result: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Up to 20 rows from the last query execution.",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Wall-clock execution time of the last query (ms).",
    )

    # ── Reward signal ───────────────────────────────────────────────────────
    reward: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Scalar reward for the last action.",
    )
    reward_breakdown: Optional[RewardBreakdown] = Field(
        default=None,
        description="Detailed reward decomposition.",
    )

    # ── Episode metadata ────────────────────────────────────────────────────
    step_count: int = Field(default=0, description="Steps taken this episode.")
    max_steps: int = Field(default=10, description="Maximum steps before truncation.")
    done: bool = Field(default=False, description="Whether episode has ended.")

    def __init__(self, **data: Any) -> None:
        # Explicit defaults — works with Pydantic v1, v2, and test stubs
        data.setdefault("error_message", None)
        data.setdefault("query_result", None)
        data.setdefault("execution_time_ms", None)
        data.setdefault("reward", 0.0)
        data.setdefault("reward_breakdown", None)
        data.setdefault("step_count", 0)
        data.setdefault("done", False)
        data.setdefault("max_steps", 10)
        super().__init__(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "fix_syntax_error",
                "task_description": "Fix the syntax error in the SQL query.",
                "broken_query": "SELEC id, name FORM customers;",
                "schema_hint": "TABLE customers(id INTEGER, name TEXT, region TEXT)",
                "error_message": 'near "FORM": syntax error',
                "query_result": None,
                "execution_time_ms": None,
                "reward": 0.0,
                "reward_breakdown": None,
                "step_count": 1,
                "max_steps": 5,
                "done": False,
            }
        }


# ---------------------------------------------------------------------------
# State (episode metadata — returned by state())
# ---------------------------------------------------------------------------

@dataclass
class SQLState:
    """
    Episode-level metadata returned by the environment's state() property.

    OpenEnv requires state() to return current episode info including
    episode_id and step_count at minimum.
    """

    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    max_steps: int = 0
    done: bool = False
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "done": self.done,
            "last_reward": self.last_reward,
            "cumulative_reward": self.cumulative_reward,
            **self.extra,
        }
