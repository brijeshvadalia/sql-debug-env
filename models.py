"""
models.py — Typed Pydantic models for the SQL Debug Environment v3.0

Advanced features added:
  - conversation_history: multi-turn memory in every observation
  - query_analysis: EXPLAIN plan summary
  - hint_available / hints_used: hint system state
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SQLAction(BaseModel):
    """Action submitted by the agent each step."""

    sql_query: str = Field(
        ...,
        description="SQL query to execute against the environment database.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought reasoning (logged, not evaluated).",
    )

    def __init__(self, **data: Any) -> None:
        data.setdefault("reasoning", None)
        super().__init__(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
                "reasoning": "Fixed all three typos in the keywords.",
            }
        }


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed decomposition of the reward signal."""

    total: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    step_penalty: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str = Field(default="")

    def __init__(self, **data: Any) -> None:
        data.setdefault("total", 0.0)
        data.setdefault("correctness", 0.0)
        data.setdefault("efficiency", 0.0)
        data.setdefault("step_penalty", 1.0)
        data.setdefault("explanation", "")
        super().__init__(**data)


# ---------------------------------------------------------------------------
# Query analysis (EXPLAIN output)
# ---------------------------------------------------------------------------

class QueryAnalysis(BaseModel):
    """Structured summary of EXPLAIN QUERY PLAN output."""

    scan_count: int = Field(default=0, description="Number of full table scans.")
    uses_index: bool = Field(default=False, description="Whether any index is used.")
    tables_scanned: list[str] = Field(default_factory=list)
    plan_steps: int = Field(default=0)
    suggestion: str = Field(default="", description="Performance suggestion.")

    def __init__(self, **data: Any) -> None:
        data.setdefault("scan_count", 0)
        data.setdefault("uses_index", False)
        data.setdefault("tables_scanned", [])
        data.setdefault("plan_steps", 0)
        data.setdefault("suggestion", "")
        super().__init__(**data)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SQLObservation(BaseModel):
    """
    Observation returned by the environment after reset() and each step().

    Advanced fields:
      conversation_history : last 5 turns with SQL, error, reward
      query_analysis       : EXPLAIN plan summary
      hint_available       : whether hints remain unused
      hints_used           : how many hints taken this episode
    """

    # Task context
    task_id: str = Field(..., description="Active task identifier.")
    task_description: str = Field(..., description="Full task description.")
    broken_query: str = Field(..., description="Original broken/slow query.")
    schema_hint: str = Field(..., description="DDL + sample data context.")

    # Execution feedback
    error_message: Optional[str] = Field(default=None)
    query_result: Optional[list[dict[str, Any]]] = Field(default=None)
    execution_time_ms: Optional[float] = Field(default=None)

    # Reward
    reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_breakdown: Optional[RewardBreakdown] = Field(default=None)

    # Advanced: multi-turn memory
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Last 5 steps: sql, error, reward, result_count.",
    )

    # Advanced: query analysis
    query_analysis: Optional[QueryAnalysis] = Field(
        default=None,
        description="EXPLAIN QUERY PLAN summary.",
    )

    # Advanced: hint system
    hint_available: bool = Field(default=True)
    hints_used: int = Field(default=0)

    # Episode metadata
    step_count: int = Field(default=0)
    max_steps: int = Field(default=10)
    done: bool = Field(default=False)

    def __init__(self, **data: Any) -> None:
        data.setdefault("error_message", None)
        data.setdefault("query_result", None)
        data.setdefault("execution_time_ms", None)
        data.setdefault("reward", 0.0)
        data.setdefault("reward_breakdown", None)
        data.setdefault("conversation_history", [])
        data.setdefault("query_analysis", None)
        data.setdefault("hint_available", True)
        data.setdefault("hints_used", 0)
        data.setdefault("step_count", 0)
        data.setdefault("done", False)
        data.setdefault("max_steps", 10)
        super().__init__(**data)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SQLState:
    """Episode-level metadata returned by state() property."""
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
