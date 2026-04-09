"""
models.py — SQL Debug Environment v3.0 — Advanced Typed Models

Enhancements over baseline:
  - QueryComplexity: classifies query structural complexity
  - PerformanceMetrics: detailed timing breakdown
  - RewardBreakdown: richer with column/row coverage scores
  - SQLObservation: episode_id exposed, performance_metrics, complexity
  - EpisodeSummary: full episode analytics returned on done=True
  - SQLState: tracks best_reward, solved flag, hint_penalty_total
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
    sql_query: str = Field(..., description="SQL query to execute.")
    reasoning: Optional[str] = Field(default=None, description="Optional chain-of-thought.")

    class Config:
        json_schema_extra = {
            "example": {
                "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
                "reasoning": "Fixed SELEC→SELECT, FORM→FROM, ORDR BY→ORDER BY",
            }
        }


# ---------------------------------------------------------------------------
# Reward breakdown — now with column_coverage and row_coverage
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed decomposition of the reward signal."""
    total: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    step_penalty: float = Field(default=1.0, ge=0.0, le=1.0)
    row_coverage: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Fraction of expected rows returned.")
    column_coverage: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Fraction of expected columns present.")
    hint_penalty: float = Field(default=0.0, ge=0.0, le=0.3,
        description="Cumulative hint penalty applied.")
    explanation: str = Field(default="")


# ---------------------------------------------------------------------------
# Query complexity classifier
# ---------------------------------------------------------------------------

class QueryComplexity(BaseModel):
    """Structural complexity of the submitted SQL query."""
    has_join: bool = False
    has_subquery: bool = False
    has_aggregation: bool = False
    has_window_function: bool = False
    has_cte: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_where: bool = False
    join_count: int = 0
    subquery_depth: int = 0
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Normalised 0-1 complexity estimate.")
    label: str = Field(default="simple",
        description="simple | moderate | complex | advanced")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

class PerformanceMetrics(BaseModel):
    """Detailed timing and query plan metrics."""
    execution_ms: float = Field(default=0.0)
    baseline_ms: float = Field(default=0.0)
    speedup_ratio: float = Field(default=0.0,
        description="baseline_ms / execution_ms. >1 means faster than baseline.")
    scan_count: int = Field(default=0)
    index_count: int = Field(default=0)
    uses_index: bool = False
    plan_steps: int = Field(default=0)
    tables_scanned: list[str] = Field(default_factory=list)
    suggestion: str = Field(default="")
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Episode summary — returned when done=True
# ---------------------------------------------------------------------------

class EpisodeSummary(BaseModel):
    """Analytics summary emitted when episode terminates."""
    episode_id: str = ""
    task_id: str = ""
    total_steps: int = 0
    final_reward: float = 0.0
    best_reward: float = 0.0
    solved: bool = False
    hints_used: int = 0
    hint_penalty_total: float = 0.0
    cumulative_reward: float = 0.0
    termination_reason: str = Field(default="",
        description="solved | max_steps | truncated")
    step_rewards: list[float] = Field(default_factory=list)
    improvement_rate: float = Field(default=0.0,
        description="Avg reward improvement per step.")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SQLObservation(BaseModel):
    """Full observation returned after reset() and each step()."""

    # Identity
    episode_id: str = Field(default="", description="UUID of current episode.")
    task_id: str = Field(..., description="Active task identifier.")

    # Task context
    task_description: str = Field(...)
    broken_query: str = Field(...)
    schema_hint: str = Field(...)

    # Execution feedback
    error_message: Optional[str] = Field(default=None)
    query_result: Optional[list[dict[str, Any]]] = Field(default=None)
    row_count: int = Field(default=0, description="Rows returned by query.")

    # Performance
    execution_time_ms: Optional[float] = Field(default=None)
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None)

    # Query intelligence
    query_analysis: Optional[PerformanceMetrics] = Field(
        default=None, description="Alias for performance_metrics (backward compat).")
    query_complexity: Optional[QueryComplexity] = Field(default=None)

    # Reward
    reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_breakdown: Optional[RewardBreakdown] = Field(default=None)

    # Multi-turn memory
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Last 5 steps with sql, error, reward, result_count.")

    # Hint system
    hint_available: bool = Field(default=True)
    hints_used: int = Field(default=0)
    hint_penalty: float = Field(default=0.0)

    # Episode state
    step_count: int = Field(default=0)
    max_steps: int = Field(default=10)
    done: bool = Field(default=False)
    best_reward_so_far: float = Field(default=0.0)

    # Episode summary (only when done=True)
    episode_summary: Optional[EpisodeSummary] = Field(default=None)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SQLState:
    """Episode-level metadata."""
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    max_steps: int = 0
    done: bool = False
    last_reward: float = 0.0
    best_reward: float = 0.0
    cumulative_reward: float = 0.0
    solved: bool = False
    hints_used: int = 0
    hint_penalty: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "done": self.done,
            "last_reward": self.last_reward,
            "best_reward": self.best_reward,
            "cumulative_reward": self.cumulative_reward,
            "solved": self.solved,
            "hints_used": self.hints_used,
            "hint_penalty": self.hint_penalty,
            **self.extra,
        }
