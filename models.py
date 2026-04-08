"""
models.py — SQL Debug Environment v3.0
All reward scores strictly within (0, 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


def _strict(v: float) -> float:
    """Clamp to strictly (0, 1) — never 0.0 or 1.0."""
    return round(max(0.001, min(float(v), 0.999)), 4)


class SQLAction(BaseModel):
    sql_query: str = Field(..., description="SQL query to execute.")
    reasoning: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
                "reasoning": "Fixed SELEC→SELECT, FORM→FROM, ORDR BY→ORDER BY",
            }
        }


class RewardBreakdown(BaseModel):
    total: float = Field(default=0.001)
    correctness: float = Field(default=0.001)
    efficiency: float = Field(default=0.001)
    step_penalty: float = Field(default=0.999)
    row_coverage: float = Field(default=0.001)
    column_coverage: float = Field(default=0.001)
    hint_penalty: float = Field(default=0.001)
    explanation: str = Field(default="")

    @field_validator(
        "total", "correctness", "efficiency", "step_penalty",
        "row_coverage", "column_coverage", "hint_penalty",
        mode="before"
    )
    @classmethod
    def clamp_strict(cls, v):
        return _strict(v)


class QueryComplexity(BaseModel):
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
    complexity_score: float = Field(default=0.001)
    label: str = Field(default="simple")


class PerformanceMetrics(BaseModel):
    execution_ms: float = Field(default=0.0)
    baseline_ms: float = Field(default=0.0)
    speedup_ratio: float = Field(default=0.0)
    scan_count: int = Field(default=0)
    index_count: int = Field(default=0)
    uses_index: bool = False
    plan_steps: int = Field(default=0)
    tables_scanned: list[str] = Field(default_factory=list)
    suggestion: str = Field(default="")
    efficiency_score: float = Field(default=0.0)


class EpisodeSummary(BaseModel):
    episode_id: str = ""
    task_id: str = ""
    total_steps: int = 0
    final_reward: float = 0.0
    best_reward: float = 0.0
    solved: bool = False
    hints_used: int = 0
    hint_penalty_total: float = 0.0
    cumulative_reward: float = 0.0
    termination_reason: str = Field(default="")
    step_rewards: list[float] = Field(default_factory=list)
    improvement_rate: float = Field(default=0.0)


class SQLObservation(BaseModel):
    episode_id: str = Field(default="")
    task_id: str = Field(...)
    task_description: str = Field(...)
    broken_query: str = Field(...)
    schema_hint: str = Field(...)
    error_message: Optional[str] = Field(default=None)
    query_result: Optional[list[dict[str, Any]]] = Field(default=None)
    row_count: int = Field(default=0)
    execution_time_ms: Optional[float] = Field(default=None)
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None)
    query_analysis: Optional[PerformanceMetrics] = Field(default=None)
    query_complexity: Optional[QueryComplexity] = Field(default=None)

    # reward is strictly (0, 1)
    reward: float = Field(default=0.001)
    reward_breakdown: Optional[RewardBreakdown] = Field(default=None)

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward(cls, v):
        return _strict(v)

    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    hint_available: bool = Field(default=True)
    hints_used: int = Field(default=0)
    hint_penalty: float = Field(default=0.0)
    step_count: int = Field(default=0)
    max_steps: int = Field(default=10)
    done: bool = Field(default=False)
    best_reward_so_far: float = Field(default=0.001)
    episode_summary: Optional[EpisodeSummary] = Field(default=None)


@dataclass
class SQLState:
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
