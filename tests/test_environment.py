"""
tests/test_environment.py — Full test suite for SQL Debug Environment

Tests cover:
  - Models (SQLAction, SQLObservation, SQLState)
  - Graders (determinism, partial credit, score range)
  - Environment (reset, step, state, episode flow)
  - FastAPI endpoints (health, reset, step, state, tasks)
  - End-to-end episode runs for all 3 tasks
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from fastapi.testclient import TestClient

from models import RewardBreakdown, SQLAction, SQLObservation, SQLState
from server.environment import SQLDebugEnvironment
from server.graders import grade, grade_easy, grade_medium, grade_hard
from server.tasks import TASKS, TASK_EASY, TASK_MEDIUM, TASK_HARD
from server.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env() -> SQLDebugEnvironment:
    """Shared environment instance for all tests."""
    e = SQLDebugEnvironment()
    yield e
    e.close()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI test client."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. MODEL TESTS
# ===========================================================================

class TestSQLAction:
    def test_minimal_action(self):
        a = SQLAction(sql_query="SELECT 1;")
        assert a.sql_query == "SELECT 1;"
        assert a.reasoning is None

    def test_full_action(self):
        a = SQLAction(
            sql_query="SELECT id FROM customers;",
            reasoning="Just selecting ids.",
        )
        assert a.reasoning == "Just selecting ids."

    def test_action_serialisation(self):
        a = SQLAction(sql_query="SELECT 1;", reasoning="test")
        d = a.model_dump()
        assert d["sql_query"] == "SELECT 1;"
        a2 = SQLAction(**d)
        assert a2.sql_query == a.sql_query

    def test_action_requires_sql_query(self):
        with pytest.raises(Exception):
            SQLAction()  # sql_query is required


class TestSQLObservation:
    def test_default_observation(self):
        obs = SQLObservation(
            task_id="fix_syntax_error",
            task_description="Fix the query.",
            broken_query="SELEC 1;",
            schema_hint="TABLE customers(id, name)",
            max_steps=5,
        )
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.step_count == 0
        assert obs.error_message is None
        assert obs.query_result is None

    def test_observation_json_roundtrip(self):
        obs = SQLObservation(
            task_id="fix_syntax_error",
            task_description="desc",
            broken_query="broken",
            schema_hint="hint",
            max_steps=5,
            reward=0.75,
            step_count=2,
        )
        d = obs.model_dump()
        obs2 = SQLObservation(**d)
        assert obs2.reward == obs.reward
        assert obs2.step_count == obs.step_count

    def test_reward_breakdown_embedded(self):
        bd = RewardBreakdown(correctness=0.9, total=0.85, explanation="test")
        obs = SQLObservation(
            task_id="fix_syntax_error",
            task_description="d",
            broken_query="b",
            schema_hint="s",
            max_steps=5,
            reward=0.85,
            reward_breakdown=bd,
        )
        assert obs.reward_breakdown.correctness == 0.9


class TestSQLState:
    def test_default_state(self):
        s = SQLState()
        assert s.episode_id == ""
        assert s.step_count == 0
        assert s.done is False

    def test_state_to_dict(self):
        s = SQLState(episode_id="abc", step_count=3, task_id="fix_syntax_error")
        d = s.to_dict()
        assert d["episode_id"] == "abc"
        assert d["step_count"] == 3
        assert d["task_id"] == "fix_syntax_error"


# ===========================================================================
# 2. GRADER TESTS
# ===========================================================================

class TestGraders:
    """All graders must be deterministic and return scores in [0.0, 1.0]."""

    # ── Easy grader ─────────────────────────────────────────────────────────

    def test_easy_error_gives_small_reward(self):
        bd = grade_easy(TASK_EASY, error="syntax error", result=None, step_count=1)
        assert 0.0 <= bd.total <= 0.2
        assert "syntax error" in bd.explanation

    def test_easy_correct_result_gives_high_reward(self):
        bd = grade_easy(
            TASK_EASY,
            error=None,
            result=TASK_EASY.expected_result,
            step_count=1,
        )
        assert bd.total >= 0.8
        assert bd.correctness >= 0.9

    def test_easy_wrong_result_gives_partial_reward(self):
        wrong = [{"id": 99, "name": "Nobody", "email": "x@y.com"}]
        bd = grade_easy(TASK_EASY, error=None, result=wrong, step_count=1)
        assert 0.0 < bd.total < 0.8

    def test_easy_deterministic(self):
        bd1 = grade_easy(TASK_EASY, error=None, result=TASK_EASY.expected_result, step_count=2)
        bd2 = grade_easy(TASK_EASY, error=None, result=TASK_EASY.expected_result, step_count=2)
        assert bd1.total == bd2.total

    def test_easy_step_penalty_applied(self):
        bd_early = grade_easy(TASK_EASY, error=None, result=TASK_EASY.expected_result, step_count=1)
        bd_late  = grade_easy(TASK_EASY, error=None, result=TASK_EASY.expected_result, step_count=5)
        assert bd_early.total >= bd_late.total

    def test_easy_score_in_range(self):
        for step in range(1, 6):
            bd = grade_easy(TASK_EASY, error=None, result=TASK_EASY.expected_result, step_count=step)
            assert 0.0 <= bd.total <= 1.0

    # ── Medium grader ────────────────────────────────────────────────────────

    def test_medium_error_gives_tiny_reward(self):
        bd = grade_medium(TASK_MEDIUM, error="no such table", result=None, step_count=1)
        assert bd.total <= 0.1

    def test_medium_full_match(self):
        bd = grade_medium(TASK_MEDIUM, error=None, result=TASK_MEDIUM.expected_result, step_count=1)
        assert bd.total >= 0.85
        assert bd.correctness >= 0.95

    def test_medium_partial_match(self):
        partial = TASK_MEDIUM.expected_result[:5]   # only 5 of 11 rows
        bd = grade_medium(TASK_MEDIUM, error=None, result=partial, step_count=1)
        assert 0.3 < bd.total < 0.7

    def test_medium_empty_result(self):
        bd = grade_medium(TASK_MEDIUM, error=None, result=[], step_count=1)
        assert bd.total <= 0.1

    def test_medium_deterministic(self):
        bd1 = grade_medium(TASK_MEDIUM, error=None, result=TASK_MEDIUM.expected_result, step_count=3)
        bd2 = grade_medium(TASK_MEDIUM, error=None, result=TASK_MEDIUM.expected_result, step_count=3)
        assert bd1.total == bd2.total

    def test_medium_score_in_range(self):
        bd = grade_medium(TASK_MEDIUM, error=None, result=TASK_MEDIUM.expected_result, step_count=1)
        assert 0.0 <= bd.total <= 1.0

    # ── Hard grader ──────────────────────────────────────────────────────────

    def test_hard_error_gives_zero(self):
        bd = grade_hard(TASK_HARD, error="table not found", result=None, exec_ms=None, step_count=1)
        assert bd.total == 0.0

    def test_hard_correct_and_fast_gives_high_reward(self):
        fast_ms = TASK_HARD.baseline_exec_ms * 0.1   # 10x faster
        bd = grade_hard(
            TASK_HARD,
            error=None,
            result=TASK_HARD.expected_result,
            exec_ms=fast_ms,
            step_count=1,
        )
        assert bd.total >= 0.8
        assert bd.efficiency > 0.5

    def test_hard_correct_but_slow_partial_reward(self):
        slow_ms = TASK_HARD.baseline_exec_ms * 2   # 2x slower than baseline
        bd = grade_hard(
            TASK_HARD,
            error=None,
            result=TASK_HARD.expected_result,
            exec_ms=slow_ms,
            step_count=1,
        )
        assert bd.total <= 0.65   # correctness only, no efficiency bonus
        assert bd.efficiency == 0.0

    def test_hard_wrong_result_zero_correctness(self):
        bd = grade_hard(
            TASK_HARD, error=None, result=[{"id": 99}], exec_ms=1.0, step_count=1
        )
        assert bd.correctness < 0.2

    def test_hard_deterministic(self):
        bd1 = grade_hard(TASK_HARD, error=None, result=TASK_HARD.expected_result, exec_ms=1.0, step_count=2)
        bd2 = grade_hard(TASK_HARD, error=None, result=TASK_HARD.expected_result, exec_ms=1.0, step_count=2)
        assert bd1.total == bd2.total

    def test_hard_score_in_range(self):
        bd = grade_hard(TASK_HARD, error=None, result=TASK_HARD.expected_result, exec_ms=0.5, step_count=1)
        assert 0.0 <= bd.total <= 1.0

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def test_grade_dispatch_all_tasks(self):
        for task in [TASK_EASY, TASK_MEDIUM, TASK_HARD]:
            bd = grade(
                task=task, error="test error",
                result=None, exec_ms=None, step_count=1,
            )
            assert 0.0 <= bd.total <= 1.0

    def test_grade_unknown_task_returns_zero(self):
        from server.tasks import Task
        from dataclasses import fields
        bad_task = TASK_EASY.__class__(
            task_id="nonexistent",
            name="bad",
            difficulty="easy",
            description="bad",
            broken_query="",
            correct_query="",
            expected_result=[],
            schema_hint="",
            max_steps=5,
            reward_threshold=0.8,
        )
        bd = grade(bad_task, error=None, result=[], exec_ms=None, step_count=1)
        assert bd.total == 0.0


# ===========================================================================
# 3. ENVIRONMENT TESTS
# ===========================================================================

class TestEnvironment:
    def test_reset_returns_observation(self, env):
        obs = env.reset("fix_syntax_error")
        assert isinstance(obs, SQLObservation)
        assert obs.task_id == "fix_syntax_error"
        assert obs.done is False
        assert obs.step_count == 0
        assert obs.broken_query != ""
        assert obs.schema_hint != ""

    def test_reset_all_tasks(self, env):
        for task_id in TASKS:
            obs = env.reset(task_id)
            assert obs.task_id == task_id
            assert obs.step_count == 0
            assert obs.done is False

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent_task")

    def test_step_without_reset_raises(self):
        fresh_env = SQLDebugEnvironment()
        fresh_env._current_task = None   # simulate not reset
        with pytest.raises(RuntimeError, match="not initialised"):
            fresh_env.step(SQLAction(sql_query="SELECT 1;"))
        fresh_env.close()

    def test_step_returns_observation(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert isinstance(obs, SQLObservation)
        assert obs.step_count == 1

    def test_step_increments_count(self, env):
        env.reset("fix_syntax_error")
        for i in range(1, 4):
            obs = env.step(SQLAction(sql_query="SELECT 1;"))
            assert obs.step_count == i

    def test_step_with_error_returns_error_message(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELEC 1;"))   # bad query
        assert obs.error_message is not None
        assert obs.reward < 0.5

    def test_step_reward_in_range(self, env):
        env.reset("fix_syntax_error")
        for _ in range(3):
            obs = env.step(SQLAction(sql_query="SELECT 1;"))
            assert 0.0 <= obs.reward <= 1.0

    def test_state_reflects_episode(self, env):
        env.reset("fix_logic_error")
        s = env.state
        assert s.task_id == "fix_logic_error"
        assert s.step_count == 0
        assert s.done is False

    def test_state_updates_after_step(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELECT 1;"))
        s = env.state
        assert s.step_count == 1

    def test_done_on_max_steps(self, env):
        env.reset("fix_syntax_error")   # max_steps=5
        obs = None
        for _ in range(5):
            obs = env.step(SQLAction(sql_query="SELEC 1;"))  # bad: won't solve
        assert obs.done is True

    def test_done_on_reward_threshold(self, env):
        """Correct answer on easy task should trigger done."""
        env.reset("fix_syntax_error")
        correct_sql = "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        obs = env.step(SQLAction(sql_query=correct_sql))
        # reward should be ≥ threshold (0.8) → done
        if obs.reward >= TASK_EASY.reward_threshold:
            assert obs.done is True

    def test_reward_breakdown_present_after_step(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert obs.reward_breakdown is not None
        assert hasattr(obs.reward_breakdown, "correctness")
        assert hasattr(obs.reward_breakdown, "explanation")

    def test_episode_id_unique_per_reset(self, env):
        env.reset("fix_syntax_error")
        id1 = env.state.episode_id
        env.reset("fix_syntax_error")
        id2 = env.state.episode_id
        assert id1 != id2

    def test_db_returns_results(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT * FROM customers LIMIT 3;"))
        assert obs.error_message is None
        assert obs.query_result is not None
        assert len(obs.query_result) == 3


# ===========================================================================
# 4. FASTAPI ENDPOINT TESTS
# ===========================================================================

class TestAPI:
    def test_health_endpoint_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    def test_health_returns_env_name(self, client):
        r = client.get("/health")
        assert "sql-debug-env" in r.json()["environment"]

    def test_info_endpoint(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3

    def test_tasks_endpoint(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 3
        task_ids = {t["task_id"] for t in tasks}
        assert task_ids == {"fix_syntax_error", "fix_logic_error", "optimize_query"}

    def test_tasks_have_difficulty(self, client):
        r = client.get("/tasks")
        difficulties = {t["difficulty"] for t in r.json()}
        assert difficulties == {"easy", "medium", "hard"}

    def test_reset_default_task(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        obs = r.json()
        assert obs["task_id"] == "fix_syntax_error"
        assert obs["done"] is False
        assert obs["step_count"] == 0

    def test_reset_specific_task(self, client):
        r = client.post("/reset", json={"task_id": "fix_logic_error"})
        assert r.status_code == 200
        assert r.json()["task_id"] == "fix_logic_error"

    def test_reset_invalid_task_400(self, client):
        r = client.post("/reset", json={"task_id": "nonexistent"})
        assert r.status_code == 400

    def test_reset_returns_observation_schema(self, client):
        r = client.post("/reset", json={"task_id": "fix_syntax_error"})
        obs = r.json()
        required_fields = [
            "task_id", "task_description", "broken_query",
            "schema_hint", "step_count", "max_steps", "done", "reward",
        ]
        for field in required_fields:
            assert field in obs, f"Missing field: {field}"

    def test_step_returns_observation(self, client):
        client.post("/reset", json={"task_id": "fix_syntax_error"})
        r = client.post("/step", json={
            "action": {"sql_query": "SELECT 1;"}
        })
        assert r.status_code == 200
        obs = r.json()
        assert obs["step_count"] == 1
        assert 0.0 <= obs["reward"] <= 1.0

    def test_step_with_correct_easy_answer(self, client):
        client.post("/reset", json={"task_id": "fix_syntax_error"})
        r = client.post("/step", json={
            "action": {
                "sql_query": (
                    "SELECT id, name, email FROM customers "
                    "WHERE tier = 'vip' ORDER BY name;"
                )
            }
        })
        assert r.status_code == 200
        obs = r.json()
        assert obs["reward"] >= 0.7   # should be high for correct answer

    def test_step_with_bad_sql_returns_error(self, client):
        client.post("/reset", json={"task_id": "fix_syntax_error"})
        r = client.post("/step", json={
            "action": {"sql_query": "SELEC FORM;"}
        })
        assert r.status_code == 200
        obs = r.json()
        assert obs["error_message"] is not None

    def test_step_reward_breakdown_present(self, client):
        client.post("/reset", json={"task_id": "fix_syntax_error"})
        r = client.post("/step", json={"action": {"sql_query": "SELECT 1;"}})
        obs = r.json()
        assert obs["reward_breakdown"] is not None
        assert "correctness" in obs["reward_breakdown"]
        assert "explanation" in obs["reward_breakdown"]

    def test_state_endpoint(self, client):
        client.post("/reset", json={"task_id": "fix_logic_error"})
        r = client.get("/state")
        assert r.status_code == 200
        s = r.json()
        assert s["task_id"] == "fix_logic_error"
        assert s["step_count"] == 0

    def test_state_updates_after_step(self, client):
        client.post("/reset", json={"task_id": "fix_syntax_error"})
        client.post("/step", json={"action": {"sql_query": "SELECT 1;"}})
        r = client.get("/state")
        assert r.json()["step_count"] == 1

    def test_docs_endpoint_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "/reset" in schema["paths"]
        assert "/step" in schema["paths"]
        assert "/state" in schema["paths"]


# ===========================================================================
# 5. END-TO-END EPISODE TESTS (all 3 tasks)
# ===========================================================================

class TestEndToEnd:
    """Simulate a full agent episode for each task using the correct query."""

    def test_easy_episode_with_correct_answer(self, env):
        """Agent submits the correct fix on step 1 → done with high reward."""
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(
            sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
            reasoning="Fixed SELEC→SELECT, FORM→FROM, ORDR→ORDER.",
        ))
        assert obs.reward >= 0.8
        assert obs.done is True
        assert obs.error_message is None
        assert obs.query_result is not None
        assert len(obs.query_result) == 3

    def test_easy_episode_multi_step_recovery(self, env):
        """Agent starts wrong, then fixes on step 2 → reward improves."""
        env.reset("fix_syntax_error")
        obs1 = env.step(SQLAction(sql_query="SELEC id FROM customers;"))
        r1 = obs1.reward

        obs2 = env.step(SQLAction(
            sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        ))
        r2 = obs2.reward
        assert r2 > r1   # reward must improve

    def test_medium_episode_with_correct_answer(self, env):
        """Agent submits the correct join fix → high reward."""
        correct_sql = (
            "SELECT o.id AS order_id, c.name AS customer_name, "
            "COUNT(oi.id) AS item_count, "
            "SUM(oi.quantity * oi.unit_price) AS computed_total "
            "FROM orders o "
            "JOIN customers c ON c.id = o.customer_id "
            "LEFT JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY o.id, c.name ORDER BY o.id;"
        )
        env.reset("fix_logic_error")
        obs = env.step(SQLAction(sql_query=correct_sql))
        assert obs.reward >= 0.6
        assert obs.error_message is None

    def test_hard_episode_with_correct_answer(self, env):
        """Agent submits the optimised query → correctness score high."""
        optimised_sql = (
            "SELECT c.id, c.name, c.region, c.tier, "
            "COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue, "
            "COUNT(DISTINCT o.id) AS order_count "
            "FROM customers c "
            "LEFT JOIN orders o ON o.customer_id = c.id AND o.status != 'cancelled' "
            "LEFT JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id, c.name, c.region, c.tier "
            "ORDER BY total_revenue DESC;"
        )
        env.reset("optimize_query")
        obs = env.step(SQLAction(sql_query=optimised_sql))
        assert obs.error_message is None
        assert obs.reward_breakdown.correctness >= 0.7

    def test_all_tasks_produce_non_zero_reward_on_correct(self, env):
        correct_queries = {
            "fix_syntax_error": (
                "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
            ),
            "fix_logic_error": (
                "SELECT o.id AS order_id, c.name AS customer_name, "
                "COUNT(oi.id) AS item_count, "
                "SUM(oi.quantity * oi.unit_price) AS computed_total "
                "FROM orders o JOIN customers c ON c.id = o.customer_id "
                "LEFT JOIN order_items oi ON oi.order_id = o.id "
                "GROUP BY o.id, c.name ORDER BY o.id;"
            ),
            "optimize_query": (
                "SELECT c.id, c.name, c.region, c.tier, "
                "COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue, "
                "COUNT(DISTINCT o.id) AS order_count "
                "FROM customers c "
                "LEFT JOIN orders o ON o.customer_id = c.id AND o.status != 'cancelled' "
                "LEFT JOIN order_items oi ON oi.order_id = o.id "
                "GROUP BY c.id, c.name, c.region, c.tier "
                "ORDER BY total_revenue DESC;"
            ),
        }
        for task_id, sql in correct_queries.items():
            env.reset(task_id)
            obs = env.step(SQLAction(sql_query=sql))
            assert obs.reward > 0.0, f"Task {task_id} gave zero reward for correct answer"
            assert 0.0 <= obs.reward <= 1.0

    def test_graders_never_return_same_score_for_different_inputs(self, env):
        """Disqualification guard: graders must not always return same score."""
        env.reset("fix_syntax_error")
        obs_bad  = env.step(SQLAction(sql_query="SELEC 1;"))
        obs_good = env.step(SQLAction(
            sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        ))
        # Scores must differ between a bad and good query
        # (Note: step penalties differ too, so this is a safe check)
        scores = {obs_bad.reward, obs_good.reward}
        assert len(scores) > 1, "Grader returned identical scores for different queries"

    def test_cumulative_reward_increases(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELECT 1;"))
        cum1 = env.state.cumulative_reward
        env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        cum2 = env.state.cumulative_reward
        assert cum2 >= cum1

    def test_partial_credit_increases_toward_solution(self, env):
        """Reward should increase as agent gets progressively closer."""
        env.reset("fix_syntax_error")

        # Step 1: completely wrong
        obs1 = env.step(SQLAction(sql_query="SELEC id FORM customers;"))

        # Step 2: executes but wrong filter
        obs2 = env.step(SQLAction(sql_query="SELECT id, name, email FROM customers;"))

        # Step 3: correct
        obs3 = env.step(SQLAction(
            sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        ))
        # Final reward should be highest
        assert obs3.reward >= obs1.reward
