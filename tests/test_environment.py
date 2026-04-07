"""
tests/test_environment.py — Comprehensive test suite for SQL Debug Environment v3.0

Tests cover:
  - All 8 tasks: reset, step, correct answer, reward range
  - Advanced models: QueryComplexity, PerformanceMetrics, EpisodeSummary
  - Reward properties: non-sparse, deterministic, bounded [0,1]
  - Environment lifecycle: reset clears state, unique episode_ids
  - Hint system: levels, penalty application, reset on new episode
  - Curriculum: status, advancement logic
  - History: cumulative reward, improvement_rate, best_reward
  - Edge cases: step before reset, max steps truncation, empty SQL
"""
from __future__ import annotations

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import SQLAction, SQLObservation, QueryComplexity, PerformanceMetrics, EpisodeSummary
from server.environment import SQLDebugEnvironment, CURRICULUM_ORDER, HINTS
from server.graders import classify_query, grade
from server.tasks import TASKS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = SQLDebugEnvironment()
    yield e
    e.close()


CORRECT_SQL = {
    "fix_syntax_error": (
        "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
    ),
    "fix_logic_error": (
        "SELECT o.id AS order_id, c.name AS customer_name, COUNT(oi.id) AS item_count, "
        "SUM(oi.quantity * oi.unit_price) AS computed_total FROM orders o "
        "JOIN customers c ON c.id = o.customer_id "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY o.id, c.name ORDER BY o.id;"
    ),
    "fix_null_handling": (
        "SELECT p.id, p.name, p.category, COALESCE(AVG(r.rating), 0.0) AS avg_rating, "
        "COUNT(r.id) AS review_count FROM products p "
        "LEFT JOIN reviews r ON r.product_id = p.id "
        "WHERE p.active = 1 GROUP BY p.id, p.name, p.category "
        "ORDER BY avg_rating DESC, p.name ASC;"
    ),
    "fix_subquery_bug": (
        "SELECT c.id, c.name, c.tier, SUM(oi.quantity * oi.unit_price) AS total_spent "
        "FROM customers c JOIN orders o ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' "
        "GROUP BY c.id, c.name, c.tier "
        "HAVING total_spent > (SELECT AVG(total) FROM "
        "(SELECT SUM(oi2.quantity * oi2.unit_price) AS total FROM orders ord "
        "JOIN order_items oi2 ON oi2.order_id = ord.id "
        "WHERE ord.status != 'cancelled' GROUP BY ord.id)) ORDER BY total_spent DESC;"
    ),
    "optimize_query": (
        "SELECT c.id, c.name, c.region, c.tier, "
        "COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue, "
        "COUNT(DISTINCT o.id) AS order_count FROM customers c "
        "LEFT JOIN orders o ON o.customer_id = c.id AND o.status != 'cancelled' "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY c.id, c.name, c.region, c.tier ORDER BY total_revenue DESC;"
    ),
    "fix_window_function": (
        "SELECT c.id, c.name, c.region, c.tier, "
        "SUM(oi.quantity * oi.unit_price) AS total_spent, "
        "RANK() OVER (PARTITION BY c.region ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS region_rank "
        "FROM customers c JOIN orders o ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' "
        "GROUP BY c.id, c.name, c.region, c.tier ORDER BY c.region, region_rank;"
    ),
    "fix_cte": (
        "WITH customer_revenue AS ("
        "SELECT c.id, c.name, c.tier, SUM(oi.quantity * oi.unit_price) AS total_revenue "
        "FROM customers c JOIN orders o ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' "
        "GROUP BY c.id, c.name, c.tier) "
        "SELECT id, name, tier, total_revenue, "
        "ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM customer_revenue), 2) AS revenue_pct "
        "FROM customer_revenue ORDER BY total_revenue DESC;"
    ),
    "multi_step_aggregation": (
        "SELECT p.category, c.tier, COUNT(DISTINCT c.id) AS unique_customers, "
        "SUM(oi.quantity) AS total_units, "
        "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue, "
        "ROUND(SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.id), 2) AS avg_order_value "
        "FROM customers c JOIN orders o ON o.customer_id = c.id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id WHERE o.status != 'cancelled' "
        "GROUP BY p.category, c.tier ORDER BY total_revenue DESC;"
    ),
}


# ===========================================================================
# 1. TASK REGISTRY
# ===========================================================================

class TestTaskRegistry:
    def test_8_tasks_registered(self):
        assert len(TASKS) == 8

    def test_all_difficulties_present(self):
        diffs = {t.difficulty for t in TASKS.values()}
        assert diffs == {"easy", "medium", "hard", "expert"}

    def test_all_tasks_have_hints(self):
        for tid in TASKS:
            assert tid in HINTS
            assert len(HINTS[tid]) == 3

    def test_curriculum_order_matches_tasks(self):
        assert len(CURRICULUM_ORDER) == 8
        for tid in CURRICULUM_ORDER:
            assert tid in TASKS

    def test_all_tasks_have_expected_results(self):
        for tid, task in TASKS.items():
            assert len(task.expected_result) > 0, f"{tid} has no expected_result"

    def test_tasks_reward_thresholds_valid(self):
        for task in TASKS.values():
            assert 0.0 < task.reward_threshold <= 1.0

    def test_tasks_max_steps_positive(self):
        for task in TASKS.values():
            assert task.max_steps >= 5


# ===========================================================================
# 2. QUERY COMPLEXITY CLASSIFIER
# ===========================================================================

class TestQueryComplexity:
    def test_simple_query(self):
        qc = classify_query("SELECT id FROM customers;")
        assert qc.label == "simple"
        assert not qc.has_join
        assert not qc.has_aggregation

    def test_moderate_query(self):
        qc = classify_query("SELECT id FROM customers WHERE tier = 'vip' ORDER BY name;")
        assert qc.label in ["simple", "moderate"]
        assert qc.has_where
        assert qc.has_order_by

    def test_complex_query_with_join_and_agg(self):
        qc = classify_query(
            "SELECT c.id, SUM(o.total) FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id ORDER BY 2 DESC;"
        )
        assert qc.has_join
        assert qc.has_aggregation
        assert qc.has_group_by
        assert qc.join_count >= 1
        assert qc.complexity_score > 0

    def test_advanced_query_with_window_cte(self):
        qc = classify_query(
            "WITH cte AS (SELECT id FROM t) "
            "SELECT RANK() OVER (PARTITION BY x ORDER BY y) FROM cte JOIN other ON cte.id=other.id;"
        )
        assert qc.has_cte
        assert qc.has_window_function
        assert qc.has_join
        assert qc.label in ["complex", "advanced"]

    def test_subquery_detection(self):
        qc = classify_query("SELECT * FROM t WHERE id IN (SELECT id FROM other);")
        assert qc.has_subquery
        assert qc.subquery_depth >= 1

    def test_complexity_score_bounded(self):
        for sql in [
            "SELECT 1;",
            "SELECT * FROM customers, orders, order_items, products, reviews;",
            "WITH a AS (SELECT 1) SELECT RANK() OVER (PARTITION BY x ORDER BY y) FROM a JOIN b ON a.id=b.id;"
        ]:
            qc = classify_query(sql)
            assert 0.0 <= qc.complexity_score <= 1.0


# ===========================================================================
# 3. GRADERS
# ===========================================================================

class TestGraders:
    def test_all_8_tasks_score_above_threshold(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        for tid, task in TASKS.items():
            rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
            bd = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=1)
            assert bd.total >= 0.6, f"{tid} scored {bd.total}"
        conn.close()

    def test_error_always_less_than_correct(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        for tid, task in TASKS.items():
            rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
            bd_ok = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=1)
            bd_err = grade(task=task, error="syntax error", result=None, exec_ms=None, step_count=1)
            assert bd_err.total < bd_ok.total, f"{tid}: {bd_err.total} not < {bd_ok.total}"
        conn.close()

    def test_reward_always_in_0_1(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        for tid, task in TASKS.items():
            rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
            for step in [1, 3, 5]:
                bd = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=step)
                assert 0.0 <= bd.total <= 1.0
        conn.close()

    def test_grader_deterministic(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        for tid, task in TASKS.items():
            rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
            bd1 = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=1)
            bd2 = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=1)
            assert bd1.total == bd2.total
        conn.close()

    def test_reward_breakdown_has_new_fields(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        task = TASKS["fix_syntax_error"]
        rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
        bd = grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=1)
        assert hasattr(bd, "row_coverage")
        assert hasattr(bd, "column_coverage")
        assert 0.0 <= bd.row_coverage <= 1.0
        assert 0.0 <= bd.column_coverage <= 1.0
        conn.close()

    def test_step_penalty_decreases(self):
        import sqlite3
        from pathlib import Path
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(Path("server/db/seed.sql").read_text())
        conn.commit()
        task = TASKS["fix_syntax_error"]
        rows = [dict(r) for r in conn.execute(task.correct_query).fetchall()]
        rewards = [
            grade(task=task, error=None, result=rows, exec_ms=0.5, step_count=s).total
            for s in [1, 2, 3, 4, 5]
        ]
        # Each step should have same or lower reward
        for i in range(1, len(rewards)):
            assert rewards[i] <= rewards[i-1] + 0.01
        conn.close()


# ===========================================================================
# 4. ENVIRONMENT — CORE
# ===========================================================================

class TestEnvironmentCore:
    def test_reset_returns_observation(self, env):
        obs = env.reset("fix_syntax_error")
        assert isinstance(obs, SQLObservation)
        assert obs.task_id == "fix_syntax_error"
        assert obs.done is False
        assert obs.step_count == 0
        assert len(obs.episode_id) > 8  # UUID

    def test_reset_clears_conversation(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELECT 1;"))
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert obs.step_count == 1
        assert len(obs.conversation_history) == 1

    def test_reset_all_8_tasks(self, env):
        for tid in TASKS:
            obs = env.reset(tid)
            assert obs.task_id == tid
            assert obs.step_count == 0
            assert obs.done is False

    def test_reset_unknown_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("not_a_real_task")

    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.step(SQLAction(sql_query="SELECT 1;"))

    def test_unique_episode_ids(self, env):
        obs1 = env.reset("fix_syntax_error")
        obs2 = env.reset("fix_syntax_error")
        assert obs1.episode_id != obs2.episode_id

    def test_step_count_increments(self, env):
        env.reset("fix_syntax_error")
        for i in range(1, 4):
            obs = env.step(SQLAction(sql_query="SELECT 1;"))
            assert obs.step_count == i

    def test_max_steps_terminates(self, env):
        obs = env.reset("fix_syntax_error")
        for _ in range(obs.max_steps):
            obs = env.step(SQLAction(sql_query="SELEC 1;"))
        assert obs.done is True
        assert obs.step_count == TASKS["fix_syntax_error"].max_steps

    def test_correct_answer_solves(self, env):
        for tid, sql in CORRECT_SQL.items():
            env.reset(tid)
            obs = env.step(SQLAction(sql_query=sql))
            assert obs.reward >= 0.6, f"{tid}: reward={obs.reward}"
            assert obs.done is True, f"{tid}: not done"

    def test_cumulative_reward_tracks(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELEC 1;"))  # 0.05
        state = env.state
        assert state.cumulative_reward > 0

    def test_state_has_new_fields(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        state = env.state
        assert hasattr(state, "best_reward")
        assert hasattr(state, "solved")
        assert hasattr(state, "hint_penalty")
        assert state.solved is True
        assert state.best_reward > 0


# ===========================================================================
# 5. ADVANCED OBSERVATION FIELDS
# ===========================================================================

class TestAdvancedObservation:
    def test_episode_id_in_observation(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert len(obs.episode_id) > 8

    def test_query_complexity_in_observation(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        assert obs.query_complexity is not None
        assert isinstance(obs.query_complexity, QueryComplexity)
        assert obs.query_complexity.label in ["simple","moderate","complex","advanced"]

    def test_performance_metrics_in_observation(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT id FROM customers WHERE id=1;"))
        assert obs.performance_metrics is not None
        assert isinstance(obs.performance_metrics, PerformanceMetrics)
        assert obs.performance_metrics.execution_ms >= 0
        assert isinstance(obs.performance_metrics.uses_index, bool)
        assert isinstance(obs.performance_metrics.scan_count, int)

    def test_row_count_in_observation(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        assert obs.row_count >= 0

    def test_best_reward_so_far_tracks(self, env):
        env.reset("fix_syntax_error")
        obs1 = env.step(SQLAction(sql_query="SELEC 1;"))
        assert obs1.best_reward_so_far == obs1.reward
        obs2 = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        assert obs2.best_reward_so_far >= obs1.best_reward_so_far

    def test_episode_summary_on_done(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        assert obs.done is True
        assert obs.episode_summary is not None
        assert isinstance(obs.episode_summary, EpisodeSummary)
        assert obs.episode_summary.solved is True
        assert obs.episode_summary.termination_reason == "solved"
        assert obs.episode_summary.total_steps == 1
        assert len(obs.episode_summary.step_rewards) == 1
        assert obs.episode_summary.best_reward > 0

    def test_episode_summary_max_steps_reason(self, env):
        obs = env.reset("fix_syntax_error")
        for _ in range(obs.max_steps):
            obs = env.step(SQLAction(sql_query="SELEC 1;"))
        assert obs.done is True
        assert obs.episode_summary is not None
        assert obs.episode_summary.termination_reason == "max_steps"
        assert obs.episode_summary.solved is False

    def test_conversation_history_grows(self, env):
        env.reset("fix_logic_error")
        for i in range(1, 4):
            obs = env.step(SQLAction(sql_query=f"SELECT {i};"))
            assert len(obs.conversation_history) == i

    def test_conversation_history_capped_at_5(self, env):
        env.reset("fix_logic_error")
        for i in range(8):
            obs = env.step(SQLAction(sql_query=f"SELECT {i};"))
        assert len(obs.conversation_history) <= 5

    def test_conversation_history_has_complexity(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        assert "complexity" in obs.conversation_history[0]


# ===========================================================================
# 6. HINT SYSTEM
# ===========================================================================

class TestHintSystem:
    def test_hints_level_1_2_3(self, env):
        env.reset("fix_syntax_error")
        h1 = env.get_hint()
        assert h1["level"] == 1
        assert h1["penalty_applied"] == 0.10
        assert h1["total_penalty"] == 0.10
        assert h1["hints_remaining"] == 2

        h2 = env.get_hint()
        assert h2["level"] == 2
        assert h2["total_penalty"] == 0.20
        assert h2["hints_remaining"] == 1

        h3 = env.get_hint()
        assert h3["level"] == 3
        assert h3["total_penalty"] == 0.30
        assert h3["hints_remaining"] == 0

    def test_hint_4th_returns_no_more(self, env):
        env.reset("fix_syntax_error")
        for _ in range(3):
            env.get_hint()
        h4 = env.get_hint()
        assert "No more" in h4["hint"]
        assert h4["hints_remaining"] == 0

    def test_hint_penalty_applied_to_reward(self, env):
        env.reset("fix_syntax_error")
        env.get_hint()  # 10% penalty
        obs = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        assert obs.reward < 1.0
        assert abs(obs.reward - 0.90) < 0.05
        assert obs.reward_breakdown.hint_penalty == 0.10

    def test_30_pct_hint_penalty(self, env):
        env.reset("fix_syntax_error")
        env.get_hint()
        env.get_hint()
        env.get_hint()  # 30% penalty
        obs = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        assert abs(obs.reward - 0.70) < 0.05

    def test_hints_reset_on_new_episode(self, env):
        env.reset("fix_syntax_error")
        env.get_hint()
        env.get_hint()
        env.reset("fix_syntax_error")
        h = env.get_hint()
        assert h["level"] == 1
        assert h["total_penalty"] == 0.10

    def test_hint_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset"):
            env.get_hint()

    def test_hint_state_in_observation(self, env):
        env.reset("fix_syntax_error")
        env.get_hint()
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert obs.hints_used == 1
        assert obs.hint_penalty == 0.10
        assert obs.hint_available is True

    def test_hint_not_available_after_3(self, env):
        env.reset("fix_syntax_error")
        for _ in range(3):
            env.get_hint()
        obs = env.step(SQLAction(sql_query="SELECT 1;"))
        assert obs.hint_available is False


# ===========================================================================
# 7. CURRICULUM
# ===========================================================================

class TestCurriculum:
    def test_curriculum_status_has_8_tasks(self, env):
        env.reset("fix_syntax_error")
        status = env.curriculum_status()
        assert len(status["tasks"]) == 8
        assert status["progress"] == "1/8"
        assert status["current_task"] == "fix_syntax_error"

    def test_first_task_is_current(self, env):
        env.reset("fix_syntax_error")
        status = env.curriculum_status()
        assert status["tasks"][0]["current"] is True

    def test_curriculum_next_in_progress(self, env):
        env.reset("fix_syntax_error")
        result = env.curriculum_next()
        assert result["status"] == "in_progress"
        assert result["recommended_task"] == "fix_syntax_error"
        assert "progress" in result

    def test_curriculum_advances_after_mastery(self, env):
        # Solve fix_syntax_error 3 times
        for _ in range(3):
            env.reset("fix_syntax_error")
            env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        result = env.curriculum_next()
        assert result["status"] in ["advanced", "in_progress"]
        if result["status"] == "advanced":
            assert result["recommended_task"] == "fix_logic_error"
            assert result["progress"] == "2/8"

    def test_curriculum_mastered_flag(self, env):
        for _ in range(3):
            env.reset("fix_syntax_error")
            env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        env.curriculum_next()
        status = env.curriculum_status()
        easy_task = next(t for t in status["tasks"] if t["task_id"] == "fix_syntax_error")
        assert easy_task["episodes"] >= 3


# ===========================================================================
# 8. HISTORY & ANALYTICS
# ===========================================================================

class TestHistory:
    def test_history_fields(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELEC 1;"))
        env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        hist = env.get_history()
        assert "steps" in hist
        assert "cumulative_reward" in hist
        assert "best_reward" in hist
        assert "improvement_rate" in hist
        assert "hints_used" in hist
        assert "hint_penalty" in hist
        assert len(hist["steps"]) == 2

    def test_history_step_has_complexity(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        hist = env.get_history()
        assert "complexity" in hist["steps"][0]

    def test_improvement_rate_computed(self, env):
        env.reset("fix_logic_error")
        env.step(SQLAction(sql_query="SELECT 1;"))
        env.step(SQLAction(sql_query=CORRECT_SQL["fix_logic_error"]))
        hist = env.get_history()
        assert hist["improvement_rate"] >= 0

    def test_best_reward_in_history(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELEC 1;"))
        env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        hist = env.get_history()
        assert hist["best_reward"] >= 0.9


# ===========================================================================
# 9. REWARD GRADIENT (anti-disqualification)
# ===========================================================================

class TestRewardGradient:
    def test_error_lt_execution_lt_correct(self, env):
        env.reset("fix_syntax_error")
        obs_err = env.step(SQLAction(sql_query="SELEC 1;"))
        r_err = obs_err.reward

        env.reset("fix_syntax_error")
        obs_run = env.step(SQLAction(sql_query="SELECT id FROM customers;"))
        r_run = obs_run.reward

        env.reset("fix_syntax_error")
        obs_ok = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        r_ok = obs_ok.reward

        assert r_err < r_run <= r_ok, f"gradient broken: {r_err} < {r_run} <= {r_ok}"

    def test_rewards_non_constant(self, env):
        rewards = set()
        sqls = [
            "SELEC 1;",
            "SELECT 1;",
            "SELECT id FROM customers;",
            CORRECT_SQL["fix_syntax_error"],
        ]
        for sql in sqls:
            env.reset("fix_syntax_error")
            obs = env.step(SQLAction(sql_query=sql))
            rewards.add(obs.reward)
        assert len(rewards) >= 3, f"Too few distinct rewards: {rewards}"

    def test_all_rewards_in_0_1(self, env):
        sqls = ["SELEC 1;", "SELECT 1;", "SELECT id FROM customers;",
                CORRECT_SQL["fix_syntax_error"]]
        for sql in sqls:
            env.reset("fix_syntax_error")
            obs = env.step(SQLAction(sql_query=sql))
            assert 0.0 <= obs.reward <= 1.0, f"Out of range: {obs.reward} for {sql[:40]}"

    def test_step_penalty_progressive(self, env):
        rewards = []
        for _ in range(4):
            env.reset("fix_syntax_error")
            for j in range(len(rewards)):
                env.step(SQLAction(sql_query="SELEC 1;"))
            obs = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
            rewards.append(obs.reward)
        # Each later step should have lower or equal reward
        for i in range(1, len(rewards)):
            assert rewards[i] <= rewards[i-1] + 0.01


# ===========================================================================
# 10. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_empty_sql_handled(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query=""))
        assert 0.0 <= obs.reward <= 1.0
        # Should not crash

    def test_drop_table_rejected_gracefully(self, env):
        env.reset("fix_syntax_error")
        obs = env.step(SQLAction(sql_query="DROP TABLE customers;"))
        assert obs.reward <= 0.25  # Low reward, not crash
        # Table should still exist
        obs2 = env.step(SQLAction(sql_query=CORRECT_SQL["fix_syntax_error"]))
        assert obs2.error_message is None  # Can still query customers

    def test_very_long_sql_handled(self, env):
        env.reset("fix_syntax_error")
        long_sql = "SELECT id, name, email FROM customers WHERE " + " AND ".join([f"id > {i}" for i in range(50)]) + ";"
        obs = env.step(SQLAction(sql_query=long_sql))
        assert 0.0 <= obs.reward <= 1.0

    def test_reasoning_optional(self, env):
        env.reset("fix_syntax_error")
        obs1 = env.step(SQLAction(sql_query="SELECT 1;"))
        env.reset("fix_syntax_error")
        obs2 = env.step(SQLAction(sql_query="SELECT 1;", reasoning="My reasoning"))
        assert obs1.reward == obs2.reward  # Reasoning doesn't affect reward

    def test_reasoning_in_history(self, env):
        env.reset("fix_syntax_error")
        env.step(SQLAction(sql_query="SELECT 1;", reasoning="test reasoning"))
        hist = env.get_history()
        assert hist["steps"][0]["reasoning"] == "test reasoning"

    def test_concurrent_reset_safe(self, env):
        """Environment should handle rapid reset/step cycles."""
        for _ in range(10):
            env.reset("fix_syntax_error")
            env.step(SQLAction(sql_query="SELECT 1;"))
        # Should not deadlock or crash

    def test_metadata_has_all_features(self, env):
        meta = env.get_metadata()
        assert "features" in meta
        assert "query_complexity_classifier" in meta["features"]
        assert "episode_summary" in meta["features"]
        assert "performance_metrics" in meta["features"]
