"""
validate_submission.py — Pre-submission validation script

Run this before submitting to the Scaler dashboard.
Checks ALL disqualification criteria automatically.

Usage:
    # Against local server
    python validate_submission.py

    # Against deployed HF Space
    ENV_BASE_URL=https://Brijesh8128-sql-debug-env.hf.space python validate_submission.py
"""

import json
import os
import sys
import time

import httpx

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
TIMEOUT = 30.0

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []
fails = 0


def check(name: str, cond: bool, detail: str = "", critical: bool = True):
    global fails
    status = PASS if cond else (FAIL if critical else WARN)
    if not cond and critical:
        fails += 1
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f"  [{detail}]" if detail else ""))
    return cond


def get(path: str, timeout: float = TIMEOUT):
    return httpx.get(f"{ENV_BASE_URL}{path}", timeout=timeout)


def post(path: str, body: dict, timeout: float = TIMEOUT):
    return httpx.post(
        f"{ENV_BASE_URL}{path}",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def main():
    print("=" * 60)
    print(" SQL Debug Environment — Pre-submission Validator")
    print(f" Target: {ENV_BASE_URL}")
    print("=" * 60)

    # ── 1. Health check ────────────────────────────────────────────────
    print("\n1. HEALTH CHECK (must return 200)")
    try:
        r = get("/health")
        check("GET /health returns 200", r.status_code == 200,
              f"status={r.status_code}")
        data = r.json()
        check("health.status = ok", data.get("status") == "ok",
              f"got {data.get('status')}")
        check("health has environment field", "environment" in data)
        print(f"     Server: {data.get('environment')} v{data.get('version','?')}")
    except Exception as e:
        check("GET /health reachable", False, str(e))
        print("\n  Cannot reach server — stopping.")
        sys.exit(1)

    # ── 2. Reset endpoint ──────────────────────────────────────────────
    print("\n2. RESET ENDPOINT (POST /reset)")
    TASK_IDS = [
        "fix_syntax_error", "fix_logic_error", "fix_null_handling",
        "fix_subquery_bug", "optimize_query", "fix_window_function",
        "fix_cte", "multi_step_aggregation",
    ]
    for tid in TASK_IDS:
        try:
            r = post("/reset", {"task_id": tid})
            obs = r.json()
            check(f"reset({tid})", r.status_code == 200 and obs.get("task_id") == tid,
                  f"status={r.status_code}")
        except Exception as e:
            check(f"reset({tid})", False, str(e))

    # ── 3. Step endpoint ───────────────────────────────────────────────
    print("\n3. STEP ENDPOINT (POST /step)")

    # Test with bad SQL — should return error, not crash
    post("/reset", {"task_id": "fix_syntax_error"})
    try:
        r = post("/step", {"action": {"sql_query": "SELEC 1;"}})
        check("step: bad SQL returns 200", r.status_code == 200,
              f"status={r.status_code}")
        obs = r.json()
        check("step: error_message set for bad SQL", obs.get("error_message") is not None)
        check("step: reward in [0,1]", 0.0 <= float(obs.get("reward", -1)) <= 1.0,
              f"got {obs.get('reward')}")
        check("step: reward_breakdown present", obs.get("reward_breakdown") is not None)
    except Exception as e:
        check("step: bad SQL handled", False, str(e))

    # Test with correct SQL — should score high
    post("/reset", {"task_id": "fix_syntax_error"})
    try:
        r = post("/step", {"action": {
            "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        }})
        obs = r.json()
        check("step: correct SQL high reward", float(obs.get("reward", 0)) >= 0.8,
              f"got {obs.get('reward')}")
        check("step: done=True on correct easy", obs.get("done") == True)
    except Exception as e:
        check("step: correct SQL handled", False, str(e))

    # ── 4. Grader non-constant check (anti-disqualification) ───────────
    print("\n4. GRADER PRODUCES DIFFERENT SCORES (anti-disqualification)")
    post("/reset", {"task_id": "fix_syntax_error"})
    try:
        r_bad = post("/step", {"action": {"sql_query": "SELEC 1;"}})
        bad_reward = float(r_bad.json().get("reward", 0))
        post("/reset", {"task_id": "fix_syntax_error"})
        r_good = post("/step", {"action": {
            "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;"
        }})
        good_reward = float(r_good.json().get("reward", 0))
        check("grader: different scores for different inputs",
              bad_reward != good_reward,
              f"bad={bad_reward:.4f} good={good_reward:.4f}")
        check("grader: error < correct (gradient exists)",
              bad_reward < good_reward,
              f"{bad_reward:.4f} < {good_reward:.4f}")
    except Exception as e:
        check("grader: non-constant scores", False, str(e))

    # ── 5. State endpoint ──────────────────────────────────────────────
    print("\n5. STATE ENDPOINT (GET /state)")
    try:
        r = get("/state")
        check("GET /state returns 200", r.status_code == 200)
        state = r.json()
        check("state has episode_id", "episode_id" in state)
        check("state has step_count", "step_count" in state)
        check("state has task_id", "task_id" in state)
    except Exception as e:
        check("GET /state works", False, str(e))

    # ── 6. Tasks endpoint ──────────────────────────────────────────────
    print("\n6. TASKS ENDPOINT (GET /tasks)")
    try:
        r = get("/tasks")
        check("GET /tasks returns 200", r.status_code == 200)
        tasks = r.json()
        check("tasks: list returned", isinstance(tasks, list))
        check("tasks: 8 tasks", len(tasks) == 8, f"got {len(tasks)}")
        difficulties = {t.get("difficulty") for t in tasks}
        check("tasks: has easy", "easy" in difficulties)
        check("tasks: has medium", "medium" in difficulties)
        check("tasks: has hard", "hard" in difficulties)
        check("tasks: has expert", "expert" in difficulties)
        for t in tasks:
            check(f"tasks: {t.get('task_id')} has max_steps",
                  "max_steps" in t, critical=False)
    except Exception as e:
        check("GET /tasks works", False, str(e))

    # ── 7. Advanced endpoints ──────────────────────────────────────────
    print("\n7. ADVANCED ENDPOINTS")

    # History
    try:
        r = get("/history")
        check("GET /history returns 200", r.status_code == 200,
              f"status={r.status_code}", critical=False)
        hist = r.json()
        check("history has steps field", "steps" in hist, critical=False)
    except Exception as e:
        check("GET /history works", False, str(e), critical=False)

    # Hint
    post("/reset", {"task_id": "fix_syntax_error"})
    try:
        r = post("/hint", {})
        check("POST /hint returns 200", r.status_code == 200,
              f"status={r.status_code}", critical=False)
        h = r.json()
        check("hint has level", "level" in h, critical=False)
        check("hint has hint text", "hint" in h and len(h["hint"]) > 5, critical=False)
        check("hint has penalty", "total_penalty" in h, critical=False)
    except Exception as e:
        check("POST /hint works", False, str(e), critical=False)

    # Curriculum
    try:
        r = get("/curriculum")
        check("GET /curriculum returns 200", r.status_code == 200,
              critical=False)
        curr = r.json()
        check("curriculum has tasks", "tasks" in curr, critical=False)
        check("curriculum has 8 tasks",
              len(curr.get("tasks", [])) == 8, critical=False)
    except Exception as e:
        check("GET /curriculum works", False, str(e), critical=False)

    # Evaluate
    try:
        r = post("/evaluate", {
            "task_id": "fix_syntax_error",
            "sql_query": "SELECT id FROM customers;"
        })
        check("POST /evaluate returns 200", r.status_code == 200,
              f"status={r.status_code}", critical=False)
        ev = r.json()
        check("evaluate has reward", "reward" in ev, critical=False)
    except Exception as e:
        check("POST /evaluate works", False, str(e), critical=False)

    # Batch evaluate
    try:
        r = post("/evaluate/batch", {"sql_query": "SELECT 1;"})
        check("POST /evaluate/batch returns 200", r.status_code == 200,
              critical=False)
        batch = r.json()
        check("batch has results_by_task", "results_by_task" in batch, critical=False)
        check("batch has mean_score", "mean_score" in batch, critical=False)
    except Exception as e:
        check("POST /evaluate/batch works", False, str(e), critical=False)

    # Leaderboard
    try:
        r = get("/leaderboard")
        check("GET /leaderboard returns 200", r.status_code == 200,
              critical=False)
    except Exception as e:
        check("GET /leaderboard works", False, str(e), critical=False)

    # ── 8. Observation schema check ────────────────────────────────────
    print("\n8. OBSERVATION SCHEMA VALIDATION")
    post("/reset", {"task_id": "fix_syntax_error"})
    r = post("/step", {"action": {"sql_query": "SELECT 1;"}})
    obs = r.json()

    required_obs_fields = [
        "task_id", "task_description", "broken_query", "schema_hint",
        "reward", "reward_breakdown", "step_count", "max_steps", "done",
    ]
    for field in required_obs_fields:
        check(f"obs has field: {field}", field in obs)

    advanced_obs_fields = [
        "conversation_history", "query_analysis", "hint_available", "hints_used",
    ]
    for field in advanced_obs_fields:
        check(f"obs has advanced field: {field}", field in obs, critical=False)

    reward_bd = obs.get("reward_breakdown", {}) or {}
    for field in ["total", "correctness", "efficiency", "step_penalty", "explanation"]:
        check(f"reward_breakdown has: {field}", field in reward_bd)

    # ── 9. Docs endpoint ───────────────────────────────────────────────
    print("\n9. SWAGGER DOCS")
    try:
        r = get("/docs")
        check("GET /docs returns 200", r.status_code == 200)
    except Exception as e:
        check("GET /docs works", False, str(e))

    # ── 10. Run all 8 tasks end-to-end ────────────────────────────────
    print("\n10. ALL 8 TASKS — END-TO-END")
    CORRECT_SQL = {
        "fix_syntax_error": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
        "fix_logic_error": "SELECT o.id AS order_id, c.name AS customer_name, COUNT(oi.id) AS item_count, SUM(oi.quantity * oi.unit_price) AS computed_total FROM orders o JOIN customers c ON c.id = o.customer_id LEFT JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id, c.name ORDER BY o.id;",
        "fix_null_handling": "SELECT p.id, p.name, p.category, COALESCE(AVG(r.rating), 0.0) AS avg_rating, COUNT(r.id) AS review_count FROM products p LEFT JOIN reviews r ON r.product_id = p.id WHERE p.active = 1 GROUP BY p.id, p.name, p.category ORDER BY avg_rating DESC, p.name ASC;",
        "fix_subquery_bug": "SELECT c.id, c.name, c.tier, SUM(oi.quantity * oi.unit_price) AS total_spent FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' GROUP BY c.id, c.name, c.tier HAVING total_spent > (SELECT AVG(total) FROM (SELECT SUM(oi2.quantity * oi2.unit_price) AS total FROM orders ord JOIN order_items oi2 ON oi2.order_id = ord.id WHERE ord.status != 'cancelled' GROUP BY ord.id)) ORDER BY total_spent DESC;",
        "optimize_query": "SELECT c.id, c.name, c.region, c.tier, COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue, COUNT(DISTINCT o.id) AS order_count FROM customers c LEFT JOIN orders o ON o.customer_id = c.id AND o.status != 'cancelled' LEFT JOIN order_items oi ON oi.order_id = o.id GROUP BY c.id, c.name, c.region, c.tier ORDER BY total_revenue DESC;",
        "fix_window_function": "SELECT c.id, c.name, c.region, c.tier, SUM(oi.quantity * oi.unit_price) AS total_spent, RANK() OVER (PARTITION BY c.region ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS region_rank FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' GROUP BY c.id, c.name, c.region, c.tier ORDER BY c.region, region_rank;",
        "fix_cte": "WITH customer_revenue AS (SELECT c.id, c.name, c.tier, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' GROUP BY c.id, c.name, c.tier) SELECT id, name, tier, total_revenue, ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM customer_revenue), 2) AS revenue_pct FROM customer_revenue ORDER BY total_revenue DESC;",
        "multi_step_aggregation": "SELECT p.category, c.tier, COUNT(DISTINCT c.id) AS unique_customers, SUM(oi.quantity) AS total_units, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue, ROUND(SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.id), 2) AS avg_order_value FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE o.status != 'cancelled' GROUP BY p.category, c.tier ORDER BY total_revenue DESC;",
    }
    task_scores = {}
    for tid, sql in CORRECT_SQL.items():
        try:
            post("/reset", {"task_id": tid})
            r = post("/step", {"action": {"sql_query": sql}})
            obs = r.json()
            reward = float(obs.get("reward", 0))
            task_scores[tid] = reward
            check(f"{tid}: reward≥0.6", reward >= 0.6, f"reward={reward:.4f}")
            check(f"{tid}: no error", obs.get("error_message") is None,
                  str(obs.get("error_message", ""))[:60])
            check(f"{tid}: score in [0,1]", 0.0 <= reward <= 1.0)
        except Exception as e:
            check(f"{tid}: runs without exception", False, str(e))

    # ── Summary ────────────────────────────────────────────────────────
    mean = sum(task_scores.values()) / len(task_scores) if task_scores else 0
    print(f"\n{'=' * 60}")
    print(" VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    critical_fails = sum(1 for s, n, d in results if s == FAIL)
    warnings = sum(1 for s, n, d in results if s == WARN)
    passed = sum(1 for s, n, d in results if s == PASS)
    total = len(results)

    print(f"\n  Passed   : {passed}/{total}")
    print(f"  Failures : {critical_fails} (critical)")
    print(f"  Warnings : {warnings} (non-critical)")
    print(f"\n  Task scores:")
    for tid, score in task_scores.items():
        bar = "█" * int(score * 20)
        status = PASS if score >= 0.6 else FAIL
        print(f"    {status} {tid:<30} {score:.4f} |{bar:<20}|")
    print(f"\n  Mean score: {mean:.4f}")

    if critical_fails == 0:
        print(f"\n  🎉 ALL CRITICAL CHECKS PASSED")
        print(f"  ✅ Ready to submit!")
        print(f"\n  Submit URL:")
        print(f"  https://huggingface.co/spaces/Brijesh8128/sql-debug-env")
    else:
        print(f"\n  ❌ {critical_fails} CRITICAL FAILURE(S) — fix before submitting")

    # Machine-readable output
    result_json = {
        "passed": passed,
        "failed": critical_fails,
        "warnings": warnings,
        "total": total,
        "task_scores": task_scores,
        "mean_score": round(mean, 4),
        "ready_to_submit": critical_fails == 0,
    }
    print(f"\nJSON output:")
    print(json.dumps(result_json, indent=2))
    sys.exit(0 if critical_fails == 0 else 1)


if __name__ == "__main__":
    main()
