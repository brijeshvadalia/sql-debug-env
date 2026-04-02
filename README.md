---
title: SQL Debug Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛠️ SQL Debug Environment v3.0

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![Tasks](https://img.shields.io/badge/tasks-8-orange)](https://huggingface.co/spaces/Brijesh8128/sql-debug-env/tasks)

A **real-world OpenEnv environment** where AI agents learn to debug, fix, and optimize SQL queries against a realistic e-commerce database. Built for the **Scaler × Meta PyTorch OpenEnv Hackathon 2026**.

---

## Why SQL Debugging?

SQL debugging is performed by millions of data engineers, analysts, and DBAs every day. It is:

- **Objectively verifiable** — the query either returns the right rows or it does not
- **Naturally graduated** — syntax errors (trivial) → logic errors (moderate) → performance optimization (hard)
- **Multi-turn by nature** — agents learn from error messages and partial results across steps
- **Fully self-contained** — SQLite runs in-memory, no external APIs, no internet required
- **Production-relevant** — skills directly transfer to real data engineering work

---

## 🎮 Tasks — 8 Across 4 Difficulty Levels

| Task ID | Name | Difficulty | Max Steps | Threshold |
|---|---|---|---|---|
| `fix_syntax_error` | Fix SQL Syntax Error | 🟢 Easy | 5 | 0.80 |
| `fix_logic_error` | Fix Logic / Join Error | 🟡 Medium | 8 | 0.70 |
| `fix_null_handling` | Fix NULL Handling | 🟡 Medium | 6 | 0.75 |
| `fix_subquery_bug` | Fix Subquery Alias Bug | 🟡 Medium | 6 | 0.75 |
| `optimize_query` | Optimize N+1 Query | 🔴 Hard | 10 | 0.60 |
| `fix_window_function` | Fix Window Function | 🔴 Hard | 8 | 0.60 |
| `fix_cte` | Fix CTE Bug | 🔴 Hard | 8 | 0.60 |
| `multi_step_aggregation` | Multi-Dimension Report | 🟣 Expert | 10 | 0.60 |

### Task Descriptions

**fix_syntax_error (easy)** — Agent receives a query with 3 misspelled keywords (`SELEC`, `FORM`, `ORDR BY`) and must fix them to retrieve VIP customers.

**fix_logic_error (medium)** — Query uses `INNER JOIN` (drops orders) and sums the wrong column (`orders.total_amount` instead of line items). Agent must fix both bugs.

**fix_null_handling (medium)** — `INNER JOIN` excludes products with no reviews, and `AVG()` returns NULL instead of 0.0. Agent must use `LEFT JOIN` and `COALESCE`.

**fix_subquery_bug (medium)** — Alias `oi` is reused in inner subquery, shadowing outer query alias, producing wrong HAVING results. Agent must rename inner alias to `oi2`.

**optimize_query (hard)** — Two correlated scalar subqueries (N+1 pattern) re-scan tables for every customer row. Agent must rewrite as a single `JOIN + GROUP BY`.

**fix_window_function (hard)** — `ROW_NUMBER()` used instead of `RANK()`, and `PARTITION BY` uses wrong column (tier vs region). Agent must fix both.

**fix_cte (hard)** — CTE divides `total_revenue / total_revenue` (always 100%). Agent must use `(SELECT SUM(total_revenue) FROM customer_revenue)` as divisor.

**multi_step_aggregation (expert)** — Query missing `GROUP BY` and uses `0` for `avg_order_value`. Agent must add grouping and compute real average.

---

## 🗃️ Database Schema

```sql
customers(id, name, email, region, tier, created_at)
  tier ∈ {standard, premium, vip}
  region ∈ {US, EU, APAC}

products(id, name, category, unit_price, stock, active)
  category ∈ {electronics, furniture, stationery}

orders(id, customer_id, status, created_at, shipped_at, total_amount)
  status ∈ {pending, shipped, delivered, cancelled}

order_items(id, order_id, product_id, quantity, unit_price)

reviews(id, product_id, customer_id, rating, body, created_at)
  rating ∈ {1..5}
```

**Seed data:** 10 customers, 11 products, 11 orders, 17 order items, 7 reviews.

---

## 📦 Action Space

```python
class SQLAction(BaseModel):
    sql_query: str           # SQL to execute (required)
    reasoning: Optional[str] # Chain-of-thought (optional, logged not evaluated)
```

**Example:**
```json
{
  "action": {
    "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
    "reasoning": "Fixed SELEC->SELECT, FORM->FROM, ORDR->ORDER BY"
  }
}
```

---

## 👁️ Observation Space

```python
class SQLObservation(BaseModel):
    # Task context
    task_id: str
    task_description: str
    broken_query: str
    schema_hint: str

    # Execution feedback
    error_message: Optional[str]
    query_result: Optional[list[dict]]
    execution_time_ms: Optional[float]

    # Reward signal
    reward: float                         # [0.0, 1.0]
    reward_breakdown: Optional[RewardBreakdown]

    # Advanced: multi-turn memory
    conversation_history: list[dict]      # last 5 steps

    # Advanced: query analysis
    query_analysis: Optional[QueryAnalysis]

    # Advanced: hint system
    hint_available: bool
    hints_used: int

    # Episode metadata
    step_count: int
    max_steps: int
    done: bool
```

---

## ⚖️ Reward Function

All rewards are non-sparse — meaningful signal at every step.

```
gradient: error(0.05) < executes_wrong(0.25) < partial(0.3-0.7) < correct(0.8+)
```

**Easy tasks:**
```
reward = max(correctness × step_penalty, 0.25_execution_bonus)
```

**Medium tasks:**
```
reward = unordered_row_match × step_penalty
```

**Hard tasks:**
```
reward = (0.6 × correctness + 0.4 × efficiency) × step_penalty
efficiency = min(baseline_ms / exec_ms / 5.0, 1.0)
```

**Expert tasks:**
```
reward = (correctness + 0.1_count_bonus) × step_penalty
```

**Step penalty:**
```
step_penalty = max(0.70, 1.0 - 0.3 × (step / max_steps))
```

**Hint penalty:**
```
Each hint used: -10% on future rewards (max 3 hints = -30%)
```

---

## 🚀 Advanced Features

### 1. Multi-Turn Memory
Every observation includes `conversation_history` — the last 5 steps with SQL query, error, reward, and row count. Agents can learn from failed attempts within the same episode.

### 2. Three-Level Hint System
```
POST /hint
```
Returns progressive hints with reward penalty:
- Level 1: General direction ("The bug is in the JOIN type")
- Level 2: Specific location ("Change INNER JOIN to LEFT JOIN")
- Level 3: Exact fix ("Use LEFT JOIN and COALESCE(AVG(...), 0.0)")

Each hint costs 10% reward penalty (max 30%).

### 3. EXPLAIN Query Analysis
Every `step()` response includes `query_analysis`:
```json
{
  "scan_count": 2,
  "uses_index": false,
  "tables_scanned": ["customers", "orders"],
  "plan_steps": 4,
  "suggestion": "Consider rewriting subqueries as JOINs."
}
```

### 4. Curriculum Learning
```
GET  /curriculum        — view mastery progress
POST /curriculum/next   — get next recommended task
```
Auto-advances easy→medium→hard→expert when agent achieves avg≥0.8 over 3 episodes.

### 5. Leaderboard
```
GET  /leaderboard              — view model rankings
POST /leaderboard/submit       — submit your model's scores
```

### 6. Batch Evaluation
```
POST /evaluate/batch
Body: {"sql_query": "..."}
```
Score any SQL against all 8 tasks in a single API call.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dark homepage with live stats |
| `/tester` | GET | Visual 5-tab interactive tester |
| `/health` | GET | Liveness probe → `{"status":"ok"}` |
| `/info` | GET | Environment metadata + features |
| `/tasks` | GET | All 8 tasks with metadata |
| `/stats` | GET | Aggregate performance stats |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit SQL action |
| `/state` | GET | Current episode state |
| `/history` | GET | Full step trajectory |
| `/evaluate` | POST | Score SQL without affecting episode |
| `/evaluate/batch` | POST | Score SQL against all 8 tasks |
| `/hint` | POST | Get progressive hint (10% penalty) |
| `/curriculum` | GET | Mastery progress all 8 tasks |
| `/curriculum/next` | POST | Next recommended task |
| `/leaderboard` | GET | Model comparison rankings |
| `/leaderboard/submit` | POST | Submit model scores |
| `/docs` | GET | Swagger UI |

---

## 🚀 Setup & Usage

### Local development

```bash
git clone https://github.com/your-username/sql-debug-env
cd sql-debug-env
pip install fastapi "uvicorn[standard]" pydantic httpx openai

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test
curl http://localhost:7860/health
open http://localhost:7860/tester
```

### Docker

```bash
docker build -t sql-debug-env:latest .
docker run -p 7860:7860 sql-debug-env:latest
curl http://localhost:7860/health
```

### Python client

```python
from client import SQLDebugEnv
from models import SQLAction

with SQLDebugEnv(base_url="http://localhost:7860").sync() as env:
    # Reset for any task
    obs = env.reset(task_id="fix_syntax_error")
    print(f"Broken query: {obs.broken_query}")

    # Submit fix
    obs = env.step(SQLAction(
        sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
        reasoning="Fixed all three keyword typos."
    ))
    print(f"Reward: {obs.reward}")          # 1.0
    print(f"Done: {obs.done}")              # True
    print(f"History: {obs.conversation_history}")
    print(f"Analysis: {obs.query_analysis}")
```

### Run inference script

```bash
# Start server (separate terminal)
uvicorn server.app:app --port 7860

# Run baseline
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=hf_xxx \
ENV_BASE_URL=http://localhost:7860 \
python inference.py

# Curriculum mode (progressive difficulty)
CURRICULUM_MODE=1 python inference.py

# With hints enabled
USE_HINTS=1 python inference.py

# Specific tasks only
TASK_IDS=fix_syntax_error,optimize_query python inference.py
```

---

## 📊 Baseline Scores

Baseline run using `Qwen/Qwen2.5-72B-Instruct`, `MAX_STEPS=6`, `TEMPERATURE=0.1`:

| Task | Score | Solved in |
|---|---|---|
| `fix_syntax_error` | 1.0000 | Step 1 |
| `fix_logic_error` | 0.9091 | Step 1 |
| `fix_null_handling` | 1.0000 | Step 1 |
| `fix_subquery_bug` | 1.0000 | Step 1 |
| `optimize_query` | 0.9400 | Step 1 |
| `fix_window_function` | 1.0000 | Step 1 |
| `fix_cte` | 1.0000 | Step 1 |
| `multi_step_aggregation` | 1.0000 | Step 1 |
| **Mean** | **0.9811** | |

---

## ✅ Pre-Submission Checklist

- [x] HF Space deploys — `/health` returns 200 and `/reset` responds
- [x] OpenEnv spec — `openenv.yaml`, typed models, all endpoints
- [x] Dockerfile builds — multi-stage, non-root, port 7860, HEALTHCHECK
- [x] Baseline reproduces — `inference.py` at root, OpenAI client, <20 min
- [x] 3+ tasks — 8 tasks across easy/medium/hard/expert
- [x] Graders deterministic — same input always produces same score
- [x] All scores in [0.0, 1.0]
- [x] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars used

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built for the [Scaler × Meta PyTorch OpenEnv Hackathon 2026](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/) using the [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch and Hugging Face.
