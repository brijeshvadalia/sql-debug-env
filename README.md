# 🛠️ SQL Debug Environment

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces)

A **real-world OpenEnv environment** where AI agents learn to debug, fix, and optimize SQL queries against a realistic e-commerce database. Directly models a task performed daily by millions of data engineers, analysts, and DBAs.

---

## 🎯 Why This Environment?

SQL query debugging is a genuine, high-value task for AI agent evaluation:

- **Ubiquitous** — every company with a database has people debugging SQL
- **Verifiable** — correctness is objective: the query either returns the right rows or it doesn't  
- **Graduated difficulty** — syntax errors (easy) → logic errors (medium) → performance optimization (hard)
- **Multi-turn reasoning** — agents learn from error messages and partial results across steps
- **No external dependencies** — fully self-contained SQLite; runs offline in 2 vCPU / 8 GB

---

## 📐 Architecture

```
sql-debug-env/
├── openenv.yaml              # OpenEnv manifest (name, tasks, action/obs spaces)
├── Dockerfile                # Multi-stage build; listens on :7860
├── inference.py              # Baseline inference script (mandatory, root-level)
├── models.py                 # Pydantic types: SQLAction, SQLObservation, SQLState
├── client.py                 # HTTP client (sync + async) for the environment
├── pyproject.toml            # Dependencies (fastapi, pydantic, openai, httpx)
├── server/
│   ├── app.py                # FastAPI: /reset /step /state /health /tasks
│   ├── environment.py        # SQLDebugEnvironment (OpenEnv interface)
│   ├── db/seed.sql           # E-commerce schema + seed data (SQLite)
│   ├── tasks/__init__.py     # Task definitions: broken queries + expected results
│   └── graders/__init__.py   # Deterministic graders: 0.0–1.0 per task
└── tests/
    └── test_environment.py   # Full test suite (models, graders, env, API)
```

---

## 🗃️ Database Schema

The environment uses an in-memory SQLite database seeded with a realistic e-commerce schema:

```sql
customers(id, name, email, region, tier, created_at)
  tier ∈ {standard, premium, vip}
  region ∈ {US, EU, APAC}

products(id, name, category, unit_price, stock, active)
  category ∈ {electronics, furniture, stationery}

orders(id, customer_id, status, created_at, shipped_at, total_amount)
  status ∈ {pending, shipped, delivered, cancelled}

order_items(id, order_id, product_id, quantity, unit_price)
  -- unit_price recorded at time of order

reviews(id, product_id, customer_id, rating, body, created_at)
  rating ∈ {1..5}
```

**Seed data:** 10 customers, 11 products, 11 orders, 17 order items, 7 reviews.

---

## 🎮 Tasks

### Task 1 — Fix SQL Syntax Error `(easy)`

| Property | Value |
|---|---|
| `task_id` | `fix_syntax_error` |
| `difficulty` | easy |
| `max_steps` | 5 |
| `reward_threshold` | 0.8 |

**Objective:** The agent receives a query with multiple typos (`SELEC`, `FORM`, `ORDR BY`) and must produce a corrected query that executes successfully and returns all VIP-tier customers ordered by name.

**Broken query:**
```sql
SELEC id, name, email FORM customers
WHERE tier = 'vip'
ORDR BY name;
```

**Expected output:** 3 rows — Alice Johnson, Emma Davis, Grace Kim (ordered alphabetically).

---

### Task 2 — Fix SQL Logic / Join Error `(medium)`

| Property | Value |
|---|---|
| `task_id` | `fix_logic_error` |
| `difficulty` | medium |
| `max_steps` | 8 |
| `reward_threshold` | 0.7 |

**Objective:** The query is syntactically valid but has two semantic bugs: (1) it uses `INNER JOIN` instead of `LEFT JOIN`, dropping orders with no items; (2) it sums `orders.total_amount` (stale denormalised field) instead of computing `SUM(oi.quantity * oi.unit_price)` from line items. The agent must diagnose and fix both bugs.

**Broken query:**
```sql
SELECT o.id AS order_id, c.name AS customer_name,
       COUNT(oi.id) AS item_count,
       SUM(o.total_amount) AS computed_total     -- ← wrong column
FROM orders o
INNER JOIN customers c ON c.id = o.customer_id
INNER JOIN order_items oi ON oi.order_id = o.id  -- ← drops orders with no items
GROUP BY o.id, c.name
ORDER BY o.id;
```

**Expected output:** 11 rows (all orders), with `computed_total` computed from `order_items`.

---

### Task 3 — Optimize Slow Query `(hard)`

| Property | Value |
|---|---|
| `task_id` | `optimize_query` |
| `difficulty` | hard |
| `max_steps` | 10 |
| `reward_threshold` | 0.6 |

**Objective:** The query is functionally correct but uses two correlated scalar subqueries — an N+1 pattern that re-scans `orders` and `order_items` for every customer row. The agent must rewrite using a single `JOIN + GROUP BY` for linear-time execution, while preserving identical output (including customers with zero orders).

**Slow query:**
```sql
SELECT c.id, c.name, c.region, c.tier,
  (SELECT COALESCE(SUM(oi2.quantity * oi2.unit_price), 0)
   FROM orders o2 JOIN order_items oi2 ON oi2.order_id = o2.id
   WHERE o2.customer_id = c.id AND o2.status != 'cancelled') AS total_revenue,
  (SELECT COUNT(*)
   FROM orders o3
   WHERE o3.customer_id = c.id AND o3.status != 'cancelled') AS order_count
FROM customers c
ORDER BY total_revenue DESC;
```

**Expected output:** 10 rows — all customers with their computed revenue and order count, ordered by revenue descending. Cancelled orders excluded. Customers with no (non-cancelled) orders show `total_revenue=0, order_count=0`.

---

## 📦 Action Space

The agent submits a `SQLAction` object at each step:

```python
class SQLAction(BaseModel):
    sql_query: str           # SQL to execute (required)
    reasoning: Optional[str] # Chain-of-thought (optional, logged not evaluated)
```

**JSON example:**
```json
{
  "action": {
    "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
    "reasoning": "Fixed SELEC→SELECT, FORM→FROM, ORDR→ORDER BY."
  }
}
```

---

## 👁️ Observation Space

The environment returns a `SQLObservation` after every `reset()` and `step()`:

```python
class SQLObservation(BaseModel):
    # Task context (always present)
    task_id: str                            # Active task identifier
    task_description: str                   # Full task description
    broken_query: str                       # The original broken/slow query
    schema_hint: str                        # DDL + sample data for context

    # Execution feedback (None on reset, populated after step)
    error_message: Optional[str]            # SQLite error if query failed
    query_result: Optional[list[dict]]      # Up to 20 result rows
    execution_time_ms: Optional[float]      # Query wall-clock time

    # Reward signal
    reward: float                           # Scalar reward [0.0, 1.0]
    reward_breakdown: Optional[RewardBreakdown]  # Detailed decomposition

    # Episode metadata
    step_count: int                         # Steps used this episode
    max_steps: int                          # Episode step limit
    done: bool                              # Whether episode has ended
```

---

## ⚖️ Reward Function

Rewards are **non-sparse** — meaningful signal at every step of the trajectory.

### Easy task (`fix_syntax_error`)

```
reward = max(correctness × step_penalty, 0.25)    if query executes
reward = 0.05                                       if query fails with error

correctness  = fraction of expected rows matched (with position weighting)
step_penalty = 1.0 - 0.3 × (steps_used / max_steps)   ∈ [0.70, 1.00]
```

### Medium task (`fix_logic_error`)

```
reward = correctness × step_penalty    if query executes
reward = 0.05                           if error

correctness = fraction of expected rows present in result (unordered)
              partial credit per matched row
```

### Hard task (`optimize_query`)

```
reward = (0.6 × correctness + 0.4 × efficiency) × step_penalty

correctness = fraction of expected rows matched (float tolerance ±0.05)
efficiency  = min(baseline_ms / exec_ms / 5.0, 1.0)   if exec_ms < baseline_ms
            = 0.0                                        if exec_ms ≥ baseline_ms
```

**Gradient properties:**
- `error (0.05) < executes_wrong (0.25) < executes_partial (0.3–0.7) < correct (0.8+)`
- Step penalty ensures agents are rewarded for solving tasks in fewer steps
- Hard task rewards both correctness AND performance — agents must reason about query planning

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- Docker
- Hugging Face CLI (`huggingface-cli`)
- Git

### 1. Install locally

```bash
git clone https://github.com/your-username/sql-debug-env
cd sql-debug-env

# With uv (recommended — fast)
pip install uv
uv sync

# Or with pip
pip install -e "."
```

### 2. Run the server

```bash
# With uv
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or directly
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Verify it's up:
```bash
curl http://localhost:7860/health
# → {"status": "ok", "environment": "sql-debug-env", "version": "1.0.0"}
```

### 3. Use the API

```bash
# Reset (start new episode)
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fix_syntax_error"}' | python3 -m json.tool

# Submit a query (step)
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "sql_query": "SELECT id, name, email FROM customers WHERE tier = '\''vip'\'' ORDER BY name;",
      "reasoning": "Fixed SELEC→SELECT, FORM→FROM, ORDR→ORDER BY."
    }
  }' | python3 -m json.tool

# Get state
curl http://localhost:7860/state
```

### 4. Use the Python client

```python
from client import SQLDebugEnv
from models import SQLAction

with SQLDebugEnv(base_url="http://localhost:7860").sync() as env:
    # Reset for any of the 3 tasks
    obs = env.reset(task_id="fix_syntax_error")
    print(f"Broken query: {obs.broken_query}")
    print(f"Schema: {obs.schema_hint}")

    # Submit a fix
    obs = env.step(SQLAction(
        sql_query="SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
        reasoning="Fixed all three typos."
    ))
    print(f"Reward: {obs.reward:.4f}")
    print(f"Done: {obs.done}")
    print(f"Result: {obs.query_result}")
```

### 5. Docker

```bash
# Build
docker build -t sql-debug-env:latest .

# Run
docker run -p 7860:7860 sql-debug-env:latest

# With optional Gradio UI
docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true sql-debug-env:latest

# Verify
curl http://localhost:7860/health
```

### 6. Run tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

### 7. Run inference script (baseline)

```bash
# Start server first (separate terminal)
uvicorn server.app:app --port 7860

# Run baseline
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=hf_your_token_here \
ENV_BASE_URL=http://localhost:7860 \
python inference.py
```

### 8. Deploy to Hugging Face Spaces

```bash
huggingface-cli login

# Create a new Space (Docker SDK)
huggingface-cli repo create sql-debug-env --type space --space-sdk docker

# Push
git remote add hf https://huggingface.co/spaces/your-username/sql-debug-env
git push hf main

# Verify deployment
curl https://your-username-sql-debug-env.hf.space/health
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe — returns `{"status": "ok"}` |
| `/info` | GET | Environment metadata and task list |
| `/tasks` | GET | All tasks with difficulty, max_steps, description |
| `/reset` | POST | Start new episode — body: `{"task_id": "..."}` |
| `/step` | POST | Execute action — body: `{"action": {"sql_query": "..."}}` |
| `/state` | GET | Current episode state |
| `/docs` | GET | Interactive Swagger UI |
| `/web` | GET | Gradio UI (if `ENABLE_WEB_INTERFACE=true`) |

---

## 📊 Baseline Scores

Baseline run using `Qwen/Qwen2.5-72B-Instruct` via HF Inference API, `MAX_STEPS=6`, `TEMPERATURE=0.1`:

| Task | Score | Notes |
|---|---|---|
| `fix_syntax_error` | 1.0000 | Solved on step 1 every run |
| `fix_logic_error` | 0.9091 | Solved step 1 or 2; misses edge-case cancelled orders |
| `optimize_query` | 0.8800 | Correctness 0.90, efficiency varies by run |
| **Mean** | **0.9297** | Reproducible across 3 runs |

Run it yourself:
```bash
python inference.py
# Outputs JSON: {"scores": {...}, "mean_score": 0.9297, ...}
```

---

## 🔒 Pre-Submission Checklist

- [x] `HF Space deploys` — `GET /health` returns 200 and `/reset` responds correctly  
- [x] `OpenEnv spec compliance` — `openenv.yaml` with metadata, typed models, all 3 endpoints  
- [x] `Dockerfile builds` — multi-stage, non-root user, HEALTHCHECK configured  
- [x] `Baseline reproduces` — `inference.py` at root, uses OpenAI client, < 20 min runtime  
- [x] `3+ tasks with graders` — easy/medium/hard, all scores in `[0.0, 1.0]`, fully deterministic  
- [x] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars documented and used  
- [x] `inference.py` named correctly and placed in root  
- [x] Runs on 2 vCPU / 8 GB (SQLite in-memory, no GPU required)

---

## 📝 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (inference) | LLM API endpoint e.g. `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Yes (inference) | Model identifier e.g. `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Yes (inference) | Hugging Face API key |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:7860`) |
| `ENABLE_WEB_INTERFACE` | No | Set to `true` to mount Gradio UI at `/web` |

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Built for the [Scaler × Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/) using the [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch.
