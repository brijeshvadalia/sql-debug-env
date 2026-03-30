"""
server/tasks/__init__.py

Defines the three hackathon tasks:
  1. fix_syntax_error  (easy)   — broken DDL keyword, missing FROM
  2. fix_logic_error   (medium) — wrong JOIN type + wrong aggregation column
  3. optimize_query    (hard)   — N+1 scalar subquery rewrite required

Each Task dataclass is the single source of truth:
  - broken_query    : what the agent sees
  - correct_query   : reference answer used by the grader
  - expected_result : deterministic expected rows
  - schema_hint     : context given to the agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    name: str
    difficulty: str                         # easy | medium | hard
    description: str
    broken_query: str                       # query shown to the agent
    correct_query: str                      # reference (used only internally)
    expected_result: list[dict[str, Any]]   # deterministic expected rows
    schema_hint: str                        # DDL + sample rows shown to agent
    max_steps: int
    reward_threshold: float
    baseline_exec_ms: float = 0.0           # used only for optimize_query


# ---------------------------------------------------------------------------
# Task 1 — Easy: Fix Syntax Error
# ---------------------------------------------------------------------------
# Broken: "SELEC", "FORM", missing semicolon (two typos, one missing token)
# Agent must fix the query so it executes and returns the correct rows.
# ---------------------------------------------------------------------------

TASK_EASY = Task(
    task_id="fix_syntax_error",
    name="Fix SQL Syntax Error",
    difficulty="easy",
    description=(
        "The following SQL query has syntax errors and will not execute. "
        "Fix all errors so the query runs successfully and returns the correct result. "
        "The goal is to retrieve the id, name, and email of all VIP-tier customers "
        "ordered by name alphabetically.\n\n"
        "Broken query:\n"
        "  SELEC id, name, email FORM customers\n"
        "  WHERE tier = 'vip'\n"
        "  ORDR BY name;"
    ),
    broken_query=(
        "SELEC id, name, email FORM customers\n"
        "WHERE tier = 'vip'\n"
        "ORDR BY name;"
    ),
    correct_query=(
        "SELECT id, name, email FROM customers\n"
        "WHERE tier = 'vip'\n"
        "ORDER BY name;"
    ),
    expected_result=[
        {"id": 1, "name": "Alice Johnson",  "email": "alice@example.com"},
        {"id": 5, "name": "Emma Davis",     "email": "emma@example.com"},
        {"id": 7, "name": "Grace Kim",      "email": "grace@example.com"},
    ],
    schema_hint=(
        "TABLE customers(\n"
        "  id         INTEGER PRIMARY KEY,\n"
        "  name       TEXT NOT NULL,\n"
        "  email      TEXT NOT NULL UNIQUE,\n"
        "  region     TEXT NOT NULL,\n"
        "  tier       TEXT NOT NULL   -- values: standard | premium | vip\n"
        ")\n\n"
        "Sample rows:\n"
        "  (1, 'Alice Johnson', 'alice@example.com', 'US', 'vip')\n"
        "  (2, 'Bob Smith',     'bob@example.com',   'EU', 'premium')\n"
        "  (3, 'Carol White',   'carol@example.com', 'US', 'standard')"
    ),
    max_steps=5,
    reward_threshold=0.8,
)


# ---------------------------------------------------------------------------
# Task 2 — Medium: Fix Logic / Join Error
# ---------------------------------------------------------------------------
# The query uses INNER JOIN instead of LEFT JOIN, causing cancelled orders
# to vanish, and aggregates the wrong column (o.total_amount instead of
# computing SUM(oi.quantity * oi.unit_price)).
#
# Agent must return: order_id, customer_name, item_count, computed_total
# for ALL orders (including those with no shipped items).
# ---------------------------------------------------------------------------

TASK_MEDIUM = Task(
    task_id="fix_logic_error",
    name="Fix SQL Logic / Join Error",
    difficulty="medium",
    description=(
        "The query below is syntactically valid but logically incorrect. "
        "It should return for every order: the order id, the customer name, "
        "the number of line items, and the correct computed total "
        "(SUM of quantity × unit_price per item). "
        "The current query has two bugs: (1) it drops orders that have no shipped "
        "items due to an incorrect JOIN type, and (2) it sums the wrong column. "
        "Fix both bugs.\n\n"
        "Broken query:\n"
        "  SELECT o.id AS order_id, c.name AS customer_name,\n"
        "         COUNT(oi.id) AS item_count,\n"
        "         SUM(o.total_amount) AS computed_total\n"
        "  FROM orders o\n"
        "  INNER JOIN customers c ON c.id = o.customer_id\n"
        "  INNER JOIN order_items oi ON oi.order_id = o.id\n"
        "  GROUP BY o.id, c.name\n"
        "  ORDER BY o.id;"
    ),
    broken_query=(
        "SELECT o.id AS order_id, c.name AS customer_name,\n"
        "       COUNT(oi.id) AS item_count,\n"
        "       SUM(o.total_amount) AS computed_total\n"
        "FROM orders o\n"
        "INNER JOIN customers c ON c.id = o.customer_id\n"
        "INNER JOIN order_items oi ON oi.order_id = o.id\n"
        "GROUP BY o.id, c.name\n"
        "ORDER BY o.id;"
    ),
    correct_query=(
        "SELECT o.id AS order_id, c.name AS customer_name,\n"
        "       COUNT(oi.id) AS item_count,\n"
        "       SUM(oi.quantity * oi.unit_price) AS computed_total\n"
        "FROM orders o\n"
        "JOIN customers c ON c.id = o.customer_id\n"
        "LEFT JOIN order_items oi ON oi.order_id = o.id\n"
        "GROUP BY o.id, c.name\n"
        "ORDER BY o.id;"
    ),
    expected_result=[
        {"order_id": 1,  "customer_name": "Alice Johnson", "item_count": 2, "computed_total": 1329.98},
        {"order_id": 2,  "customer_name": "Alice Johnson", "item_count": 1, "computed_total": 49.99},
        {"order_id": 3,  "customer_name": "Bob Smith",     "item_count": 2, "computed_total": 629.93},
        {"order_id": 4,  "customer_name": "Carol White",   "item_count": 2, "computed_total": 14.98},
        {"order_id": 5,  "customer_name": "David Lee",     "item_count": 1, "computed_total": 349.99},
        {"order_id": 6,  "customer_name": "Emma Davis",    "item_count": 3, "computed_total": 1929.97},
        {"order_id": 7,  "customer_name": "Frank Brown",   "item_count": 1, "computed_total": 89.99},
        {"order_id": 8,  "customer_name": "Grace Kim",     "item_count": 2, "computed_total": 629.98},
        {"order_id": 9,  "customer_name": "Henry Wilson",  "item_count": 1, "computed_total": 79.98},
        {"order_id": 10, "customer_name": "Iris Chen",     "item_count": 1, "computed_total": 399.99},
        {"order_id": 11, "customer_name": "Jack Taylor",   "item_count": 1, "computed_total": 29.99},
    ],
    schema_hint=(
        "TABLE customers(id, name, email, region, tier, created_at)\n"
        "TABLE orders(id, customer_id FK→customers.id, status, created_at, shipped_at, total_amount)\n"
        "TABLE order_items(id, order_id FK→orders.id, product_id FK→products.id, quantity, unit_price)\n\n"
        "Key facts:\n"
        "  - orders.status ∈ {pending, shipped, delivered, cancelled}\n"
        "  - orders.total_amount is a denormalised field and may be stale — "
        "    compute totals from order_items instead\n"
        "  - Not all orders have line items (edge case to handle)\n"
        "  - The result must include ALL orders, even those with no shipped items\n\n"
        "Sample order_items rows:\n"
        "  (order_id=1, product_id=1, quantity=1, unit_price=1299.99)\n"
        "  (order_id=1, product_id=2, quantity=1, unit_price=29.99)"
    ),
    max_steps=8,
    reward_threshold=0.7,
)


# ---------------------------------------------------------------------------
# Task 3 — Hard: Optimize Slow Query
# ---------------------------------------------------------------------------
# The broken query uses a correlated scalar subquery (N+1) to compute
# revenue per customer, causing O(n) full-table scans.
# The agent must rewrite using a single JOIN + GROUP BY.
# Graded: 60% correctness + 40% speedup vs baseline.
# ---------------------------------------------------------------------------

TASK_HARD = Task(
    task_id="optimize_query",
    name="Optimize Slow SQL Query",
    difficulty="hard",
    description=(
        "The query below is functionally correct but extremely slow. "
        "It uses a correlated scalar subquery that re-scans order_items for every "
        "customer row (an N+1 pattern). On large datasets this is O(n²). "
        "Rewrite the query to produce identical results — "
        "customer id, name, region, tier, total_revenue, order_count — "
        "while eliminating the correlated subquery. "
        "The rewritten query must be faster than the original "
        "AND return the same result set (including customers with zero orders).\n\n"
        "Broken/slow query:\n"
        "  SELECT\n"
        "    c.id,\n"
        "    c.name,\n"
        "    c.region,\n"
        "    c.tier,\n"
        "    (\n"
        "      SELECT COALESCE(SUM(oi2.quantity * oi2.unit_price), 0)\n"
        "      FROM orders o2\n"
        "      JOIN order_items oi2 ON oi2.order_id = o2.id\n"
        "      WHERE o2.customer_id = c.id\n"
        "        AND o2.status != 'cancelled'\n"
        "    ) AS total_revenue,\n"
        "    (\n"
        "      SELECT COUNT(*)\n"
        "      FROM orders o3\n"
        "      WHERE o3.customer_id = c.id\n"
        "        AND o3.status != 'cancelled'\n"
        "    ) AS order_count\n"
        "  FROM customers c\n"
        "  ORDER BY total_revenue DESC;"
    ),
    broken_query=(
        "SELECT\n"
        "  c.id,\n"
        "  c.name,\n"
        "  c.region,\n"
        "  c.tier,\n"
        "  (\n"
        "    SELECT COALESCE(SUM(oi2.quantity * oi2.unit_price), 0)\n"
        "    FROM orders o2\n"
        "    JOIN order_items oi2 ON oi2.order_id = o2.id\n"
        "    WHERE o2.customer_id = c.id\n"
        "      AND o2.status != 'cancelled'\n"
        "  ) AS total_revenue,\n"
        "  (\n"
        "    SELECT COUNT(*)\n"
        "    FROM orders o3\n"
        "    WHERE o3.customer_id = c.id\n"
        "      AND o3.status != 'cancelled'\n"
        "  ) AS order_count\n"
        "FROM customers c\n"
        "ORDER BY total_revenue DESC;"
    ),
    correct_query=(
        "SELECT\n"
        "  c.id,\n"
        "  c.name,\n"
        "  c.region,\n"
        "  c.tier,\n"
        "  COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue,\n"
        "  COUNT(DISTINCT o.id)                          AS order_count\n"
        "FROM customers c\n"
        "LEFT JOIN orders o\n"
        "  ON o.customer_id = c.id AND o.status != 'cancelled'\n"
        "LEFT JOIN order_items oi ON oi.order_id = o.id\n"
        "GROUP BY c.id, c.name, c.region, c.tier\n"
        "ORDER BY total_revenue DESC;"
    ),
    expected_result=[
        {"id": 5,  "name": "Emma Davis",    "region": "EU",   "tier": "vip",      "total_revenue": 1929.97, "order_count": 1},
        {"id": 1,  "name": "Alice Johnson", "region": "US",   "tier": "vip",      "total_revenue": 1379.97, "order_count": 2},
        {"id": 7,  "name": "Grace Kim",     "region": "APAC", "tier": "vip",      "total_revenue": 629.98,  "order_count": 1},
        {"id": 2,  "name": "Bob Smith",     "region": "EU",   "tier": "premium",  "total_revenue": 629.93,  "order_count": 1},
        {"id": 9,  "name": "Iris Chen",     "region": "US",   "tier": "premium",  "total_revenue": 399.99,  "order_count": 1},
        {"id": 4,  "name": "David Lee",     "region": "APAC", "tier": "premium",  "total_revenue": 349.99,  "order_count": 1},
        {"id": 8,  "name": "Henry Wilson",  "region": "EU",   "tier": "standard", "total_revenue": 79.98,   "order_count": 1},
        {"id": 3,  "name": "Carol White",   "region": "US",   "tier": "standard", "total_revenue": 14.98,   "order_count": 1},
        {"id": 10, "name": "Jack Taylor",   "region": "US",   "tier": "standard", "total_revenue": 29.99,   "order_count": 1},
        {"id": 6,  "name": "Frank Brown",   "region": "US",   "tier": "standard", "total_revenue": 0.0,     "order_count": 0},
    ],
    schema_hint=(
        "TABLE customers(id, name, region, tier, email, created_at)\n"
        "TABLE orders(id, customer_id FK→customers.id, status, created_at, shipped_at, total_amount)\n"
        "  status ∈ {pending, shipped, delivered, cancelled}\n"
        "TABLE order_items(id, order_id FK→orders.id, product_id, quantity, unit_price)\n\n"
        "Performance hint:\n"
        "  The current query uses TWO correlated scalar subqueries — one for revenue,\n"
        "  one for order_count. Each re-scans orders and order_items per customer row.\n"
        "  Rewrite with a single JOIN + GROUP BY to achieve a linear scan.\n\n"
        "Required output columns (exact names):\n"
        "  id, name, region, tier, total_revenue, order_count\n"
        "  Ordered by total_revenue DESC.\n"
        "  Cancelled orders excluded. Customers with no (non-cancelled) orders: total_revenue=0, order_count=0."
    ),
    max_steps=10,
    reward_threshold=0.6,
    baseline_exec_ms=5.0,   # Baseline time recorded at environment startup
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, Task] = {
    TASK_EASY.task_id:   TASK_EASY,
    TASK_MEDIUM.task_id: TASK_MEDIUM,
    TASK_HARD.task_id:   TASK_HARD,
}

__all__ = ["Task", "TASKS", "TASK_EASY", "TASK_MEDIUM", "TASK_HARD"]
