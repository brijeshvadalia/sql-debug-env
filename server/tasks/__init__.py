"""
server/tasks/__init__.py — 8 Tasks across 4 difficulty levels
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Task:
    task_id: str
    name: str
    difficulty: str
    description: str
    broken_query: str
    correct_query: str
    expected_result: list
    schema_hint: str
    max_steps: int
    reward_threshold: float
    baseline_exec_ms: float = 0.0
    tags: list = field(default_factory=list)

TASK_EASY = Task(
    task_id="fix_syntax_error",
    name="Fix SQL Syntax Error",
    difficulty="easy",
    description=(
        "The following SQL query has syntax errors and will not execute. "
        "Fix all errors so the query runs successfully and returns the correct result. "
        "Retrieve the id, name, and email of all VIP-tier customers ordered by name.\n\n"
        "Broken query:\n  SELEC id, name, email FORM customers\n  WHERE tier = 'vip'\n  ORDR BY name;"
    ),
    broken_query="SELEC id, name, email FORM customers\nWHERE tier = 'vip'\nORDR BY name;",
    correct_query="SELECT id, name, email FROM customers\nWHERE tier = 'vip'\nORDER BY name;",
    expected_result=[
        {"id": 1, "name": "Alice Johnson",  "email": "alice@example.com"},
        {"id": 5, "name": "Emma Davis",     "email": "emma@example.com"},
        {"id": 7, "name": "Grace Kim",      "email": "grace@example.com"},
    ],
    schema_hint=(
        "TABLE customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE,\n"
        "                region TEXT, tier TEXT)\n"
        "  tier values: standard | premium | vip\n\n"
        "Sample rows:\n"
        "  (1,'Alice Johnson','alice@example.com','US','vip')\n"
        "  (2,'Bob Smith','bob@example.com','EU','premium')"
    ),
    max_steps=5,
    reward_threshold=0.8,
    tags=["syntax","easy","beginner"],
)

TASK_MEDIUM = Task(
    task_id="fix_logic_error",
    name="Fix SQL Logic / Join Error",
    difficulty="medium",
    description=(
        "The query is syntactically valid but has two logic bugs:\n"
        "  1. Uses INNER JOIN — drops orders with no items\n"
        "  2. Sums orders.total_amount (stale) instead of SUM(oi.quantity * oi.unit_price)\n"
        "Fix both bugs. Return ALL orders with correct computed_total.\n\n"
        "Broken query:\n"
        "  SELECT o.id AS order_id, c.name AS customer_name,\n"
        "         COUNT(oi.id) AS item_count, SUM(o.total_amount) AS computed_total\n"
        "  FROM orders o\n"
        "  INNER JOIN customers c ON c.id = o.customer_id\n"
        "  INNER JOIN order_items oi ON oi.order_id = o.id\n"
        "  GROUP BY o.id, c.name ORDER BY o.id;"
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
        "TABLE customers(id, name, email, region, tier)\n"
        "TABLE orders(id, customer_id FK->customers.id, status, total_amount)\n"
        "  total_amount is STALE - compute from order_items instead\n"
        "TABLE order_items(id, order_id FK->orders.id, product_id, quantity, unit_price)\n\n"
        "Fixes needed:\n"
        "  1. INNER JOIN order_items -> LEFT JOIN order_items\n"
        "  2. SUM(o.total_amount) -> SUM(oi.quantity * oi.unit_price)"
    ),
    max_steps=8,
    reward_threshold=0.7,
    tags=["join","aggregation","logic","medium"],
)

TASK_NULL = Task(
    task_id="fix_null_handling",
    name="Fix NULL Handling in Aggregation",
    difficulty="medium",
    description=(
        "The query calculates average rating per active product. Two bugs:\n"
        "  1. INNER JOIN excludes products with zero reviews\n"
        "  2. AVG returns NULL for unreviewed products instead of 0.0\n"
        "Fix to include ALL active products, showing 0.0 for unreviewed ones.\n\n"
        "Broken query:\n"
        "  SELECT p.id, p.name, p.category,\n"
        "         AVG(r.rating) AS avg_rating, COUNT(r.id) AS review_count\n"
        "  FROM products p\n"
        "  INNER JOIN reviews r ON r.product_id = p.id\n"
        "  WHERE p.active = 1\n"
        "  GROUP BY p.id, p.name, p.category\n"
        "  ORDER BY avg_rating DESC, p.name ASC;"
    ),
    broken_query=(
        "SELECT p.id, p.name, p.category,\n"
        "       AVG(r.rating) AS avg_rating,\n"
        "       COUNT(r.id) AS review_count\n"
        "FROM products p\n"
        "INNER JOIN reviews r ON r.product_id = p.id\n"
        "WHERE p.active = 1\n"
        "GROUP BY p.id, p.name, p.category\n"
        "ORDER BY avg_rating DESC, p.name ASC;"
    ),
    correct_query=(
        "SELECT p.id, p.name, p.category,\n"
        "       COALESCE(AVG(r.rating), 0.0) AS avg_rating,\n"
        "       COUNT(r.id) AS review_count\n"
        "FROM products p\n"
        "LEFT JOIN reviews r ON r.product_id = p.id\n"
        "WHERE p.active = 1\n"
        "GROUP BY p.id, p.name, p.category\n"
        "ORDER BY avg_rating DESC, p.name ASC;"
    ),
    expected_result=[
        {"id": 5,  "name": "Ergonomic Chair",     "category": "furniture",   "avg_rating": 5.0, "review_count": 1},
        {"id": 3,  "name": "USB-C Hub",           "category": "electronics", "avg_rating": 5.0, "review_count": 1},
        {"id": 1,  "name": "Laptop Pro 15",       "category": "electronics", "avg_rating": 4.5, "review_count": 2},
        {"id": 8,  "name": "Monitor 27\"",         "category": "electronics", "avg_rating": 4.0, "review_count": 1},
        {"id": 2,  "name": "Wireless Mouse",      "category": "electronics", "avg_rating": 3.0, "review_count": 1},
        {"id": 9,  "name": "Keyboard Mechanical", "category": "electronics", "avg_rating": 2.0, "review_count": 1},
        {"id": 7,  "name": "Ballpoint Pens (10)", "category": "stationery",  "avg_rating": 0.0, "review_count": 0},
        {"id": 6,  "name": "Notebook Pack (3)",   "category": "stationery",  "avg_rating": 0.0, "review_count": 0},
        {"id": 4,  "name": "Standing Desk",       "category": "furniture",   "avg_rating": 0.0, "review_count": 0},
        {"id": 10, "name": "Webcam HD",           "category": "electronics", "avg_rating": 0.0, "review_count": 0},
    ],
    schema_hint=(
        "TABLE products(id, name, category, unit_price, stock, active)\n"
        "  active=1 available, active=0 discontinued\n"
        "TABLE reviews(id, product_id FK->products.id, customer_id, rating, body)\n"
        "  rating INTEGER 1-5\n\n"
        "Fixes:\n"
        "  1. INNER JOIN reviews -> LEFT JOIN reviews\n"
        "  2. AVG(r.rating) -> COALESCE(AVG(r.rating), 0.0)\n\n"
        "Only include active=1 products (exclude Old Fax Machine)"
    ),
    max_steps=6,
    reward_threshold=0.75,
    tags=["null","coalesce","left-join","medium"],
)

TASK_SUBQUERY = Task(
    task_id="fix_subquery_bug",
    name="Fix Correlated Subquery Bug",
    difficulty="medium",
    description=(
        "The query finds customers who spent more than average order value.\n"
        "Bug: inner subquery uses alias 'oi' which shadows outer query's 'oi',\n"
        "causing wrong results. Fix by renaming inner alias to 'oi2'.\n\n"
        "Broken query:\n"
        "  SELECT c.id, c.name, c.tier,\n"
        "         SUM(oi.quantity * oi.unit_price) AS total_spent\n"
        "  FROM customers c\n"
        "  JOIN orders o ON o.customer_id = c.id\n"
        "  JOIN order_items oi ON oi.order_id = o.id\n"
        "  WHERE o.status != 'cancelled'\n"
        "  GROUP BY c.id, c.name, c.tier\n"
        "  HAVING total_spent > (\n"
        "    SELECT AVG(total) FROM (\n"
        "      SELECT SUM(oi.quantity * oi.unit_price) AS total\n"
        "      FROM orders ord\n"
        "      JOIN order_items oi ON oi.order_id = ord.id\n"
        "      WHERE ord.status != 'cancelled'\n"
        "      GROUP BY ord.id\n"
        "    )\n"
        "  ) ORDER BY total_spent DESC;"
    ),
    broken_query=(
        "SELECT c.id, c.name, c.tier,\n"
        "       SUM(oi.quantity * oi.unit_price) AS total_spent\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "WHERE o.status != 'cancelled'\n"
        "GROUP BY c.id, c.name, c.tier\n"
        "HAVING total_spent > (\n"
        "  SELECT AVG(total) FROM (\n"
        "    SELECT SUM(oi.quantity * oi.unit_price) AS total\n"
        "    FROM orders ord\n"
        "    JOIN order_items oi ON oi.order_id = ord.id\n"
        "    WHERE ord.status != 'cancelled'\n"
        "    GROUP BY ord.id\n"
        "  )\n"
        ")\n"
        "ORDER BY total_spent DESC;"
    ),
    correct_query=(
        "SELECT c.id, c.name, c.tier,\n"
        "       SUM(oi.quantity * oi.unit_price) AS total_spent\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "WHERE o.status != 'cancelled'\n"
        "GROUP BY c.id, c.name, c.tier\n"
        "HAVING total_spent > (\n"
        "  SELECT AVG(total) FROM (\n"
        "    SELECT SUM(oi2.quantity * oi2.unit_price) AS total\n"
        "    FROM orders ord\n"
        "    JOIN order_items oi2 ON oi2.order_id = ord.id\n"
        "    WHERE ord.status != 'cancelled'\n"
        "    GROUP BY ord.id\n"
        "  )\n"
        ")\n"
        "ORDER BY total_spent DESC;"
    ),
    expected_result=[
        {"id": 5, "name": "Emma Davis",    "tier": "vip",     "total_spent": 1929.97},
        {"id": 1, "name": "Alice Johnson", "tier": "vip",     "total_spent": 1379.97},
        {"id": 7, "name": "Grace Kim",     "tier": "vip",     "total_spent": 629.98},
        {"id": 2, "name": "Bob Smith",     "tier": "premium", "total_spent": 629.93},
    ],
    schema_hint=(
        "TABLE customers(id, name, tier)\n"
        "TABLE orders(id, customer_id, status)\n"
        "TABLE order_items(id, order_id, quantity, unit_price)\n\n"
        "Bug: alias 'oi' used in both outer and inner query.\n"
        "Fix: rename inner subquery alias from 'oi' to 'oi2'\n\n"
        "Expected: 2 customers whose total > avg order value (~560)"
    ),
    max_steps=6,
    reward_threshold=0.75,
    tags=["subquery","alias","having","medium"],
)

TASK_HARD = Task(
    task_id="optimize_query",
    name="Optimize Slow SQL Query",
    difficulty="hard",
    description=(
        "Functionally correct but extremely slow — uses TWO correlated scalar subqueries (N+1 pattern).\n"
        "Rewrite using a single JOIN + GROUP BY to produce identical results faster.\n"
        "Must include customers with zero orders (total_revenue=0, order_count=0).\n\n"
        "Broken/slow query:\n"
        "  SELECT c.id, c.name, c.region, c.tier,\n"
        "    (SELECT COALESCE(SUM(oi2.quantity*oi2.unit_price),0)\n"
        "     FROM orders o2 JOIN order_items oi2 ON oi2.order_id=o2.id\n"
        "     WHERE o2.customer_id=c.id AND o2.status!='cancelled') AS total_revenue,\n"
        "    (SELECT COUNT(*) FROM orders o3\n"
        "     WHERE o3.customer_id=c.id AND o3.status!='cancelled') AS order_count\n"
        "  FROM customers c ORDER BY total_revenue DESC;"
    ),
    broken_query=(
        "SELECT\n"
        "  c.id, c.name, c.region, c.tier,\n"
        "  (SELECT COALESCE(SUM(oi2.quantity * oi2.unit_price), 0)\n"
        "   FROM orders o2 JOIN order_items oi2 ON oi2.order_id = o2.id\n"
        "   WHERE o2.customer_id = c.id AND o2.status != 'cancelled') AS total_revenue,\n"
        "  (SELECT COUNT(*) FROM orders o3\n"
        "   WHERE o3.customer_id = c.id AND o3.status != 'cancelled') AS order_count\n"
        "FROM customers c\n"
        "ORDER BY total_revenue DESC;"
    ),
    correct_query=(
        "SELECT c.id, c.name, c.region, c.tier,\n"
        "       COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_revenue,\n"
        "       COUNT(DISTINCT o.id) AS order_count\n"
        "FROM customers c\n"
        "LEFT JOIN orders o ON o.customer_id = c.id AND o.status != 'cancelled'\n"
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
        "TABLE customers(id, name, region, tier)\n"
        "TABLE orders(id, customer_id, status)\n"
        "TABLE order_items(id, order_id, quantity, unit_price)\n\n"
        "Fix: Replace 2 correlated subqueries with:\n"
        "  LEFT JOIN orders o ON o.customer_id=c.id AND o.status!='cancelled'\n"
        "  LEFT JOIN order_items oi ON oi.order_id=o.id\n"
        "  GROUP BY c.id, c.name, c.region, c.tier\n"
        "  Use COALESCE(SUM(...),0) and COUNT(DISTINCT o.id)"
    ),
    max_steps=10,
    reward_threshold=0.6,
    baseline_exec_ms=5.0,
    tags=["performance","n+1","optimization","hard"],
)

TASK_WINDOW = Task(
    task_id="fix_window_function",
    name="Fix Window Function (RANK)",
    difficulty="hard",
    description=(
        "Query ranks customers by spending within each group using a window function.\n"
        "Two bugs:\n"
        "  1. Uses ROW_NUMBER() instead of RANK() — ties broken arbitrarily\n"
        "  2. PARTITION BY c.tier instead of PARTITION BY c.region\n"
        "Fix to rank within REGION using RANK().\n\n"
        "Broken query:\n"
        "  SELECT c.id, c.name, c.region, c.tier,\n"
        "         SUM(oi.quantity * oi.unit_price) AS total_spent,\n"
        "         ROW_NUMBER() OVER (\n"
        "           PARTITION BY c.tier\n"
        "           ORDER BY SUM(oi.quantity * oi.unit_price) DESC\n"
        "         ) AS region_rank\n"
        "  FROM customers c\n"
        "  JOIN orders o ON o.customer_id = c.id\n"
        "  JOIN order_items oi ON oi.order_id = o.id\n"
        "  WHERE o.status != 'cancelled'\n"
        "  GROUP BY c.id, c.name, c.region, c.tier\n"
        "  ORDER BY c.region, region_rank;"
    ),
    broken_query=(
        "SELECT c.id, c.name, c.region, c.tier,\n"
        "       SUM(oi.quantity * oi.unit_price) AS total_spent,\n"
        "       ROW_NUMBER() OVER (\n"
        "         PARTITION BY c.tier\n"
        "         ORDER BY SUM(oi.quantity * oi.unit_price) DESC\n"
        "       ) AS region_rank\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "WHERE o.status != 'cancelled'\n"
        "GROUP BY c.id, c.name, c.region, c.tier\n"
        "ORDER BY c.region, region_rank;"
    ),
    correct_query=(
        "SELECT c.id, c.name, c.region, c.tier,\n"
        "       SUM(oi.quantity * oi.unit_price) AS total_spent,\n"
        "       RANK() OVER (\n"
        "         PARTITION BY c.region\n"
        "         ORDER BY SUM(oi.quantity * oi.unit_price) DESC\n"
        "       ) AS region_rank\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "WHERE o.status != 'cancelled'\n"
        "GROUP BY c.id, c.name, c.region, c.tier\n"
        "ORDER BY c.region, region_rank;"
    ),
    expected_result=[
        {"id": 7,  "name": "Grace Kim",     "region": "APAC", "tier": "vip",      "total_spent": 629.98,  "region_rank": 1},
        {"id": 4,  "name": "David Lee",     "region": "APAC", "tier": "premium",  "total_spent": 349.99,  "region_rank": 2},
        {"id": 5,  "name": "Emma Davis",    "region": "EU",   "tier": "vip",      "total_spent": 1929.97, "region_rank": 1},
        {"id": 2,  "name": "Bob Smith",     "region": "EU",   "tier": "premium",  "total_spent": 629.93,  "region_rank": 2},
        {"id": 8,  "name": "Henry Wilson",  "region": "EU",   "tier": "standard", "total_spent": 159.96,  "region_rank": 3},
        {"id": 1,  "name": "Alice Johnson", "region": "US",   "tier": "vip",      "total_spent": 1379.97, "region_rank": 1},
        {"id": 9,  "name": "Iris Chen",     "region": "US",   "tier": "premium",  "total_spent": 399.99,  "region_rank": 2},
        {"id": 10, "name": "Jack Taylor",   "region": "US",   "tier": "standard", "total_spent": 29.99,   "region_rank": 3},
        {"id": 3,  "name": "Carol White",   "region": "US",   "tier": "standard", "total_spent": 14.98,   "region_rank": 4},
    ],
    schema_hint=(
        "TABLE customers(id, name, region, tier)\n"
        "  region in {US, EU, APAC}, tier in {standard, premium, vip}\n"
        "TABLE orders(id, customer_id, status)\n"
        "TABLE order_items(id, order_id, quantity, unit_price)\n\n"
        "Fixes:\n"
        "  1. ROW_NUMBER() -> RANK()\n"
        "  2. PARTITION BY c.tier -> PARTITION BY c.region\n\n"
        "Frank Brown excluded (no non-cancelled orders)"
    ),
    max_steps=8,
    reward_threshold=0.6,
    baseline_exec_ms=5.0,
    tags=["window-function","rank","partition","hard"],
)

TASK_CTE = Task(
    task_id="fix_cte",
    name="Fix CTE (Common Table Expression)",
    difficulty="hard",
    description=(
        "Query uses a CTE to calculate each customer's revenue percentage.\n"
        "Bug: divides total_revenue by itself (always = 100%) instead of grand total.\n"
        "Fix: divide by (SELECT SUM(total_revenue) FROM customer_revenue).\n\n"
        "Broken query:\n"
        "  WITH customer_revenue AS (\n"
        "    SELECT c.id, c.name, c.tier,\n"
        "           SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
        "    FROM customers c\n"
        "    JOIN orders o ON o.customer_id = c.id\n"
        "    JOIN order_items oi ON oi.order_id = o.id\n"
        "    WHERE o.status != 'cancelled'\n"
        "    GROUP BY c.id, c.name, c.tier\n"
        "  )\n"
        "  SELECT id, name, tier, total_revenue,\n"
        "         ROUND(total_revenue * 100.0 / total_revenue, 2) AS revenue_pct\n"
        "  FROM customer_revenue\n"
        "  ORDER BY total_revenue DESC;"
    ),
    broken_query=(
        "WITH customer_revenue AS (\n"
        "  SELECT c.id, c.name, c.tier,\n"
        "         SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
        "  FROM customers c\n"
        "  JOIN orders o ON o.customer_id = c.id\n"
        "  JOIN order_items oi ON oi.order_id = o.id\n"
        "  WHERE o.status != 'cancelled'\n"
        "  GROUP BY c.id, c.name, c.tier\n"
        ")\n"
        "SELECT id, name, tier, total_revenue,\n"
        "       ROUND(total_revenue * 100.0 / total_revenue, 2) AS revenue_pct\n"
        "FROM customer_revenue\n"
        "ORDER BY total_revenue DESC;"
    ),
    correct_query=(
        "WITH customer_revenue AS (\n"
        "  SELECT c.id, c.name, c.tier,\n"
        "         SUM(oi.quantity * oi.unit_price) AS total_revenue\n"
        "  FROM customers c\n"
        "  JOIN orders o ON o.customer_id = c.id\n"
        "  JOIN order_items oi ON oi.order_id = o.id\n"
        "  WHERE o.status != 'cancelled'\n"
        "  GROUP BY c.id, c.name, c.tier\n"
        ")\n"
        "SELECT id, name, tier, total_revenue,\n"
        "       ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM customer_revenue), 2) AS revenue_pct\n"
        "FROM customer_revenue\n"
        "ORDER BY total_revenue DESC;"
    ),
    expected_result=[
        {"id": 5,  "name": "Emma Davis",    "tier": "vip",      "total_revenue": 1929.97, "revenue_pct": 34.93},
        {"id": 1,  "name": "Alice Johnson", "tier": "vip",      "total_revenue": 1379.97, "revenue_pct": 24.98},
        {"id": 7,  "name": "Grace Kim",     "tier": "vip",      "total_revenue": 629.98,  "revenue_pct": 11.40},
        {"id": 2,  "name": "Bob Smith",     "tier": "premium",  "total_revenue": 629.93,  "revenue_pct": 11.40},
        {"id": 9,  "name": "Iris Chen",     "tier": "premium",  "total_revenue": 399.99,  "revenue_pct": 7.24},
        {"id": 4,  "name": "David Lee",     "tier": "premium",  "total_revenue": 349.99,  "revenue_pct": 6.33},
        {"id": 8,  "name": "Henry Wilson",  "tier": "standard", "total_revenue": 159.96,  "revenue_pct": 2.90},
        {"id": 10, "name": "Jack Taylor",   "tier": "standard", "total_revenue": 29.99,   "revenue_pct": 0.54},
        {"id": 3,  "name": "Carol White",   "tier": "standard", "total_revenue": 14.98,   "revenue_pct": 0.27},
    ],
    schema_hint=(
        "TABLE customers(id, name, tier)\n"
        "TABLE orders(id, customer_id, status)\n"
        "TABLE order_items(id, order_id, quantity, unit_price)\n\n"
        "Bug: ROUND(total_revenue * 100.0 / total_revenue, 2) = always 100.00\n"
        "Fix: ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM customer_revenue), 2)\n\n"
        "Grand total revenue ~5604.81. Frank Brown excluded."
    ),
    max_steps=8,
    reward_threshold=0.6,
    baseline_exec_ms=5.0,
    tags=["cte","percentage","hard"],
)

TASK_EXPERT = Task(
    task_id="multi_step_aggregation",
    name="Multi-Dimension Sales Report",
    difficulty="expert",
    description=(
        "Expert task: Write a complete sales report query.\n"
        "For each product CATEGORY + customer TIER combination show:\n"
        "  category, tier, unique_customers, total_units, total_revenue, avg_order_value\n\n"
        "Rules:\n"
        "  - Exclude cancelled orders\n"
        "  - Only combinations with at least 1 sale\n"
        "  - Round revenue and avg_order_value to 2 decimal places\n"
        "  - Order by total_revenue DESC\n\n"
        "Broken query (missing GROUP BY and avg_order_value):\n"
        "  SELECT p.category, c.tier,\n"
        "         COUNT(DISTINCT c.id) AS unique_customers,\n"
        "         SUM(oi.quantity) AS total_units,\n"
        "         ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue,\n"
        "         0 AS avg_order_value\n"
        "  FROM customers c\n"
        "  JOIN orders o ON o.customer_id = c.id\n"
        "  JOIN order_items oi ON oi.order_id = o.id\n"
        "  JOIN products p ON p.id = oi.product_id\n"
        "  WHERE o.status != 'cancelled'\n"
        "  ORDER BY total_revenue DESC;"
    ),
    broken_query=(
        "SELECT p.category, c.tier,\n"
        "       COUNT(DISTINCT c.id) AS unique_customers,\n"
        "       SUM(oi.quantity) AS total_units,\n"
        "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue,\n"
        "       0 AS avg_order_value\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "JOIN products p ON p.id = oi.product_id\n"
        "WHERE o.status != 'cancelled'\n"
        "ORDER BY total_revenue DESC;"
    ),
    correct_query=(
        "SELECT p.category, c.tier,\n"
        "       COUNT(DISTINCT c.id) AS unique_customers,\n"
        "       SUM(oi.quantity) AS total_units,\n"
        "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue,\n"
        "       ROUND(SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.id), 2) AS avg_order_value\n"
        "FROM customers c\n"
        "JOIN orders o ON o.customer_id = c.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "JOIN products p ON p.id = oi.product_id\n"
        "WHERE o.status != 'cancelled'\n"
        "GROUP BY p.category, c.tier\n"
        "ORDER BY total_revenue DESC;"
    ),
    expected_result=[
        {"category": "electronics", "tier": "vip",      "unique_customers": 3, "total_units": 6, "total_revenue": 2739.94, "avg_order_value": 684.99},
        {"category": "furniture",   "tier": "vip",      "unique_customers": 2, "total_units": 2, "total_revenue": 1199.98, "avg_order_value": 599.99},
        {"category": "furniture",   "tier": "premium",  "unique_customers": 2, "total_units": 2, "total_revenue": 999.98,  "avg_order_value": 499.99},
        {"category": "electronics", "tier": "premium",  "unique_customers": 1, "total_units": 1, "total_revenue": 349.99,  "avg_order_value": 349.99},
        {"category": "stationery",  "tier": "standard", "unique_customers": 2, "total_units": 4, "total_revenue": 174.94,  "avg_order_value": 87.47},
        {"category": "electronics", "tier": "standard", "unique_customers": 1, "total_units": 1, "total_revenue": 29.99,   "avg_order_value": 29.99},
        {"category": "stationery",  "tier": "premium",  "unique_customers": 1, "total_units": 6, "total_revenue": 29.94,   "avg_order_value": 29.94},
    ],
    schema_hint=(
        "TABLE customers(id, name, tier)\n"
        "TABLE orders(id, customer_id, status)\n"
        "TABLE order_items(id, order_id, product_id, quantity, unit_price)\n"
        "TABLE products(id, name, category)\n"
        "  category in {electronics, furniture, stationery}\n\n"
        "Fixes:\n"
        "  1. Add: GROUP BY p.category, c.tier\n"
        "  2. Replace '0 AS avg_order_value' with:\n"
        "     ROUND(SUM(oi.quantity*oi.unit_price)/COUNT(DISTINCT o.id),2)\n\n"
        "Expected 7 rows"
    ),
    max_steps=10,
    reward_threshold=0.6,
    tags=["aggregation","group-by","multi-dimension","expert"],
)

TASKS: dict[str, Task] = {
    TASK_EASY.task_id:     TASK_EASY,
    TASK_MEDIUM.task_id:   TASK_MEDIUM,
    TASK_NULL.task_id:     TASK_NULL,
    TASK_SUBQUERY.task_id: TASK_SUBQUERY,
    TASK_HARD.task_id:     TASK_HARD,
    TASK_WINDOW.task_id:   TASK_WINDOW,
    TASK_CTE.task_id:      TASK_CTE,
    TASK_EXPERT.task_id:   TASK_EXPERT,
}

__all__ = [
    "Task", "TASKS",
    "TASK_EASY", "TASK_MEDIUM", "TASK_NULL", "TASK_SUBQUERY",
    "TASK_HARD", "TASK_WINDOW", "TASK_CTE", "TASK_EXPERT",
]
