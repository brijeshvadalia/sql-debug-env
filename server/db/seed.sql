-- =============================================================================
-- sql-debug-env  ·  E-Commerce Database Seed
-- SQLite 3 compatible
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- CUSTOMERS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    email      TEXT    NOT NULL UNIQUE,
    region     TEXT    NOT NULL DEFAULT 'US',
    tier       TEXT    NOT NULL DEFAULT 'standard',   -- standard | premium | vip
    created_at TEXT    NOT NULL DEFAULT (DATE('now'))
);

INSERT INTO customers (name, email, region, tier, created_at) VALUES
  ('Alice Johnson',   'alice@example.com',   'US', 'vip',      '2023-01-15'),
  ('Bob Smith',       'bob@example.com',     'EU', 'premium',  '2023-02-10'),
  ('Carol White',     'carol@example.com',   'US', 'standard', '2023-03-05'),
  ('David Lee',       'david@example.com',   'APAC','premium', '2023-04-20'),
  ('Emma Davis',      'emma@example.com',    'EU', 'vip',      '2023-05-11'),
  ('Frank Brown',     'frank@example.com',   'US', 'standard', '2023-06-01'),
  ('Grace Kim',       'grace@example.com',   'APAC','vip',     '2023-07-07'),
  ('Henry Wilson',    'henry@example.com',   'EU', 'standard', '2023-08-22'),
  ('Iris Chen',       'iris@example.com',    'US', 'premium',  '2023-09-14'),
  ('Jack Taylor',     'jack@example.com',    'US', 'standard', '2023-10-30');

-- ---------------------------------------------------------------------------
-- PRODUCTS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    unit_price   REAL    NOT NULL,
    stock        INTEGER NOT NULL DEFAULT 0,
    active       INTEGER NOT NULL DEFAULT 1   -- 0 = discontinued
);

INSERT INTO products (name, category, unit_price, stock, active) VALUES
  ('Laptop Pro 15',       'electronics', 1299.99, 45,  1),
  ('Wireless Mouse',      'electronics',   29.99, 200, 1),
  ('USB-C Hub',           'electronics',   49.99, 150, 1),
  ('Standing Desk',       'furniture',    399.99,  20, 1),
  ('Ergonomic Chair',     'furniture',    599.99,  15, 1),
  ('Notebook Pack (3)',   'stationery',     9.99, 500, 1),
  ('Ballpoint Pens (10)', 'stationery',     4.99, 800, 1),
  ('Monitor 27"',         'electronics',  349.99,  60, 1),
  ('Keyboard Mechanical', 'electronics',   89.99, 120, 1),
  ('Webcam HD',           'electronics',   59.99,  90, 1),
  ('Old Fax Machine',     'electronics',  199.99,   0, 0);  -- discontinued

-- ---------------------------------------------------------------------------
-- ORDERS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending|shipped|delivered|cancelled
    created_at   TEXT    NOT NULL,
    shipped_at   TEXT,
    total_amount REAL    NOT NULL DEFAULT 0.0
);

INSERT INTO orders (customer_id, status, created_at, shipped_at, total_amount) VALUES
  (1, 'delivered', '2024-01-10', '2024-01-12', 1329.98),
  (1, 'shipped',   '2024-03-01', '2024-03-03', 49.99),
  (2, 'delivered', '2024-01-20', '2024-01-23', 629.98),
  (3, 'delivered', '2024-02-05', '2024-02-07', 14.98),
  (4, 'pending',   '2024-03-15', NULL,          349.99),
  (5, 'delivered', '2024-02-28', '2024-03-02', 1949.97),
  (6, 'cancelled', '2024-03-10', NULL,          89.99),
  (7, 'delivered', '2024-01-30', '2024-02-01', 659.98),
  (8, 'shipped',   '2024-03-20', '2024-03-22', 79.98),
  (9, 'delivered', '2024-02-14', '2024-02-16', 399.99),
  (10,'pending',   '2024-03-25', NULL,          29.99);

-- ---------------------------------------------------------------------------
-- ORDER ITEMS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL DEFAULT 1,
    unit_price  REAL    NOT NULL   -- price at time of order
);

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
  -- Order 1: Alice — Laptop + Mouse
  (1, 1, 1, 1299.99),
  (1, 2, 1,   29.99),
  -- Order 2: Alice — USB-C Hub
  (2, 3, 1,   49.99),
  -- Order 3: Bob — Chair + Pen
  (3, 5, 1,  599.99),
  (3, 7, 6,    4.99),
  -- Order 4: Carol — Notebooks + Pens
  (4, 6, 1,    9.99),
  (4, 7, 1,    4.99),
  -- Order 5: David — Monitor
  (5, 8, 1,  349.99),
  -- Order 6: Emma — Laptop + Chair + Mouse
  (6, 1, 1, 1299.99),
  (6, 5, 1,  599.99),
  (6, 2, 1,   29.99),  -- rounding: 1929.97 not 1949.97 — intentional for grader
  -- Order 7: Frank — Keyboard (cancelled)
  (7, 9, 1,   89.99),
  -- Order 8: Grace — Monitor + Chair
  (8, 5, 1,  599.99),
  (8, 2, 1,   29.99),  -- 2 items
  -- Order 9: Henry — Notebooks + Pens
  (9, 6, 2,   79.98),
  -- Order 10: Iris — Desk
  (10, 4, 1, 399.99),
  -- Order 11: Jack — Mouse
  (11, 2, 1,  29.99);

-- ---------------------------------------------------------------------------
-- PRODUCT REVIEWS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER NOT NULL REFERENCES products(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    body        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (DATE('now'))
);

INSERT INTO reviews (product_id, customer_id, rating, body, created_at) VALUES
  (1, 1, 5, 'Best laptop I have owned.', '2024-01-20'),
  (1, 5, 4, 'Great but runs hot.',       '2024-03-05'),
  (2, 1, 3, 'Works fine, nothing special.','2024-01-25'),
  (5, 2, 5, 'Super comfortable chair.',  '2024-02-01'),
  (8, 4, 4, 'Sharp display.',            '2024-03-20'),
  (9, 6, 2, 'Stopped working after a month.','2024-04-01'),
  (3, 9, 5, 'Exactly what I needed.',   '2024-02-20');

-- ---------------------------------------------------------------------------
-- Materialise order totals view for optimisation task baseline
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_order_totals AS
SELECT
    o.id                                         AS order_id,
    o.customer_id,
    c.name                                       AS customer_name,
    o.status,
    SUM(oi.quantity * oi.unit_price)             AS computed_total,
    COUNT(oi.id)                                 AS item_count
FROM orders o
JOIN customers c  ON c.id = o.customer_id
JOIN order_items oi ON oi.order_id = o.id
GROUP BY o.id, o.customer_id, c.name, o.status;
