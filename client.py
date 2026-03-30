"""
client.py — SQL Debug Environment Client

Implements a typed HTTP client for the SQL Debug Environment server.
Follows the OpenEnv EnvClient pattern with both sync and async support.

Usage (sync):
    env = SQLDebugEnv(base_url="http://localhost:7860").sync()
    with env:
        obs = env.reset(task_id="fix_syntax_error")
        obs = env.step(SQLAction(sql_query="SELECT ..."))

Usage (async):
    async with SQLDebugEnv(base_url="http://localhost:7860") as env:
        obs = await env.reset(task_id="fix_syntax_error")
        obs = await env.step(SQLAction(sql_query="SELECT ..."))

Usage (remote HF Space):
    env = SQLDebugEnv(base_url="https://your-user-sql-debug-env.hf.space").sync()
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Generator

import httpx

from models import SQLAction, SQLObservation, SQLState


class _SyncSQLDebugEnv:
    """Synchronous HTTP client for the SQL Debug Environment."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "_SyncSQLDebugEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def reset(self, task_id: str = "fix_syntax_error") -> SQLObservation:
        """Reset the environment and return the initial observation."""
        resp = self._client.post(
            f"{self._base}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return SQLObservation(**resp.json())

    def step(self, action: SQLAction) -> SQLObservation:
        """Submit a SQL query and get back the next observation."""
        resp = self._client.post(
            f"{self._base}/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        return SQLObservation(**resp.json())

    def state(self) -> SQLState:
        """Return the current episode state."""
        resp = self._client.get(f"{self._base}/state")
        resp.raise_for_status()
        data = resp.json()
        s = SQLState()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def health(self) -> dict:
        """Ping the health endpoint."""
        resp = self._client.get(f"{self._base}/health")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list[dict]:
        """List available tasks."""
        resp = self._client.get(f"{self._base}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


class SQLDebugEnv:
    """
    SQL Debug Environment client.

    Call .sync() to get a synchronous wrapper.
    Use as async context manager for async usage.

    Example:
        # Sync
        with SQLDebugEnv(base_url="http://localhost:7860").sync() as env:
            obs = env.reset("fix_syntax_error")

        # Async
        async with SQLDebugEnv(base_url="http://localhost:7860") as env:
            obs = await env.reset("fix_syntax_error")
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url
        self._timeout = timeout

    def sync(self) -> _SyncSQLDebugEnv:
        """Return a synchronous wrapper client."""
        return _SyncSQLDebugEnv(self._base_url, self._timeout)

    # ── Async interface ──────────────────────────────────────────────────────

    async def __aenter__(self) -> "SQLDebugEnv":
        self._async_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._async_client.aclose()

    async def reset(self, task_id: str = "fix_syntax_error") -> SQLObservation:
        resp = await self._async_client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return SQLObservation(**resp.json())

    async def step(self, action: SQLAction) -> SQLObservation:
        resp = await self._async_client.post(
            "/step", json={"action": action.model_dump()}
        )
        resp.raise_for_status()
        return SQLObservation(**resp.json())

    async def state(self) -> dict:
        resp = await self._async_client.get("/state")
        resp.raise_for_status()
        return resp.json()
