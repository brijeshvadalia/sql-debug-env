"""
server/app.py — FastAPI HTTP Server for SQL Debug Environment

Endpoints:
  GET  /          → Beautiful homepage (HTML)
  GET  /health    → Liveness probe
  GET  /info      → Environment metadata
  GET  /tasks     → All 8 tasks with metadata
  POST /reset     → Start new episode
  POST /step      → Submit SQL action
  GET  /state     → Current episode state
  GET  /history   → Full step history of current episode
  POST /evaluate  → Run correct answer for a task and return score
  GET  /stats     → Aggregate stats across all episodes
  GET  /docs      → Swagger UI
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from models import SQLAction, SQLObservation
from server.environment import SQLDebugEnvironment
from server.tasks import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_env: Optional[SQLDebugEnvironment] = None
_episode_history: list[dict[str, Any]] = []
_global_stats: dict[str, Any] = {
    "total_episodes": 0,
    "total_steps": 0,
    "scores_by_task": {tid: [] for tid in TASKS},
}


def get_env() -> SQLDebugEnvironment:
    global _env
    if _env is None:
        _env = SQLDebugEnvironment()
    return _env


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "fix_syntax_error"

class StepRequest(BaseModel):
    action: SQLAction

class EvaluateRequest(BaseModel):
    task_id: str
    sql_query: str
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    env = get_env()
    env.reset("fix_syntax_error")
    logger.info("Environment warmed up and ready.")
    yield
    env.close()
    logger.info("Environment shut down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="SQL Debug Environment",
        description=(
            "OpenEnv-compatible environment for SQL debugging and optimisation. "
            "8 tasks across easy/medium/hard/expert difficulties. "
            "Built for the Scaler x Meta PyTorch OpenEnv Hackathon 2026."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Homepage ─────────────────────────────────────────────────────────────

    @app.get("/", tags=["meta"], response_class=HTMLResponse)
    async def homepage():
        stats = _global_stats
        task_rows = ""
        difficulty_colors = {
            "easy": "#22c55e", "medium": "#f59e0b",
            "hard": "#ef4444", "expert": "#8b5cf6"
        }
        for t in TASKS.values():
            scores = stats["scores_by_task"].get(t.task_id, [])
            avg = f"{sum(scores)/len(scores):.3f}" if scores else "—"
            color = difficulty_colors.get(t.difficulty, "#6b7280")
            task_rows += f"""
            <tr>
              <td><code>{t.task_id}</code></td>
              <td style="color:{color};font-weight:bold">{t.difficulty.upper()}</td>
              <td>{t.max_steps}</td>
              <td>{t.reward_threshold}</td>
              <td>{avg}</td>
              <td>{len(scores)}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SQL Debug Environment</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
    .hero{{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:60px 40px;text-align:center;border-bottom:1px solid #1e40af}}
    .hero h1{{font-size:2.8rem;color:#60a5fa;margin-bottom:12px}}
    .hero p{{font-size:1.15rem;color:#94a3b8;max-width:700px;margin:0 auto 24px}}
    .badges{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:10px}}
    .badge{{padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600}}
    .badge.green{{background:#166534;color:#86efac}}
    .badge.blue{{background:#1e3a8a;color:#93c5fd}}
    .badge.purple{{background:#4c1d95;color:#c4b5fd}}
    .container{{max-width:1100px;margin:0 auto;padding:40px 20px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:40px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px}}
    .card h3{{color:#60a5fa;margin-bottom:12px;font-size:1.1rem}}
    .stat{{font-size:2rem;font-weight:bold;color:#f1f5f9}}
    .stat-label{{color:#64748b;font-size:0.85rem;margin-top:4px}}
    .endpoint{{display:block;background:#0f172a;border:1px solid #334155;border-radius:6px;
               padding:8px 14px;margin:6px 0;font-family:monospace;font-size:13px;color:#a5f3fc;
               text-decoration:none}}
    .endpoint:hover{{border-color:#60a5fa;color:#60a5fa}}
    .method{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;
             font-weight:bold;margin-right:8px}}
    .get{{background:#166534;color:#86efac}}
    .post{{background:#1e3a8a;color:#93c5fd}}
    table{{width:100%;border-collapse:collapse;margin-top:10px}}
    th{{background:#1e3a8a;color:#93c5fd;padding:10px;text-align:left;font-size:13px}}
    td{{padding:10px;border-bottom:1px solid #334155;font-size:13px}}
    tr:hover td{{background:#1e293b}}
    code{{background:#0f172a;padding:2px 6px;border-radius:4px;font-size:12px;color:#a5f3fc}}
    .footer{{text-align:center;padding:30px;color:#475569;font-size:13px;border-top:1px solid #1e293b}}
    .live{{display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;
           margin-right:6px;animation:pulse 2s infinite}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
  </style>
</head>
<body>
  <div class="hero">
    <h1>🛠️ SQL Debug Environment</h1>
    <p>A real-world OpenEnv environment where AI agents learn to debug, fix, and optimize SQL queries
       against a realistic e-commerce database.</p>
    <div class="badges">
      <span class="badge green"><span class="live"></span>Running v2.0.0</span>
      <span class="badge blue">OpenEnv Compatible</span>
      <span class="badge purple">8 Tasks · 4 Difficulty Levels</span>
      <span class="badge blue">Scaler × Meta PyTorch Hackathon 2026</span>
    </div>
  </div>

  <div class="container">
    <div class="grid">
      <div class="card">
        <h3>📊 Live Stats</h3>
        <div class="stat">{stats["total_episodes"]}</div>
        <div class="stat-label">Total Episodes</div>
        <div class="stat" style="margin-top:16px">{stats["total_steps"]}</div>
        <div class="stat-label">Total Steps Taken</div>
      </div>
      <div class="card">
        <h3>🎮 Tasks Overview</h3>
        <div class="stat">8</div>
        <div class="stat-label">Total Tasks</div>
        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <span class="badge" style="background:#166534;color:#86efac">2 Easy</span>
          <span class="badge" style="background:#92400e;color:#fde68a">3 Medium</span>
          <span class="badge" style="background:#7f1d1d;color:#fca5a5">2 Hard</span>
          <span class="badge" style="background:#4c1d95;color:#c4b5fd">1 Expert</span>
        </div>
      </div>
      <div class="card">
        <h3>⚡ Quick Start</h3>
        <p style="color:#94a3b8;font-size:13px;line-height:1.8">
          1. <code>POST /reset</code> with a task_id<br>
          2. <code>POST /step</code> with your SQL fix<br>
          3. Read reward (0.0–1.0) and iterate<br>
          4. <code>done=true</code> ends the episode
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:24px">
      <h3>🔗 API Endpoints</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px">
        <a class="endpoint" href="/health"><span class="method get">GET</span>/health — Liveness probe</a>
        <a class="endpoint" href="/tasks"><span class="method get">GET</span>/tasks — List all tasks</a>
        <a class="endpoint" href="/state"><span class="method get">GET</span>/state — Episode state</a>
        <a class="endpoint" href="/history"><span class="method get">GET</span>/history — Step history</a>
        <a class="endpoint" href="/stats"><span class="method get">GET</span>/stats — Aggregate stats</a>
        <a class="endpoint" href="/info"><span class="method get">GET</span>/info — Environment info</a>
        <a class="endpoint" href="/docs"><span class="method post">POST</span>/reset — Start episode</a>
        <a class="endpoint" href="/docs"><span class="method post">POST</span>/step — Submit SQL query</a>
        <a class="endpoint" href="/docs"><span class="method post">POST</span>/evaluate — Score any query</a>
        <a class="endpoint" href="/docs"><span class="method get">GET</span>/docs — Swagger UI</a>
      </div>
    </div>

    <div class="card">
      <h3>📋 All Tasks</h3>
      <table>
        <tr>
          <th>Task ID</th><th>Difficulty</th><th>Max Steps</th>
          <th>Threshold</th><th>Avg Score</th><th>Episodes</th>
        </tr>
        {task_rows}
      </table>
    </div>
  </div>

  <div class="footer">
    Built for <strong>Scaler × Meta PyTorch OpenEnv Hackathon 2026</strong> ·
    <a href="/docs" style="color:#60a5fa">API Docs</a> ·
    <a href="/tasks" style="color:#60a5fa">Tasks JSON</a> ·
    <a href="/stats" style="color:#60a5fa">Stats</a>
  </div>
</body>
</html>"""
        return HTMLResponse(content=html)

    # ── Meta endpoints ────────────────────────────────────────────────────────

    @app.get("/health", tags=["meta"])
    async def health():
        return {"status": "ok", "environment": "sql-debug-env", "version": "2.0.0"}

    @app.get("/info", tags=["meta"])
    async def info():
        return get_env().get_metadata()

    @app.get("/tasks", tags=["meta"])
    async def list_tasks():
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "tags": t.tags,
                "max_steps": t.max_steps,
                "reward_threshold": t.reward_threshold,
                "description": t.description[:300] + "...",
            }
            for t in TASKS.values()
        ]

    @app.get("/stats", tags=["meta"])
    async def stats():
        s = _global_stats
        by_task = {}
        for tid, scores in s["scores_by_task"].items():
            by_task[tid] = {
                "episodes": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "best_score": round(max(scores), 4) if scores else 0.0,
            }
        return {
            "total_episodes": s["total_episodes"],
            "total_steps": s["total_steps"],
            "by_task": by_task,
        }

    # ── OpenEnv core endpoints ────────────────────────────────────────────────

    @app.post("/reset", response_model=SQLObservation, tags=["openenv"])
    async def reset(body: ResetRequest):
        global _episode_history
        try:
            obs = get_env().reset(task_id=body.task_id)
            _episode_history = []
            _global_stats["total_episodes"] += 1
            return obs
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("reset() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/step", response_model=SQLObservation, tags=["openenv"])
    async def step(body: StepRequest):
        try:
            obs = get_env().step(action=body.action)
            _global_stats["total_steps"] += 1
            _episode_history.append({
                "step": obs.step_count,
                "sql_query": body.action.sql_query[:200],
                "reasoning": body.action.reasoning,
                "reward": obs.reward,
                "done": obs.done,
                "error": obs.error_message,
            })
            if obs.done:
                tid = obs.task_id
                _global_stats["scores_by_task"][tid].append(obs.reward)
            return obs
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("step() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/state", tags=["openenv"])
    async def state():
        return get_env().state.to_dict()

    @app.get("/history", tags=["openenv"])
    async def history():
        """Return full step-by-step history of the current episode."""
        return {
            "episode_id": get_env().state.episode_id,
            "task_id": get_env().state.task_id,
            "steps": _episode_history,
            "total_steps": len(_episode_history),
        }

    @app.post("/evaluate", tags=["openenv"])
    async def evaluate(body: EvaluateRequest):
        """
        Score any SQL query against a task without affecting the current episode.
        Useful for automated evaluation pipelines.
        """
        if body.task_id not in TASKS:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {body.task_id}")
        try:
            # Save current state
            current_task_id = get_env().state.task_id or "fix_syntax_error"
            # Run evaluation
            get_env().reset(task_id=body.task_id)
            action = SQLAction(sql_query=body.sql_query, reasoning=body.reasoning)
            obs = get_env().step(action=action)
            # Restore previous episode
            get_env().reset(task_id=current_task_id)
            return {
                "task_id": body.task_id,
                "sql_query": body.sql_query,
                "reward": obs.reward,
                "reward_breakdown": obs.reward_breakdown.model_dump() if obs.reward_breakdown else None,
                "error_message": obs.error_message,
                "query_result": obs.query_result,
                "execution_time_ms": obs.execution_time_ms,
            }
        except Exception as exc:
            logger.exception("evaluate() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Optional Gradio web UI ────────────────────────────────────────────────

    if os.getenv("ENABLE_WEB_INTERFACE", "").lower() in ("1", "true", "yes"):
        try:
            import gradio as gr
            import json as _json

            def gradio_reset(task_id):
                obs = get_env().reset(task_id=task_id)
                return obs.model_dump_json(indent=2)

            def gradio_step(sql_query, reasoning):
                action = SQLAction(sql_query=sql_query, reasoning=reasoning or None)
                obs = get_env().step(action=action)
                return obs.model_dump_json(indent=2)

            def gradio_state():
                return _json.dumps(get_env().state.to_dict(), indent=2)

            with gr.Blocks(title="SQL Debug Environment") as demo:
                gr.Markdown("## 🛠️ SQL Debug Environment\nOpenEnv real-world environment.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Reset")
                        task_dd = gr.Dropdown(choices=list(TASKS.keys()),
                                              value="fix_syntax_error", label="Task")
                        reset_btn = gr.Button("Reset Episode")
                        reset_out = gr.Code(language="json", label="Observation")
                        reset_btn.click(gradio_reset, inputs=[task_dd], outputs=[reset_out])
                    with gr.Column():
                        gr.Markdown("### Step")
                        sql_in = gr.Code(language="sql", label="SQL Query")
                        reasoning_in = gr.Textbox(label="Reasoning (optional)")
                        step_btn = gr.Button("Submit")
                        step_out = gr.Code(language="json", label="Result + Reward")
                        step_btn.click(gradio_step, inputs=[sql_in, reasoning_in], outputs=[step_out])
                with gr.Row():
                    state_btn = gr.Button("Get State")
                    state_out = gr.Code(language="json", label="State")
                    state_btn.click(gradio_state, inputs=[], outputs=[state_out])

            app = gr.mount_gradio_app(app, demo, path="/web")
            logger.info("Gradio UI mounted at /web")
        except ImportError:
            logger.warning("gradio not installed — web UI disabled.")

    return app


app = create_app()
