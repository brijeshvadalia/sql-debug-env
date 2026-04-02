"""
server/app.py — Advanced FastAPI Server for SQL Debug Environment v3.0

All endpoints:
  GET  /              Beautiful dark homepage with live stats
  GET  /tester        Visual interactive tester UI
  GET  /health        Liveness probe
  GET  /info          Environment metadata
  GET  /tasks         All 8 tasks with metadata
  GET  /stats         Aggregate stats across all episodes
  POST /reset         Start new episode
  POST /step          Submit SQL action
  GET  /state         Current episode state
  GET  /history       Full step history of current episode
  POST /evaluate      Score any SQL without affecting current episode
  POST /hint          Get next hint (with reward penalty)
  GET  /curriculum    Curriculum progress and next recommended task
  GET  /leaderboard   Model comparison leaderboard
  GET  /docs          Swagger UI
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import SQLAction, SQLObservation, SQLState
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
    "total_hints_used": 0,
    "scores_by_task": {tid: [] for tid in TASKS},
    "leaderboard": [],
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

    class Config:
        json_schema_extra = {"example": {"task_id": "fix_syntax_error"}}


class StepRequest(BaseModel):
    action: SQLAction

    class Config:
        json_schema_extra = {
            "example": {
                "action": {
                    "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
                    "reasoning": "Fixed SELEC->SELECT, FORM->FROM, ORDR->ORDER BY",
                }
            }
        }


class EvaluateRequest(BaseModel):
    task_id: str
    sql_query: str
    reasoning: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "fix_syntax_error",
                "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
            }
        }


class LeaderboardEntry(BaseModel):
    model_name: str
    mean_score: float
    episodes: int
    best_task: Optional[str] = None
    submitted_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    env = get_env()
    env.reset("fix_syntax_error")
    logger.info("Environment warmed up — ready.")
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
            "Advanced OpenEnv-compatible environment for SQL debugging and optimisation. "
            "8 tasks | 4 difficulty levels | Multi-turn memory | Hint system | "
            "EXPLAIN analysis | Curriculum learning | Visual tester"
        ),
        version="3.0.0",
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

    # =========================================================================
    # HOMEPAGE
    # =========================================================================

    @app.get("/", tags=["meta"], response_class=HTMLResponse)
    async def homepage():
        s = _global_stats
        diff_colors = {
            "easy": ("#166534","#86efac"),
            "medium": ("#92400e","#fde68a"),
            "hard": ("#7f1d1d","#fca5a5"),
            "expert": ("#4c1d95","#c4b5fd"),
        }
        task_rows = ""
        for t in TASKS.values():
            scores = s["scores_by_task"].get(t.task_id, [])
            avg = f"{sum(scores)/len(scores):.3f}" if scores else "—"
            bg, fg = diff_colors.get(t.difficulty, ("#1e293b","#94a3b8"))
            task_rows += f"""
            <tr>
              <td><code style="color:#a5f3fc">{t.task_id}</code></td>
              <td><span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{t.difficulty.upper()}</span></td>
              <td style="color:#94a3b8">{t.max_steps}</td>
              <td style="color:#94a3b8">{t.reward_threshold}</td>
              <td style="color:#22c55e;font-weight:600">{avg}</td>
              <td style="color:#64748b">{len(scores)}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SQL Debug Environment v3.0</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
    a{{color:#60a5fa;text-decoration:none}}a:hover{{text-decoration:underline}}
    .hero{{background:linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);padding:56px 40px;text-align:center;border-bottom:1px solid #1e40af}}
    h1{{font-size:2.6rem;color:#60a5fa;margin-bottom:12px;font-weight:700}}
    .sub{{font-size:1.05rem;color:#94a3b8;max-width:680px;margin:0 auto 24px;line-height:1.7}}
    .badges{{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:8px}}
    .badge{{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600}}
    .bg{{background:#166534;color:#86efac}}.bb{{background:#1e3a8a;color:#93c5fd}}.bp{{background:#4c1d95;color:#c4b5fd}}.bo{{background:#7c2d12;color:#fed7aa}}
    .container{{max-width:1100px;margin:0 auto;padding:36px 20px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin-bottom:32px}}
    .card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px}}
    .card h3{{color:#60a5fa;margin-bottom:10px;font-size:1rem;font-weight:600}}
    .stat{{font-size:2rem;font-weight:700;color:#f1f5f9}}
    .stat-lbl{{color:#64748b;font-size:12px;margin-top:4px}}
    .ep{{display:block;background:#0f172a;border:1px solid #334155;border-radius:6px;padding:7px 12px;margin:5px 0;font-family:monospace;font-size:12px;color:#a5f3fc;text-decoration:none}}
    .ep:hover{{border-color:#3b82f6;color:#60a5fa}}
    .method{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;margin-right:6px}}
    .get{{background:#166534;color:#86efac}}.post{{background:#1e3a8a;color:#93c5fd}}
    table{{width:100%;border-collapse:collapse}}
    th{{background:#0f172a;color:#64748b;padding:9px 10px;text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
    td{{padding:9px 10px;border-bottom:1px solid #1e293b;font-size:13px}}
    tr:hover td{{background:#273548}}
    .feat{{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid #1e293b;font-size:13px}}
    .feat:last-child{{border-bottom:none}}
    .feat-icon{{color:#22c55e;font-size:16px;margin-top:1px;flex-shrink:0}}
    .feat-title{{color:#f1f5f9;font-weight:500}}
    .feat-desc{{color:#64748b;font-size:12px;margin-top:2px}}
    .footer{{text-align:center;padding:28px;color:#475569;font-size:12px;border-top:1px solid #1e293b;margin-top:20px}}
    .live{{display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
  </style>
</head>
<body>
<div class="hero">
  <h1>SQL Debug Environment</h1>
  <p class="sub">A real-world OpenEnv environment where AI agents learn to debug, fix, and optimize SQL queries against a realistic e-commerce database.</p>
  <div class="badges">
    <span class="badge bg"><span class="live"></span>Running v3.0.0</span>
    <span class="badge bb">OpenEnv Compatible</span>
    <span class="badge bp">8 Tasks · 4 Levels</span>
    <span class="badge bo">Multi-turn Memory</span>
    <span class="badge bp">Hint System</span>
    <span class="badge bb">Curriculum Learning</span>
  </div>
</div>

<div class="container">
  <div class="grid">
    <div class="card">
      <h3>Live Stats</h3>
      <div class="stat">{s["total_episodes"]}</div><div class="stat-lbl">Total Episodes</div>
      <div class="stat" style="margin-top:14px">{s["total_steps"]}</div><div class="stat-lbl">Total Steps</div>
      <div class="stat" style="margin-top:14px">{s["total_hints_used"]}</div><div class="stat-lbl">Hints Used</div>
    </div>
    <div class="card">
      <h3>Advanced Features</h3>
      <div class="feat"><div class="feat-icon">✓</div><div><div class="feat-title">Multi-turn Memory</div><div class="feat-desc">Agent sees full trajectory history in every observation</div></div></div>
      <div class="feat"><div class="feat-icon">✓</div><div><div class="feat-title">3-Level Hint System</div><div class="feat-desc">Progressive hints with 10% reward penalty each</div></div></div>
      <div class="feat"><div class="feat-icon">✓</div><div><div class="feat-title">EXPLAIN Analysis</div><div class="feat-desc">Query plan scan count and index usage in every response</div></div></div>
      <div class="feat"><div class="feat-icon">✓</div><div><div class="feat-title">Curriculum Learning</div><div class="feat-desc">Auto-advance easy→medium→hard→expert on mastery</div></div></div>
    </div>
    <div class="card">
      <h3>Quick Links</h3>
      <a class="ep" href="/tester"><span class="method get">UI</span>/tester — Visual tester</a>
      <a class="ep" href="/docs"><span class="method get">GET</span>/docs — Swagger UI</a>
      <a class="ep" href="/tasks"><span class="method get">GET</span>/tasks — All tasks</a>
      <a class="ep" href="/stats"><span class="method get">GET</span>/stats — Live stats</a>
      <a class="ep" href="/leaderboard"><span class="method get">GET</span>/leaderboard — Leaderboard</a>
      <a class="ep" href="/curriculum"><span class="method get">GET</span>/curriculum — Progress</a>
      <a class="ep" href="/health"><span class="method get">GET</span>/health — Health check</a>
    </div>
  </div>

  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:20px">
    <div class="card" style="text-align:center"><div class="stat" style="font-size:1.6rem;color:#22c55e">2</div><div class="stat-lbl">Easy tasks</div></div>
    <div class="card" style="text-align:center"><div class="stat" style="font-size:1.6rem;color:#f59e0b">3</div><div class="stat-lbl">Medium tasks</div></div>
    <div class="card" style="text-align:center"><div class="stat" style="font-size:1.6rem;color:#ef4444">2</div><div class="stat-lbl">Hard tasks</div></div>
    <div class="card" style="text-align:center"><div class="stat" style="font-size:1.6rem;color:#a78bfa">1</div><div class="stat-lbl">Expert tasks</div></div>
  </div>

  <div class="card">
    <h3 style="margin-bottom:14px">All 8 Tasks</h3>
    <table>
      <tr><th>Task ID</th><th>Difficulty</th><th>Max Steps</th><th>Threshold</th><th>Avg Score</th><th>Episodes</th></tr>
      {task_rows}
    </table>
  </div>
</div>

<div class="footer">
  Built for <strong>Scaler × Meta PyTorch OpenEnv Hackathon 2026</strong> &nbsp;·&nbsp;
  <a href="/tester">Visual Tester</a> &nbsp;·&nbsp;
  <a href="/docs">API Docs</a> &nbsp;·&nbsp;
  <a href="/leaderboard">Leaderboard</a>
</div>
</body>
</html>"""
        return HTMLResponse(content=html)

    # =========================================================================
    # VISUAL TESTER
    # =========================================================================

    @app.get("/tester", tags=["meta"], response_class=HTMLResponse)
    async def tester():
        """Visual interactive tester — try all 8 tasks with real-time rewards."""
        tester_path = Path(__file__).parent.parent / "static" / "tester.html"
        if tester_path.exists():
            return HTMLResponse(content=tester_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1 style='font-family:sans-serif;padding:40px'>Tester file not found. Upload static/tester.html</h1>", status_code=404)

    # =========================================================================
    # META ENDPOINTS
    # =========================================================================

    @app.get("/health", tags=["meta"])
    async def health():
        """Liveness probe — must return 200 for OpenEnv validator."""
        return {"status": "ok", "environment": "sql-debug-env", "version": "3.0.0"}

    @app.get("/info", tags=["meta"])
    async def info():
        """Environment metadata including features and task list."""
        return get_env().get_metadata()

    @app.get("/tasks", tags=["meta"])
    async def list_tasks():
        """List all 8 tasks with difficulty, description, and tags."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "tags": t.tags,
                "max_steps": t.max_steps,
                "reward_threshold": t.reward_threshold,
                "description_short": t.description.split("\n")[0][:200],
            }
            for t in TASKS.values()
        ]

    @app.get("/stats", tags=["meta"])
    async def stats():
        """Aggregate performance stats across all episodes."""
        s = _global_stats
        by_task = {}
        for tid, scores in s["scores_by_task"].items():
            by_task[tid] = {
                "episodes": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "best_score": round(max(scores), 4) if scores else 0.0,
                "worst_score": round(min(scores), 4) if scores else 0.0,
            }
        return {
            "total_episodes": s["total_episodes"],
            "total_steps": s["total_steps"],
            "total_hints_used": s["total_hints_used"],
            "overall_avg": round(
                sum(v["avg_score"] for v in by_task.values() if v["episodes"] > 0)
                / max(1, sum(1 for v in by_task.values() if v["episodes"] > 0)),
                4,
            ),
            "by_task": by_task,
        }

    @app.get("/leaderboard", tags=["meta"])
    async def leaderboard():
        """
        Model comparison leaderboard.
        Shows how different models scored across all tasks.
        Submit scores via POST /leaderboard/submit.
        """
        entries = _global_stats.get("leaderboard", [])
        # Also add current session stats
        s = _global_stats
        task_scores = s["scores_by_task"]
        all_scores = [sc for scores in task_scores.values() for sc in scores]
        session_avg = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

        current_session = {
            "model_name": "current_session",
            "mean_score": session_avg,
            "episodes": s["total_episodes"],
            "steps": s["total_steps"],
            "by_task": {
                tid: round(sum(sc)/len(sc), 3) if sc else 0.0
                for tid, sc in task_scores.items()
            },
        }
        return {
            "leaderboard": entries,
            "current_session": current_session,
            "note": "Submit your model scores via POST /leaderboard/submit",
        }

    @app.post("/leaderboard/submit", tags=["meta"])
    async def submit_leaderboard(entry: LeaderboardEntry):
        """Submit a model's scores to the leaderboard."""
        data = entry.model_dump()
        data["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _global_stats["leaderboard"].append(data)
        # Keep top 20 by mean_score
        _global_stats["leaderboard"].sort(key=lambda x: x["mean_score"], reverse=True)
        _global_stats["leaderboard"] = _global_stats["leaderboard"][:20]
        return {"status": "ok", "rank": _global_stats["leaderboard"].index(data) + 1}

    # =========================================================================
    # OPENENV CORE ENDPOINTS
    # =========================================================================

    @app.post("/reset", response_model=SQLObservation, tags=["openenv"])
    async def reset(body: ResetRequest):
        """
        Reset the environment and start a new episode.

        Args:
            task_id: One of the 8 task IDs. Defaults to fix_syntax_error.

        Returns:
            Initial SQLObservation with broken query and schema hint.
        """
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
        """
        Execute one agent action (a SQL query).

        Returns observation with:
          - reward (0.0–1.0)
          - reward_breakdown (correctness, efficiency, step_penalty)
          - conversation_history (last 5 steps)
          - query_analysis (EXPLAIN plan summary)
          - hint_available, hints_used
        """
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
                "exec_ms": obs.execution_time_ms,
            })
            if obs.done:
                _global_stats["scores_by_task"][obs.task_id].append(obs.reward)
            return obs
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("step() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/state", tags=["openenv"])
    async def state():
        """Return current episode state (episode_id, step_count, task_id, etc.)."""
        return get_env().state.to_dict()

    @app.get("/history", tags=["openenv"])
    async def history():
        """
        Return full step-by-step history of the current episode.
        Includes SQL queries, errors, rewards, and timing per step.
        """
        return get_env().get_history()

    @app.post("/evaluate", tags=["openenv"])
    async def evaluate(body: EvaluateRequest):
        """
        Score any SQL query against a task without affecting the current episode.
        Useful for automated evaluation pipelines and batch scoring.
        """
        if body.task_id not in TASKS:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {body.task_id}")
        try:
            current_task_id = get_env().state.task_id or "fix_syntax_error"
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
                "query_analysis": obs.query_analysis.model_dump() if obs.query_analysis else None,
            }
        except Exception as exc:
            logger.exception("evaluate() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # =========================================================================
    # ADVANCED ENDPOINTS
    # =========================================================================

    @app.post("/hint", tags=["advanced"])
    async def get_hint():
        """
        Get the next progressive hint for the current task.

        Hints cost a 10% reward penalty each (max 3 hints = 30% penalty).
        Level 1: general direction
        Level 2: specific bug location
        Level 3: exact fix
        """
        try:
            return get_env().get_hint()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/curriculum", tags=["advanced"])
    async def curriculum():
        """
        Get curriculum learning status and next recommended task.

        The curriculum auto-advances when the agent achieves avg score >= 0.8
        over 3 consecutive episodes. Order: easy -> medium -> hard -> expert.
        """
        return get_env().curriculum_status()

    @app.post("/curriculum/next", tags=["advanced"])
    async def curriculum_next():
        """
        Check if agent should advance to next difficulty level.
        Returns the recommended next task_id.
        """
        return get_env().curriculum_next()

    @app.post("/evaluate/batch", tags=["advanced"])
    async def evaluate_batch(body: dict):
        """
        Evaluate a SQL query against ALL 8 tasks at once.
        Body: {"sql_query": "...", "reasoning": "..."}
        Returns scores for every task.
        """
        sql = body.get("sql_query", "")
        reasoning = body.get("reasoning")
        if not sql:
            raise HTTPException(status_code=400, detail="sql_query is required")

        results = {}
        current_task_id = get_env().state.task_id or "fix_syntax_error"

        for task_id in TASKS:
            try:
                get_env().reset(task_id=task_id)
                obs = get_env().step(SQLAction(sql_query=sql, reasoning=reasoning))
                results[task_id] = {
                    "reward": obs.reward,
                    "correctness": obs.reward_breakdown.correctness if obs.reward_breakdown else 0.0,
                    "error": obs.error_message,
                    "rows": len(obs.query_result) if obs.query_result else 0,
                }
            except Exception as exc:
                results[task_id] = {"error": str(exc), "reward": 0.0}

        get_env().reset(task_id=current_task_id)
        scores = [v.get("reward", 0.0) for v in results.values()]
        return {
            "sql_query": sql,
            "results_by_task": results,
            "mean_score": round(sum(scores) / len(scores), 4),
            "tasks_solved": sum(1 for v in results.values() if v.get("reward", 0) >= 0.8),
        }

    # =========================================================================
    # OPTIONAL GRADIO WEB UI
    # =========================================================================

    if os.getenv("ENABLE_WEB_INTERFACE", "").lower() in ("1", "true", "yes"):
        try:
            import gradio as gr

            def gradio_reset(task_id: str) -> str:
                obs = get_env().reset(task_id=task_id)
                return obs.model_dump_json(indent=2)

            def gradio_step(sql_query: str, reasoning: str) -> str:
                obs = get_env().step(SQLAction(sql_query=sql_query, reasoning=reasoning or None))
                return obs.model_dump_json(indent=2)

            def gradio_hint() -> str:
                import json as _json
                return _json.dumps(get_env().get_hint(), indent=2)

            def gradio_curriculum() -> str:
                import json as _json
                return _json.dumps(get_env().curriculum_status(), indent=2)

            with gr.Blocks(title="SQL Debug Environment v3") as demo:
                gr.Markdown("## SQL Debug Environment v3.0\nAdvanced OpenEnv environment with 8 tasks.")
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
                        reason_in = gr.Textbox(label="Reasoning (optional)")
                        step_btn = gr.Button("Submit Query")
                        step_out = gr.Code(language="json", label="Result + Reward")
                        step_btn.click(gradio_step, inputs=[sql_in, reason_in], outputs=[step_out])
                with gr.Row():
                    hint_btn = gr.Button("Get Hint (costs 10% reward)")
                    hint_out = gr.Code(language="json", label="Hint")
                    hint_btn.click(gradio_hint, inputs=[], outputs=[hint_out])
                    curr_btn = gr.Button("Curriculum Status")
                    curr_out = gr.Code(language="json", label="Progress")
                    curr_btn.click(gradio_curriculum, inputs=[], outputs=[curr_out])

            app = gr.mount_gradio_app(app, demo, path="/web")
            logger.info("Gradio UI mounted at /web")
        except ImportError:
            logger.warning("gradio not installed — /web disabled.")

    # =========================================================================
    # STATIC FILES
    # =========================================================================

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()
