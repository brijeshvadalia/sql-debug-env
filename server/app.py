"""
server/app.py — FastAPI HTTP Server for SQL Debug Environment

Exposes the OpenEnv-required endpoints:
  POST /reset   → SQLObservation
  POST /step    → SQLObservation
  GET  /state   → SQLState dict
  GET  /health  → {"status": "ok"}
  GET  /info    → environment metadata
  GET  /tasks   → list of available tasks

Also mounts a /web Gradio UI when ENABLE_WEB_INTERFACE=true.

Runs on port 7860 (HF Spaces default).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
# Global environment instance (singleton per container)
# ---------------------------------------------------------------------------
_env: SQLDebugEnvironment | None = None


def get_env() -> SQLDebugEnvironment:
    global _env
    if _env is None:
        _env = SQLDebugEnvironment()
    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "fix_syntax_error"

    class Config:
        json_schema_extra = {
            "example": {"task_id": "fix_syntax_error"}
        }


class StepRequest(BaseModel):
    action: SQLAction

    class Config:
        json_schema_extra = {
            "example": {
                "action": {
                    "sql_query": "SELECT id, name, email FROM customers WHERE tier = 'vip' ORDER BY name;",
                    "reasoning": "Fixed 'SELEC' → 'SELECT', 'FORM' → 'FROM', 'ORDR' → 'ORDER'.",
                }
            }
        }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up environment at startup
    env = get_env()
    env.reset("fix_syntax_error")
    logger.info("Environment warmed up and ready.")
    yield
    env.close()
    logger.info("Environment shut down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="SQL Debug Environment",
        description=(
            "OpenEnv-compatible environment for SQL debugging and optimisation. "
            "Agents fix broken SQL queries and rewrite slow queries against a "
            "realistic e-commerce SQLite database."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — needed for HF Spaces iframe embedding
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──────────────────────────────────────────────────────────────

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        """Liveness probe. Must return 200 for HF Spaces and OpenEnv validator."""
        return {"status": "ok", "environment": "sql-debug-env", "version": "1.0.0"}

    @app.get("/info", tags=["meta"])
    async def info() -> dict:
        """Environment metadata."""
        return get_env().get_metadata()

    @app.get("/tasks", tags=["meta"])
    async def list_tasks() -> list[dict]:
        """List all available tasks with difficulty metadata."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "reward_threshold": t.reward_threshold,
                "description": t.description[:200] + "...",
            }
            for t in TASKS.values()
        ]

    # ── Core OpenEnv endpoints ──────────────────────────────────────────────

    @app.post("/reset", response_model=SQLObservation, tags=["openenv"])
    async def reset(body: ResetRequest) -> SQLObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: One of fix_syntax_error | fix_logic_error | optimize_query
        """
        try:
            obs = get_env().reset(task_id=body.task_id)
            return obs
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("reset() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/step", response_model=SQLObservation, tags=["openenv"])
    async def step(body: StepRequest) -> SQLObservation:
        """
        Execute one agent action (a SQL query).

        The agent submits a SQL query which is executed against the environment
        database. Returns observation with reward and execution feedback.
        """
        try:
            obs = get_env().step(action=body.action)
            return obs
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("step() failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/state", tags=["openenv"])
    async def state() -> dict:
        """Return current episode state metadata."""
        return get_env().state.to_dict()

    # ── Optional Gradio web UI ──────────────────────────────────────────────

    if os.getenv("ENABLE_WEB_INTERFACE", "").lower() in ("1", "true", "yes"):
        try:
            import gradio as gr

            def gradio_reset(task_id: str) -> str:
                obs = get_env().reset(task_id=task_id)
                return obs.model_dump_json(indent=2)

            def gradio_step(sql_query: str, reasoning: str) -> str:
                action = SQLAction(sql_query=sql_query, reasoning=reasoning or None)
                obs = get_env().step(action=action)
                return obs.model_dump_json(indent=2)

            def gradio_state() -> str:
                import json
                return json.dumps(get_env().state.to_dict(), indent=2)

            with gr.Blocks(title="SQL Debug Environment") as demo:
                gr.Markdown(
                    "## 🛠️ SQL Debug Environment\n"
                    "An OpenEnv real-world environment for debugging and optimising SQL queries."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Reset")
                        task_dd = gr.Dropdown(
                            choices=list(TASKS.keys()),
                            value="fix_syntax_error",
                            label="Task",
                        )
                        reset_btn = gr.Button("Reset Episode")
                        reset_out = gr.Code(language="json", label="Observation")
                        reset_btn.click(gradio_reset, inputs=[task_dd], outputs=[reset_out])

                    with gr.Column():
                        gr.Markdown("### Step")
                        sql_in = gr.Code(language="sql", label="SQL Query")
                        reasoning_in = gr.Textbox(label="Reasoning (optional)")
                        step_btn = gr.Button("Submit Query")
                        step_out = gr.Code(language="json", label="Observation + Reward")
                        step_btn.click(
                            gradio_step,
                            inputs=[sql_in, reasoning_in],
                            outputs=[step_out],
                        )

                with gr.Row():
                    state_btn = gr.Button("Get State")
                    state_out = gr.Code(language="json", label="Episode State")
                    state_btn.click(gradio_state, inputs=[], outputs=[state_out])

            app = gr.mount_gradio_app(app, demo, path="/web")
            logger.info("Gradio UI mounted at /web")

        except ImportError:
            logger.warning("gradio not installed — web UI disabled.")

    return app


app = create_app()
