"""
inference.py — SQL Debug Environment v3.0

MANDATORY: Emits structured stdout logs in [START], [STEP], [END] format
as required by the Scaler x Meta PyTorch OpenEnv Hackathon 2026 dashboard.
Any deviation causes incorrect evaluation scoring.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_xxx"
    export ENV_BASE_URL="http://localhost:7860"
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

CURRICULUM_MODE: bool = os.environ.get("CURRICULUM_MODE", "0").lower() in ("1","true","yes")
USE_HINTS: bool       = os.environ.get("USE_HINTS", "0").lower() in ("1","true","yes")
MAX_STEPS: int        = int(os.environ.get("MAX_STEPS", "6"))
TEMPERATURE: float    = float(os.environ.get("TEMPERATURE", "0.1"))
MAX_TOKENS: int       = int(os.environ.get("MAX_TOKENS", "512"))

TASK_IDS = (
    os.environ.get("TASK_IDS", "").split(",")
    if os.environ.get("TASK_IDS")
    else [
        "fix_syntax_error",
        "fix_logic_error",
        "fix_null_handling",
        "fix_subquery_bug",
        "optimize_query",
        "fix_window_function",
        "fix_cte",
        "multi_step_aggregation",
    ]
)

# ---------------------------------------------------------------------------
# MANDATORY structured log format — DO NOT CHANGE FIELD NAMES OR ORDER
# Per Scaler x Meta PyTorch OpenEnv Hackathon 2026 dashboard requirement
# ---------------------------------------------------------------------------

def log_start(task_id: str, episode_id: str) -> None:
    """Emit [START] log — must be first log for every episode."""
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "episode_id": episode_id,
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), flush=True)


def log_step(
    episode_id: str,
    step: int,
    sql_query: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Emit [STEP] log — must be emitted after every step() call."""
    print(json.dumps({
        "type": "[STEP]",
        "episode_id": episode_id,
        "step": step,
        "sql_query": sql_query[:200],
        "reward": round(reward, 4),
        "done": done,
        "error": error,
    }), flush=True)


def log_end(
    episode_id: str,
    task_id: str,
    final_reward: float,
    total_steps: int,
    elapsed_seconds: float,
) -> None:
    """Emit [END] log — must be last log for every episode."""
    print(json.dumps({
        "type": "[END]",
        "episode_id": episode_id,
        "task_id": task_id,
        "final_reward": round(final_reward, 4),
        "total_steps": total_steps,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "model": MODEL_NAME,
    }), flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL engineer and database performance specialist.

    You will be given a broken or slow SQL query with database schema information.
    Your job is to diagnose the problem and output the corrected SQL query.

    STRICT OUTPUT FORMAT:
    - Output ONLY the SQL query. Nothing else.
    - No markdown fences (no ```sql or ```).
    - No explanation, no preamble, no commentary.
    - End with a semicolon.

    RULES:
    - Use standard SQLite-compatible SQL only.
    - Preserve all column names and aliases exactly as specified.
    - If the task requires ORDER BY, preserve it.
    - Use conversation history to avoid repeating failed attempts.
    - If EXPLAIN shows high scan counts, optimise accordingly.
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, step: int) -> str:
    lines = [
        f"=== Task: {obs['task_id']} | Step {step}/{obs['max_steps']} ===",
        "",
        "TASK:",
        obs.get("task_description", "").split("\n")[0],
        "",
        "SCHEMA:",
        obs.get("schema_hint", ""),
        "",
        "BROKEN/SLOW QUERY (what you must fix):",
        obs.get("broken_query", ""),
        "",
    ]

    if obs.get("error_message"):
        lines += ["LAST ERROR:", obs["error_message"], ""]

    if obs.get("query_result") and len(obs["query_result"]) > 0:
        lines += [
            f"LAST RESULT (first 3 of {len(obs['query_result'])} rows):",
            json.dumps(obs["query_result"][:3], indent=2),
            "",
        ]

    if obs.get("reward", 0) > 0:
        lines += [f"LAST REWARD: {obs['reward']:.4f}", ""]

    if obs.get("query_analysis"):
        qa = obs["query_analysis"]
        lines += [
            "QUERY PLAN ANALYSIS:",
            f"  Scans: {qa.get('scan_count',0)} | Index: {qa.get('uses_index',False)} | {qa.get('suggestion','')}",
            "",
        ]

    history = obs.get("conversation_history", [])
    if history:
        lines += ["CONVERSATION HISTORY (your previous attempts):"]
        for turn in history[-3:]:
            lines.append(
                f"  Step {turn.get('step','?')}: reward={turn.get('reward',0):.3f} "
                + (f"error='{turn.get('error','')[:60]}'" if turn.get("error")
                   else f"rows={turn.get('result_count',0)}")
            )
        lines += ["", "Learn from these and improve your fix.", ""]

    lines.append("Output ONLY the corrected SQL query (no explanation):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extract SQL from LLM response
# ---------------------------------------------------------------------------

def extract_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    for prefix in ("SQL:", "Answer:", "Query:", "Result:"):
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].strip()
    return text if text else "SELECT 1;"


# ---------------------------------------------------------------------------
# Hint helper
# ---------------------------------------------------------------------------

def maybe_request_hint(step: int) -> Optional[str]:
    if not USE_HINTS or step < 3:
        return None
    try:
        r = httpx.post(f"{ENV_BASE_URL}/hint", timeout=10)
        if r.status_code == 200:
            return r.json().get("hint")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(llm: OpenAI, task_id: str) -> tuple[float, str, int]:
    """
    Run one episode. Returns (final_reward, episode_id, total_steps).
    Emits mandatory [START], [STEP], [END] logs.
    """
    # Reset episode
    r = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    obs = r.json()

    episode_id = obs.get("episode_id", f"ep_{task_id}_{int(time.time())}")

    # ── MANDATORY: emit [START] ──────────────────────────────────────────
    log_start(task_id=task_id, episode_id=episode_id)

    t_start = time.time()
    final_reward = 0.0
    total_steps = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        # Optional hint
        hint = maybe_request_hint(step)
        if hint:
            print(f"  [Hint] {hint[:80]}...", flush=True)

        # Build prompt and call LLM
        user_prompt = build_prompt(obs, step)
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠️  LLM call failed: {exc}", flush=True)
            raw = "SELECT 1;"

        sql = extract_sql(raw)

        # Submit to environment
        step_r = httpx.post(
            f"{ENV_BASE_URL}/step",
            json={"action": {"sql_query": sql, "reasoning": f"Step {step}"}},
            timeout=30,
        )
        step_r.raise_for_status()
        obs = step_r.json()

        final_reward = obs.get("reward", 0.0)
        total_steps = step

        # ── MANDATORY: emit [STEP] ───────────────────────────────────────
        log_step(
            episode_id=episode_id,
            step=step,
            sql_query=sql,
            reward=final_reward,
            done=obs.get("done", False),
            error=obs.get("error_message"),
        )

        if obs.get("done"):
            break

    elapsed = time.time() - t_start

    # ── MANDATORY: emit [END] ────────────────────────────────────────────
    log_end(
        episode_id=episode_id,
        task_id=task_id,
        final_reward=final_reward,
        total_steps=total_steps,
        elapsed_seconds=elapsed,
    )

    return final_reward, episode_id, total_steps


# ---------------------------------------------------------------------------
# Submit to leaderboard
# ---------------------------------------------------------------------------

def submit_leaderboard(mean_score: float, episodes: int) -> None:
    try:
        r = httpx.post(
            f"{ENV_BASE_URL}/leaderboard/submit",
            json={"model_name": MODEL_NAME, "mean_score": mean_score, "episodes": episodes},
            timeout=10,
        )
        if r.status_code == 200:
            print(f"\n  Leaderboard rank: #{r.json().get('rank','?')}", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print(" SQL Debug Environment v3.0 — Inference")
    print("=" * 62)
    print(f"  MODEL        : {MODEL_NAME}")
    print(f"  ENV_BASE_URL : {ENV_BASE_URL}")
    print(f"  MAX_STEPS    : {MAX_STEPS}")
    print(f"  TASKS        : {len(TASK_IDS)}")
    print("=" * 62, flush=True)

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY not set.", flush=True)
        sys.exit(1)

    # Health check
    try:
        r = httpx.get(f"{ENV_BASE_URL}/health", timeout=15)
        r.raise_for_status()
        v = r.json().get("version", "?")
        print(f"\n✅ Server OK — v{v}\n", flush=True)
    except Exception as exc:
        print(f"\n❌ Server health check failed: {exc}", flush=True)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores: dict[str, float] = {}
    t_total = time.time()

    for task_id in TASK_IDS:
        print(f"\n{'─'*55}")
        print(f"  Task: {task_id}")
        print(f"{'─'*55}", flush=True)

        try:
            score, ep_id, steps = run_task(llm, task_id)
            scores[task_id] = score
            print(f"  Score: {score:.4f}  Steps: {steps}", flush=True)
        except Exception as exc:
            print(f"  ❌ Task failed: {exc}", flush=True)
            scores[task_id] = 0.0

        # 20-minute runtime guard
        if time.time() - t_total > 1100:
            print("\n  ⚠️  Approaching 20min limit — stopping.", flush=True)
            break

    elapsed_total = time.time() - t_total
    mean = sum(scores.values()) / len(scores) if scores else 0.0

    # Final summary
    print(f"\n{'='*62}")
    print(" RESULTS")
    print(f"{'='*62}")
    for tid, sc in scores.items():
        bar = "█" * int(sc * 25)
        status = "✅" if sc >= 0.8 else "⚠️ " if sc >= 0.5 else "❌"
        print(f"  {status} {tid:<30} {sc:.4f}  |{bar:<25}|")

    print(f"\n  Mean   : {mean:.4f}")
    print(f"  Tasks  : {len(scores)}/{len(TASK_IDS)}")
    print(f"  Time   : {elapsed_total:.1f}s")
    print(f"{'='*62}", flush=True)

    # Machine-readable JSON
    result = {
        "scores": scores,
        "mean_score": round(mean, 4),
        "model": MODEL_NAME,
        "tasks_run": len(scores),
        "tasks_solved": sum(1 for s in scores.values() if s >= 0.8),
        "elapsed_seconds": round(elapsed_total, 2),
    }
    print("\nJSON:")
    print(json.dumps(result, indent=2), flush=True)

    submit_leaderboard(mean, len(scores))


if __name__ == "__main__":
    main()
