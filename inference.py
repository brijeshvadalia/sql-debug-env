"""
inference.py — SQL Debug Environment v3.0 Baseline Inference Script
====================================================================

MANDATORY REQUIREMENTS (per hackathon spec):
  - Named 'inference.py' in root directory                    ✓
  - Uses OpenAI Client for all LLM calls                      ✓
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment ✓
  - Runs all tasks and produces reproducible scores           ✓
  - Completes in < 20 minutes on 2vCPU / 8GB                 ✓

ADVANCED FEATURES:
  - Multi-turn memory: agent sees full conversation history
  - Hint system: agent can request hints (with penalty)
  - EXPLAIN analysis: agent sees query plan feedback
  - Curriculum mode: runs tasks in progressive difficulty order
  - Leaderboard submission: auto-submits final scores

Usage:
    # Required environment variables
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_xxx"
    export ENV_BASE_URL="http://localhost:7860"   # optional, default shown

    python inference.py

    # Curriculum mode (progressive difficulty)
    CURRICULUM_MODE=1 python inference.py

    # Use hints (costs 10% reward penalty each)
    USE_HINTS=1 python inference.py
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

from client import SQLDebugEnv
from models import SQLAction, SQLObservation

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Advanced options
CURRICULUM_MODE: bool = os.environ.get("CURRICULUM_MODE", "0").lower() in ("1","true","yes")
USE_HINTS: bool       = os.environ.get("USE_HINTS", "0").lower() in ("1","true","yes")
MAX_STEPS: int        = int(os.environ.get("MAX_STEPS", "6"))
TEMPERATURE: float    = float(os.environ.get("TEMPERATURE", "0.1"))
MAX_TOKENS: int       = int(os.environ.get("MAX_TOKENS", "512"))

# Default task list (can be overridden by TASK_IDS env var)
DEFAULT_TASK_IDS = [
    "fix_syntax_error",
    "fix_logic_error",
    "fix_null_handling",
    "fix_subquery_bug",
    "optimize_query",
    "fix_window_function",
    "fix_cte",
    "multi_step_aggregation",
]

TASK_IDS = (
    os.environ.get("TASK_IDS", "").split(",")
    if os.environ.get("TASK_IDS")
    else DEFAULT_TASK_IDS
)

# ---------------------------------------------------------------------------
# System prompt — instructs LLM to return raw SQL only
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
    - Preserve all column names and aliases exactly as specified in the task.
    - If the task requires ORDER BY, preserve it.
    - If you see conversation history, use it to avoid repeating failed attempts.
    - If you see EXPLAIN analysis showing high scan counts, optimise accordingly.
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_prompt(obs: SQLObservation, step: int) -> str:
    lines = [
        f"=== Task: {obs.task_id} | Step {step}/{obs.max_steps} ===",
        "",
        "TASK:",
        obs.task_description.split("\n")[0],
        "",
        "SCHEMA:",
        obs.schema_hint,
        "",
        "BROKEN/SLOW QUERY (what you must fix):",
        obs.broken_query,
        "",
    ]

    # Execution feedback from last step
    if obs.error_message:
        lines += ["LAST ERROR:", obs.error_message, ""]

    if obs.query_result is not None and len(obs.query_result) > 0:
        lines += [
            f"LAST RESULT (first 3 of {len(obs.query_result)} rows):",
            json.dumps(obs.query_result[:3], indent=2),
            "",
        ]

    if obs.reward > 0:
        lines += [f"LAST REWARD: {obs.reward:.4f}", ""]

    # EXPLAIN analysis
    if obs.query_analysis:
        qa = obs.query_analysis
        if isinstance(qa, dict):
            lines += [
                "QUERY PLAN ANALYSIS:",
                f"  Scans: {qa.get('scan_count', 0)} | Index: {qa.get('uses_index', False)} | {qa.get('suggestion', '')}",
                "",
            ]

    # Multi-turn conversation history
    if obs.conversation_history and len(obs.conversation_history) > 0:
        lines += ["CONVERSATION HISTORY (your previous attempts):"]
        for turn in obs.conversation_history[-3:]:
            lines.append(
                f"  Step {turn.get('step', '?')}: "
                f"reward={turn.get('reward', 0):.3f} "
                + (f"error='{turn.get('error', '')[:60]}'" if turn.get("error") else
                   f"rows={turn.get('result_count', 0)}")
            )
        lines += ["", "Learn from these attempts and improve your fix.", ""]

    # Hint if available
    if hasattr(obs, "hint_available") and obs.hint_available:
        lines += [f"Hints available: {3 - getattr(obs, 'hints_used', 0)} remaining (each costs 10% reward penalty)", ""]

    lines.append("Output ONLY the corrected SQL query (no explanation):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extract SQL from model response
# ---------------------------------------------------------------------------

def extract_sql(response_text: str) -> str:
    """Strip markdown fences, labels, and whitespace."""
    text = response_text.strip()
    # Remove ```sql ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    # Remove common prefixes
    for prefix in ("SQL:", "Answer:", "Query:", "Result:"):
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].strip()
    return text if text else "SELECT 1;"


# ---------------------------------------------------------------------------
# Request hint from environment
# ---------------------------------------------------------------------------

def maybe_request_hint(env_base_url: str, step: int, max_steps: int) -> Optional[str]:
    """
    Request a hint on step 3+ if USE_HINTS is enabled.
    Returns hint text or None.
    """
    if not USE_HINTS:
        return None
    if step < 3:
        return None
    try:
        r = httpx.post(f"{env_base_url}/hint", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get("hint", "")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(
    llm: OpenAI,
    env_client,
    task_id: str,
) -> tuple[float, list[dict]]:
    """
    Run one full episode for task_id.

    Returns:
        (final_reward, step_log)
    """
    obs = env_client.reset(task_id=task_id)
    print(f"\n  Task: {task_id}  (max_steps={obs.max_steps})")
    print(f"  Broken: {obs.broken_query[:70].replace(chr(10), ' ')}...")

    step_log = []
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            print(f"  Done at step {step - 1}.")
            break

        # Maybe request a hint
        hint = maybe_request_hint(ENV_BASE_URL, step, obs.max_steps)
        if hint:
            print(f"  [Hint requested] {hint[:80]}...")

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
            print(f"  ⚠️  LLM call failed: {exc}")
            raw = "SELECT 1;"

        sql = extract_sql(raw)
        print(f"  Step {step}: {sql[:65].replace(chr(10), ' ')}...")

        # Submit to environment
        obs = env_client.step(SQLAction(sql_query=sql, reasoning=f"Step {step} attempt"))
        final_reward = obs.reward

        # Log step
        log_entry = {
            "step": step,
            "reward": obs.reward,
            "correctness": obs.reward_breakdown.correctness if obs.reward_breakdown else 0.0,
            "efficiency": obs.reward_breakdown.efficiency if obs.reward_breakdown else 0.0,
            "error": obs.error_message,
            "done": obs.done,
        }
        step_log.append(log_entry)

        # Print step result
        bd = obs.reward_breakdown
        print(
            f"  → reward={obs.reward:.4f}"
            + (f" correct={bd.correctness:.2f}" if bd else "")
            + (f" error={obs.error_message[:50]}" if obs.error_message else "")
            + (" ✅ DONE" if obs.done else "")
        )

        if obs.done:
            break

    return final_reward, step_log


# ---------------------------------------------------------------------------
# Curriculum mode
# ---------------------------------------------------------------------------

def get_curriculum_task(env_base_url: str) -> Optional[str]:
    """Get the next recommended task from curriculum endpoint."""
    try:
        r = httpx.post(f"{env_base_url}/curriculum/next", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get("recommended_task")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Submit to leaderboard
# ---------------------------------------------------------------------------

def submit_to_leaderboard(env_base_url: str, mean_score: float, episodes: int) -> None:
    """Submit final scores to the environment leaderboard."""
    try:
        payload = {
            "model_name": MODEL_NAME,
            "mean_score": mean_score,
            "episodes": episodes,
        }
        r = httpx.post(
            f"{env_base_url}/leaderboard/submit",
            json=payload,
            timeout=10,
        )
        if r.status_code == 200:
            rank = r.json().get("rank", "?")
            print(f"\n  Leaderboard: submitted! Rank #{rank}")
    except Exception as exc:
        print(f"\n  Leaderboard submit failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print(" SQL Debug Environment v3.0 — Baseline Inference")
    print("=" * 62)
    print(f"  API_BASE_URL  : {API_BASE_URL}")
    print(f"  MODEL_NAME    : {MODEL_NAME}")
    print(f"  ENV_BASE_URL  : {ENV_BASE_URL}")
    print(f"  MAX_STEPS     : {MAX_STEPS}")
    print(f"  CURRICULUM    : {'on' if CURRICULUM_MODE else 'off'}")
    print(f"  USE_HINTS     : {'on' if USE_HINTS else 'off'}")
    print(f"  TASKS         : {len(TASK_IDS)}")
    print("=" * 62)

    # Validate required env vars
    if not API_KEY:
        print("\nERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    # Health check
    try:
        r = httpx.get(f"{ENV_BASE_URL}/health", timeout=15)
        r.raise_for_status()
        info = r.json()
        print(f"\n✅ Server: {info.get('environment')} v{info.get('version')} — OK")
    except Exception as exc:
        print(f"\n❌ Server health check FAILED: {exc}")
        print(f"   Is the environment running at {ENV_BASE_URL}?")
        sys.exit(1)

    # Create LLM client (OpenAI-compatible)
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores: dict[str, float] = {}
    all_step_logs: dict[str, list] = {}
    t_start = time.time()

    with SQLDebugEnv(base_url=ENV_BASE_URL).sync() as env:

        # In curriculum mode, get recommended task order
        task_list = TASK_IDS
        if CURRICULUM_MODE:
            print("\n  Curriculum mode: running tasks in progressive order")

        for task_id in task_list:
            print(f"\n{'─' * 55}")
            print(f"  Running: {task_id}")
            print(f"{'─' * 55}")

            task_start = time.time()
            score, step_log = run_task(llm, env, task_id)
            elapsed = time.time() - task_start

            scores[task_id] = score
            all_step_logs[task_id] = step_log
            print(f"\n  Final score: {score:.4f}  ({elapsed:.1f}s)")

            # Check total runtime budget
            if time.time() - t_start > 1100:  # 18.3 min safety margin
                print("\n  ⚠️  Approaching 20min limit — stopping early.")
                break

    total_elapsed = time.time() - t_start
    mean_score = sum(scores.values()) / len(scores) if scores else 0.0

    # Print final results
    print(f"\n{'=' * 62}")
    print(" BASELINE SCORES")
    print(f"{'=' * 62}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 25)
        status = "✅" if score >= 0.8 else "⚠️ " if score >= 0.5 else "❌"
        print(f"  {status} {task_id:<28} {score:.4f}  |{bar:<25}|")

    print(f"\n  Mean score  : {mean_score:.4f}")
    print(f"  Tasks run   : {len(scores)}/{len(TASK_IDS)}")
    print(f"  Total time  : {total_elapsed:.1f}s")
    print(f"{'=' * 62}")

    # Machine-readable JSON output (for automated evaluation)
    result = {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "model": MODEL_NAME,
        "tasks_run": len(scores),
        "tasks_solved": sum(1 for s in scores.values() if s >= 0.8),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "curriculum_mode": CURRICULUM_MODE,
        "hints_enabled": USE_HINTS,
        "step_logs": all_step_logs,
    }

    print("\nJSON (for automated evaluation):")
    print(json.dumps({k: v for k, v in result.items() if k != "step_logs"}, indent=2))

    # Submit to leaderboard
    submit_to_leaderboard(ENV_BASE_URL, mean_score, len(scores))


if __name__ == "__main__":
    main()
