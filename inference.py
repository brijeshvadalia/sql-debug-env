"""
inference.py — SQL Debug Environment Baseline Inference Script
==============================================================

MANDATORY REQUIREMENTS (per hackathon spec):
  - Named 'inference.py' and placed in root directory ✓
  - Uses OpenAI Client for all LLM calls ✓
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment ✓
  - Runs all 3 tasks and produces reproducible scores ✓
  - Completes in < 20 minutes on 2vCPU / 8GB ✓

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \\
    HF_TOKEN=hf_xxx \\
    python inference.py

    # Against local dev server:
    ENV_BASE_URL=http://localhost:7860 \\
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \\
    HF_TOKEN=hf_xxx \\
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Optional

from openai import OpenAI

from client import SQLDebugEnv
from models import SQLAction, SQLObservation

# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS     = 6       # generous but keeps total runtime well under 20 min
TEMPERATURE   = 0.1     # near-deterministic for reproducibility
MAX_TOKENS    = 512
TASK_IDS      = ["fix_syntax_error", "fix_logic_error", "optimize_query"]

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to respond with raw SQL only
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL engineer and database performance specialist.
    You will be given a broken or slow SQL query along with the database schema.

    Your job:
    1. Diagnose the problem with the query.
    2. Output a corrected or optimised SQL query.

    OUTPUT FORMAT (strict):
    - Output ONLY the SQL query. Nothing else.
    - No markdown fences (no ```sql). No explanation. No preamble.
    - End with a semicolon.
    - If you need to think, do it silently before outputting.

    RULES:
    - Preserve the exact column names and aliases specified in the task description.
    - Do not add or remove columns from the SELECT list unless required to fix a bug.
    - Use standard SQLite-compatible SQL only.
    - If the task requires ORDER BY, preserve it.
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_prompt(obs: SQLObservation, step: int, history: list[str]) -> str:
    lines = [
        f"=== Task: {obs.task_id} (Step {step}/{obs.max_steps}) ===",
        "",
        "TASK DESCRIPTION:",
        obs.task_description,
        "",
        "SCHEMA:",
        obs.schema_hint,
        "",
        "BROKEN / SLOW QUERY (what you must fix):",
        obs.broken_query,
        "",
    ]

    if obs.error_message:
        lines += ["LAST EXECUTION ERROR:", obs.error_message, ""]

    if obs.query_result is not None:
        sample = obs.query_result[:3]
        lines += [
            "LAST QUERY RESULT (first 3 rows):",
            json.dumps(sample, indent=2),
            "",
        ]

    if obs.reward_breakdown:
        bd = obs.reward_breakdown
        lines += [
            f"LAST REWARD: {obs.reward:.4f}",
            f"  Correctness={bd.correctness:.4f}  "
            f"Efficiency={bd.efficiency:.4f}  "
            f"StepPenalty={bd.step_penalty:.4f}",
            f"  Explanation: {bd.explanation}",
            "",
        ]

    if history:
        lines += ["STEP HISTORY (last 3):"]
        lines += history[-3:]
        lines += [""]

    lines.append("Output ONLY the corrected SQL query:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extract SQL from model response
# ---------------------------------------------------------------------------

def extract_sql(response_text: str) -> str:
    """
    Strip markdown fences and whitespace from the model's response.
    Returns raw SQL string.
    """
    text = response_text.strip()
    # Remove ```sql ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    # Remove leading "SQL:" labels
    for prefix in ("SQL:", "Answer:", "Query:"):
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix):].strip()
    return text or "SELECT 1;"   # fallback no-op


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(
    llm: OpenAI,
    env_client,
    task_id: str,
) -> tuple[float, list[str]]:
    """
    Run one full episode for task_id.

    Returns:
        (final_reward, step_log)
    """
    obs = env_client.reset(task_id=task_id)
    print(f"\n  Task: {task_id}  |  max_steps={obs.max_steps}")
    print(f"  Broken query preview: {obs.broken_query[:80].replace(chr(10), ' ')}…")

    history: list[str] = []
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            print(f"  Episode done at step {step - 1}.")
            break

        user_prompt = build_prompt(obs, step, history)

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
            raw_response = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  LLM call failed at step {step}: {exc}")
            raw_response = "SELECT 1;"

        sql = extract_sql(raw_response)
        print(f"  Step {step}: {sql[:70].replace(chr(10), ' ')}…")

        obs = env_client.step(SQLAction(sql_query=sql))
        final_reward = obs.reward

        history_line = (
            f"  Step {step}: reward={obs.reward:.4f}"
            + (f" error={obs.error_message[:60]}" if obs.error_message else "")
        )
        history.append(history_line)
        print(history_line)

        if obs.done:
            print(f"  ✅ Done — reward={obs.reward:.4f}")
            break

    return final_reward, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" SQL Debug Environment — Baseline Inference")
    print("=" * 60)
    print(f" API_BASE_URL : {API_BASE_URL}")
    print(f" MODEL_NAME   : {MODEL_NAME}")
    print(f" ENV_BASE_URL : {ENV_BASE_URL}")
    print(f" MAX_STEPS    : {MAX_STEPS}")
    print("=" * 60)

    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Verify server is reachable before starting
    import httpx
    try:
        r = httpx.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"\n✅ Server health check passed: {r.json()}")
    except Exception as exc:
        print(f"\n❌ Server health check FAILED: {exc}")
        print("   Is the environment server running?")
        print(f"   Expected: {ENV_BASE_URL}/health")
        sys.exit(1)

    scores: dict[str, float] = {}
    t_start = time.time()

    with SQLDebugEnv(base_url=ENV_BASE_URL).sync() as env:
        for task_id in TASK_IDS:
            print(f"\n{'─' * 50}")
            print(f"Running task: {task_id}")
            print(f"{'─' * 50}")

            task_start = time.time()
            score, _ = run_task(llm, env, task_id)
            elapsed = time.time() - task_start

            scores[task_id] = score
            print(f"\n  Task {task_id}: score={score:.4f}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print(" BASELINE SCORES")
    print("=" * 60)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:<25} {score:.4f}  |{bar:<20}|")
    mean_score = sum(scores.values()) / len(scores)
    print(f"\n  Mean score: {mean_score:.4f}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 60)

    # Output machine-readable JSON for automated evaluation
    result = {
        "scores": scores,
        "mean_score": mean_score,
        "model": MODEL_NAME,
        "total_elapsed_seconds": round(total_elapsed, 2),
    }
    print("\nJSON output (for automated evaluation):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
