# inference.py
"""
Inference Script — CodeReview-Env
===================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM
  MODEL_NAME     The model identifier
  HF_TOKEN       Your Hugging Face / API key

Run: python inference.py
"""

import os
import json
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI
from env import CodeReviewEnv, list_all_tasks
from env.models import Action
from env.tasks import get_task_description_for_prompt
from typing import Dict, Any, Optional, List

# Load .env file (keys stored here, never hardcoded)
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (from environment variables — never hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 500


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def build_system_prompt(task_level: int) -> str:
    """Build the system prompt for the LLM agent based on the task level.

    Args:
        task_level: 1=Easy (detection), 2=Medium (classification), 3=Hard (full review).

    Returns:
        A formatted system prompt string.
    """
    task_instructions = get_task_description_for_prompt(task_level)

    if task_level == 1:
        json_format = (
            '{"has_bug": true, "bug_type": "no_bug", "severity": "none", '
            '"suggested_fix": ""}'
        )
    elif task_level == 2:
        json_format = (
            '{"has_bug": true, "bug_type": '
            '"logic_error|security_vulnerability|performance_issue|syntax_error|no_bug", '
            '"severity": "none", "suggested_fix": ""}'
        )
    else:
        json_format = (
            '{"has_bug": true, "bug_type": '
            '"logic_error|security_vulnerability|performance_issue|syntax_error|no_bug", '
            '"severity": "low|medium|high|critical|none", '
            '"suggested_fix": "Specific implementable fix here"}'
        )

    return (
        "You are an expert AI code reviewer with 10+ years of experience.\n"
        "You review code across: correctness, security, performance, quality, maintainability.\n\n"
        f"TASK LEVEL {task_level} INSTRUCTIONS:\n"
        f"{task_instructions}\n\n"
        f"OUTPUT FORMAT — Return ONLY valid JSON:\n{json_format}\n\n"
        "SCORING AWARENESS:\n"
        "- Be precise. Avoid hallucination.\n"
        "- Be complete. Don't miss major issues.\n"
        "- Be specific in fixes. Vague suggestions score 0.\n"
        "- If no bug exists, say so clearly with has_bug=false.\n"
    )


def build_user_prompt(observation) -> str:
    """Build the user prompt from an Observation object.

    Args:
        observation: The Observation returned by env.reset() or env.step().

    Returns:
        A formatted user prompt string.
    """
    hint = observation.context_hint or ""
    hint_line = f"\n{hint}\n" if hint else ""

    return (
        f"Review this {observation.language} code snippet:\n\n"
        f"```{observation.language}\n"
        f"{observation.code}\n"
        f"```\n"
        f"{hint_line}\n"
        "Return ONLY the JSON object. No explanation outside the JSON."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> Action:
    """Parse the LLM response into an Action object.

    Handles markdown code fences and common JSON issues.
    Falls back to a safe default on any parse failure.

    Args:
        response_text: Raw text from the LLM.

    Returns:
        A valid Action object (never None, never crashes).
    """
    safe_default = Action(
        has_bug=False,
        bug_type="no_bug",
        severity="none",
        suggested_fix="",
    )

    if not response_text:
        return safe_default

    text = response_text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return safe_default
        else:
            return safe_default

    # Validate and build Action
    try:
        return Action(
            has_bug=bool(parsed.get("has_bug", False)),
            bug_type=parsed.get("bug_type", "no_bug"),
            severity=parsed.get("severity", "none"),
            suggested_fix=str(parsed.get("suggested_fix", "")),
        )
    except Exception:
        return safe_default


# ──────────────────────────────────────────────────────────────────────────────
# Task runner
# ──────────────────────────────────────────────────────────────────────────────

def run_task(
    env: CodeReviewEnv,
    task,
    client: OpenAI,
    num_episodes: int = 5,
) -> Dict[str, Any]:
    """Run multiple episodes for a single task level.

    Args:
        env: The CodeReviewEnv instance.
        task: TaskConfig for this task.
        client: OpenAI client.
        num_episodes: Number of snippets to evaluate.

    Returns:
        Dict with task results including scores and statistics.
    """
    system_prompt = build_system_prompt(task.level)
    scores: List[float] = []

    print(f"\n{'='*60}")
    print(f"  Task {task.level} — {task.name}")
    print(f"{'='*60}")

    for i in range(num_episodes):
        # Reset environment for a new snippet
        obs = env.reset()

        # Build prompts
        user_prompt = build_user_prompt(obs)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  Episode {i+1}/{num_episodes}: LLM error — {e}")
            raw_response = ""

        # Parse the LLM response into an Action
        action = parse_action(raw_response)

        # Step the environment
        _, reward, done, info = env.step(action)
        scores.append(reward.score)

        print(
            f"  Episode {i+1}/{num_episodes}: "
            f"score={reward.score:.3f} | {reward.feedback}"
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    print(f"\n  Mean: {mean_score:.3f}  Min: {min_score:.3f}  Max: {max_score:.3f}")

    return {
        "task_id": task.task_id,
        "level": task.level,
        "name": task.name,
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "episodes": num_episodes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_tasks() -> Dict[str, Any]:
    """Run all three task levels and print a summary.

    Returns:
        Dict containing all task results.

    Raises:
        SystemExit: If API_KEY or MODEL_NAME is not set.
    """
    # Validate required environment variables
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable is not set.")
        print("Set it with: export HF_TOKEN='your-token-here'")
        sys.exit(1)

    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is not set.")
        print("Set it with: export MODEL_NAME='your-model-name'")
        sys.exit(1)

    # Create OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_tasks = list_all_tasks()
    results: List[Dict[str, Any]] = []

    for task in all_tasks:
        env = CodeReviewEnv(task_level=task.level)
        result = run_task(env, task, client, num_episodes=5)
        results.append(result)

    # Print summary
    overall_scores = [r["mean_score"] for r in results]
    overall_mean = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    print(f"\n{'='*60}")
    print("  CodeReview-Env — Baseline Results")
    print(f"{'='*60}")
    for r in results:
        label = f"  Task {r['level']} — {r['name']}:"
        print(f"{label:<48} {r['mean_score']:.3f}")
    print(f"{'-'*60}")
    print(f"{'  Overall mean score:':<48} {overall_mean:.3f}")
    print(f"{'='*60}\n")

    return {
        "tasks": results,
        "overall_mean": round(overall_mean, 4),
    }


if __name__ == "__main__":
    run_all_tasks()
