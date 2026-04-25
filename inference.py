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
import time
from dotenv import load_dotenv
from openai import OpenAI
from env import CodeReviewEnv, list_all_tasks
from env.models import Action
from env.tasks import get_task_description_for_prompt
from env.pr_environment import PRReviewEnv
from env.models import PRAction
from typing import Dict, Any, Optional, List

# Load .env file (keys stored here, never hardcoded)
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (from environment variables — never hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")                     # no default — must be set
API_KEY = HF_TOKEN or os.getenv("API_KEY")           # fallback alias for OpenAI client
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")     # optional — used with from_docker_image()
BENCHMARK = "codereview-env"
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
        # Emit mandatory [START] line
        print(f"[START] task={task.task_id} env={BENCHMARK} model={MODEL_NAME}")

        # Reset environment for a new snippet
        obs = env.reset()

        # Build prompts
        user_prompt = build_user_prompt(obs)

        # Call LLM
        error_msg = "null"
        raw_response = ""
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
            time.sleep(3)
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            print(f"  Episode {i+1}/{num_episodes}: LLM error — {e}")

        # Parse the LLM response into an Action
        action = parse_action(raw_response)

        # Step the environment
        _, reward, done, info = env.step(action)
        scores.append(reward.score)

        # Emit mandatory [STEP] line
        action_str = f"has_bug={action.has_bug},bug_type={action.bug_type}"
        print(f"[STEP] step=1 action={action_str} reward={reward.score:.2f} done={str(done).lower()} error={error_msg}")

        # Emit mandatory [END] line (with score= as per updated spec)
        print(f"[END] success={str(reward.is_correct).lower()} steps=1 score={reward.score:.2f} rewards={reward.score:.2f}")

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

# ──────────────────────────────────────────────────────────────────────────────
# PR Review Pipeline (additive — all existing code above unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def build_pr_system_prompt(step_type: str) -> str:
    """System prompt per step type."""
    base = (
        "You are a senior software engineer performing a Pull Request code review.\n"
        "You have seen production outages from exactly these kinds of bugs.\n"
        "Return ONLY valid JSON. No explanation outside the JSON.\n\n"
    )
    if step_type == "file_review":
        return base + (
            "TASK: Review the code file for bugs, security issues, performance problems, "
            "code smells, or design issues.\n\n"
            "Return this JSON:\n"
            '{"has_bug": true/false, "bug_type": "logic_error|security_vulnerability|'
            'performance_issue|syntax_error|no_bug", "severity": "low|medium|high|critical|none", '
            '"suggested_fix": "specific actionable fix here", '
            '"better_version": null, "comparison_reason": null, '
            '"verdict": null, "verdict_summary": null, "critical_issues": null}'
        )
    elif step_type == "comparison":
        return base + (
            "TASK: Compare VERSION 1 and VERSION 2 of the same code file.\n"
            "Decide which version is better (v1, v2, or equal) and explain specifically why.\n\n"
            "Return this JSON:\n"
            '{"has_bug": false, "bug_type": "no_bug", "severity": "none", "suggested_fix": "", '
            '"better_version": "v1|v2|equal", '
            '"comparison_reason": "specific reason: what exactly is better and why", '
            '"verdict": null, "verdict_summary": null, "critical_issues": null}'
        )
    else:  # final_verdict
        return base + (
            "TASK: Based on all files reviewed, give your final PR verdict.\n"
            "APPROVE = all issues minor or non-existent, safe to merge.\n"
            "REQUEST_CHANGES = fixable issues that must be addressed before merge.\n"
            "REJECT = critical unfixable issues or fundamental design failures.\n\n"
            "Return this JSON:\n"
            '{"has_bug": false, "bug_type": "no_bug", "severity": "none", "suggested_fix": "", '
            '"better_version": null, "comparison_reason": null, '
            '"verdict": "APPROVE|REQUEST_CHANGES|REJECT", '
            '"verdict_summary": "2-sentence summary of main finding", '
            '"critical_issues": ["filename: specific issue", ...]}'
        )


def build_pr_user_prompt(obs) -> str:
    """User prompt based on observation step type."""
    if obs.step_type == "final_verdict":
        findings = "\n".join(f"  - {f}" for f in obs.previous_findings)
        return (
            f"PR: {obs.pr_title}\n"
            f"Description: {obs.pr_description}\n\n"
            f"Files reviewed:\n{findings}\n\n"
            "Give your final verdict. Return ONLY the JSON."
        )
    elif obs.step_type == "comparison":
        ctx = ""
        if obs.previous_findings:
            ctx = "Previous findings:\n" + "\n".join(f"  - {f}" for f in obs.previous_findings) + "\n\n"
        return (
            f"PR: {obs.pr_title}\n"
            f"File: {obs.current_file.file_name} "
            f"(file {obs.files_reviewed + 1}/{obs.total_files})\n\n"
            f"{ctx}"
            f"VERSION 1:\n```{obs.current_file.language}\n{obs.current_file.code}\n```\n\n"
            f"VERSION 2:\n```{obs.current_file.language}\n{obs.current_file.code_v2}\n```\n\n"
            "Which version is better and why? Return ONLY the JSON."
        )
    else:
        ctx = ""
        if obs.previous_findings:
            ctx = "Previous findings:\n" + "\n".join(f"  - {f}" for f in obs.previous_findings) + "\n\n"
        return (
            f"PR: {obs.pr_title}\n"
            f"File: {obs.current_file.file_name} ({obs.current_file.language}) "
            f"(file {obs.files_reviewed + 1}/{obs.total_files})\n\n"
            f"{ctx}"
            f"```{obs.current_file.language}\n{obs.current_file.code}\n```\n\n"
            "Review this file. Return ONLY the JSON."
        )


def parse_pr_action(text: str) -> PRAction:
    """Parse LLM output to PRAction. Never crashes."""
    safe = PRAction(has_bug=False, bug_type="no_bug", severity="none", suggested_fix="")
    if not text:
        return safe
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except Exception:
                return safe
        else:
            return safe
    try:
        return PRAction(**{k: v for k, v in parsed.items() if v is not None})
    except Exception:
        return safe


def run_pr_episode(env: PRReviewEnv, client) -> dict:
    """Run one complete PR review episode."""
    obs = env.reset()
    step_results = []

    while True:
        sys_prompt = build_pr_system_prompt(obs.step_type)
        user_prompt = build_pr_user_prompt(obs)

        try:
            kwargs = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": 600,
            }
            # Kimi K2.6: disable thinking for consistent JSON
            if "kimi" in MODEL_NAME.lower() or "moonshot" in API_BASE_URL.lower():
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = client.chat.completions.create(**kwargs)
            time.sleep(3)
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  LLM error: {e}")
            raw = ""

        action = parse_pr_action(raw)
        next_obs, reward, done, info = env.step(action)

        step_results.append({
            "step_type": info.get("step_type"),
            "file": info.get("file_name", "verdict"),
            "score": reward.score,
            "feedback": reward.feedback,
        })

        tag = f"[{info['step_type']}]"
        print(f"  {tag} {info.get('file_name', 'verdict')}: score={reward.score:.3f}")

        if done:
            ep_score = info.get("episode_score", 0.0)
            verdict = action.verdict
            correct = info.get("correct_verdict")
            print(f"  Episode: {ep_score:.3f} | Verdict: {verdict} (correct: {correct})")
            return {
                "pr_id": info["pr_id"],
                "episode_score": ep_score,
                "verdict": verdict,
                "correct_verdict": correct,
                "verdict_correct": info.get("verdict_correct", False),
                "steps": step_results,
            }

        obs = next_obs


def run_pr_baseline(num_episodes: int = 5) -> dict:
    """Run PR pipeline baseline evaluation."""
    if not API_KEY:
        print("ERROR: API key not set"); sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = PRReviewEnv()
    results = []

    print(f"\n{'='*60}")
    print("  PR Review Pipeline — Baseline")
    print(f"{'='*60}")

    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}:")
        result = run_pr_episode(env, client)
        results.append(result)

    scores = [r["episode_score"] for r in results]
    verdict_acc = sum(1 for r in results if r["verdict_correct"]) / len(results)

    print(f"\n{'='*60}")
    print(f"  Mean episode score:  {sum(scores)/len(scores):.3f}")
    print(f"  Verdict accuracy:    {verdict_acc:.1%}")
    print(f"{'='*60}")
    return {"episodes": results, "mean_score": round(sum(scores)/len(scores), 3)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pr":
        run_pr_baseline(num_episodes=5)
    else:
        run_all_tasks()
