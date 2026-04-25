# env/pr_graders.py
"""
PR-level graders: comparison grading + final verdict grading.
File-level grading delegates to existing graders.py.

Uses Kimi K2.6 (Moonshot AI) for LLM judge calls.
Thinking mode is DISABLED for fast, deterministic scoring.
"""
from env.models import PRAction, PRReward, Action
from env.graders import get_grader
from openai import OpenAI
import os
import json
from typing import List, Dict, Any


def grade_file_step(action: PRAction, ground_truth: dict) -> PRReward:
    """Grade a regular file_review step. Delegates to the appropriate task grader."""
    task_type = ground_truth.get("task_type", 1)
    grader_fn = get_grader(task_type)

    file_action = Action(
        has_bug=action.has_bug,
        bug_type=action.bug_type,
        severity=action.severity,
        suggested_fix=action.suggested_fix,
    )
    reward = grader_fn(file_action, ground_truth)

    return PRReward(
        score=reward.score,
        breakdown=reward.breakdown,
        feedback=reward.feedback,
        is_correct=reward.is_correct,
        step_type="file_review",
    )


def grade_comparison_step(action: PRAction, ground_truth: dict) -> PRReward:
    """Grade a code comparison step.

    Two layers:
    1. Deterministic: version choice correct? (0.5 weight)
    2. LLM judge: reasoning quality? (0.5 weight)
    """
    if action.better_version is None:
        return PRReward(
            score=0.01,
            breakdown={"version_choice": 0.0, "reasoning": 0.0},
            feedback="No comparison verdict provided. Return better_version field.",
            is_correct=False,
            step_type="comparison",
        )

    correct_version = ground_truth.get("better_version", "v2")
    version_score = 0.5 if action.better_version == correct_version else 0.0

    # LLM judge for reasoning quality
    reasoning_score = 0.0
    if action.comparison_reason and len(action.comparison_reason) > 15:
        reasoning_score = _llm_judge_comparison(action, ground_truth) * 0.5

    # Anti reward hacking: vague reasoning scores 0
    if action.comparison_reason in [None, "", "v2 is better", "v1 is better"]:
        reasoning_score = 0.0

    total = max(0.01, min(0.99, version_score + reasoning_score))

    return PRReward(
        score=total,
        breakdown={"version_choice": version_score, "reasoning": reasoning_score},
        feedback=f"Version: {'correct' if version_score > 0 else 'wrong'} (expected {correct_version}). Reasoning: {reasoning_score:.2f}",
        is_correct=total >= 0.6,
        step_type="comparison",
    )


def grade_verdict_step(
    action: PRAction,
    ground_truth: dict,
    per_file_scores: List[float],
) -> PRReward:
    """Grade the final PR verdict step.

    4 INDEPENDENT components:
    1. Verdict correctness (deterministic) — 0.50
    2. Summary quality (LLM judge) — 0.30
    3. Issues completeness — 0.10
    4. Anti reward hacking penalty — up to -0.20
    """
    if action.verdict is None:
        return PRReward(
            score=0.01,
            breakdown={"verdict": 0.0, "summary": 0.0, "issues": 0.0, "penalty": 0.0},
            feedback="No verdict provided. Must return APPROVE, REQUEST_CHANGES, or REJECT.",
            is_correct=False,
            step_type="final_verdict",
        )

    correct_verdict = ground_truth.get("correct_verdict", "REQUEST_CHANGES")

    # Component 1: Verdict correctness (deterministic)
    verdict_score = 0.0
    if action.verdict == correct_verdict:
        verdict_score = 0.50
    elif action.verdict == "APPROVE" and correct_verdict == "REJECT":
        verdict_score = 0.0  # critical failure
    elif action.verdict == "REJECT" and correct_verdict == "APPROVE":
        verdict_score = 0.0  # blocking clean code

    # Component 2: Summary quality (LLM judge)
    summary_score = 0.0
    if action.verdict_summary and len(action.verdict_summary) > 20:
        summary_score = _llm_judge_verdict(action, ground_truth) * 0.30

    # Component 3: Issues list completeness
    issues_score = 0.0
    if action.verdict in ["REQUEST_CHANGES", "REJECT"]:
        if action.critical_issues and len(action.critical_issues) > 0:
            issues_score = 0.10

    # Component 4: Anti reward hacking
    penalty = 0.0
    file_mean = sum(per_file_scores) / max(len(per_file_scores), 1)

    # Always-approve hack
    if action.verdict == "APPROVE" and file_mean < 0.25:
        penalty = -0.20

    # Empty verdict summary
    if not action.verdict_summary or len(action.verdict_summary) < 10:
        penalty -= 0.10

    # Detect repeated critical issues (copy-paste hack)
    if action.critical_issues and len(action.critical_issues) > 1:
        unique_issues = set(action.critical_issues)
        if len(unique_issues) < len(action.critical_issues):
            penalty -= 0.10

    total = max(0.01, min(0.99, verdict_score + summary_score + issues_score + penalty))

    return PRReward(
        score=total,
        breakdown={
            "verdict": max(0.0, verdict_score),
            "summary": summary_score,
            "issues": issues_score,
            "penalty": penalty,
        },
        feedback=(
            f"Verdict: {'CORRECT' if action.verdict == correct_verdict else f'WRONG (expected {correct_verdict})'}. "
            f"Summary: {summary_score:.2f}. Issues: {issues_score:.2f}. Penalty: {penalty:.2f}"
        ),
        is_correct=total >= 0.5,
        step_type="final_verdict",
    )


def _llm_judge_comparison(action: PRAction, ground_truth: dict) -> float:
    """LLM judge for comparison reasoning. Kimi K2.6, thinking disabled. Returns 0.0-1.0."""
    try:
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://api.moonshot.ai/v1"),
            api_key=os.getenv("MOONSHOT_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
        )
        system = (
            "Evaluate whether the agent correctly identified which code version is better and why. "
            'Return ONLY JSON: {"overall": 0.0-1.0, "reason": "one sentence"}'
        )
        user = (
            f"Correct answer: version {ground_truth.get('better_version')} is better\n"
            f"Correct reason: {ground_truth.get('comparison_reason', '')}\n"
            f"Agent chose: {action.better_version}\n"
            f"Agent reasoning: {action.comparison_reason}"
        )

        kwargs = {
            "model": os.getenv("MODEL_NAME", "kimi-k2.6"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 100,
            "temperature": 0.6,
        }
        # Only add extra_body for Kimi/Moonshot
        if "moonshot" in kwargs.get("model", "") or "kimi" in os.getenv("MODEL_NAME", ""):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = client.chat.completions.create(**kwargs)
        return float(json.loads(resp.choices[0].message.content.strip()).get("overall", 0.3))
    except Exception:
        return 0.3


def _llm_judge_verdict(action: PRAction, ground_truth: dict) -> float:
    """LLM judge for verdict summary quality. Kimi K2.6, thinking disabled. Returns 0.0-1.0."""
    try:
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://api.moonshot.ai/v1"),
            api_key=os.getenv("MOONSHOT_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
        )
        system = (
            "Evaluate the quality of this PR review verdict. "
            'Return ONLY JSON: {"overall": 0.0-1.0, "reason": "one sentence"}'
        )
        user = (
            f"Correct verdict: {ground_truth.get('correct_verdict')}\n"
            f"Key reason: {ground_truth.get('verdict_reason', '')}\n"
            f"Agent verdict: {action.verdict}\n"
            f"Agent summary: {action.verdict_summary}\n"
            f"Agent issues: {action.critical_issues}"
        )

        kwargs = {
            "model": os.getenv("MODEL_NAME", "kimi-k2.6"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 100,
            "temperature": 0.6,
        }
        if "moonshot" in kwargs.get("model", "") or "kimi" in os.getenv("MODEL_NAME", ""):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        resp = client.chat.completions.create(**kwargs)
        return float(json.loads(resp.choices[0].message.content.strip()).get("overall", 0.3))
    except Exception:
        return 0.3
