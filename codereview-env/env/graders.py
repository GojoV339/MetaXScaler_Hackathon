# env/graders.py
"""
Grading logic for CodeReview-Env.
Implements deterministic scoring for Tasks 1-2, and LLM-judged scoring for Task 3.
"""

from env.models import Action, Reward
from openai import OpenAI
import os
import json
from typing import Dict, Any, Callable


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Bug Detection (deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def grade_task1(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """Grade a Task 1 (Bug Detection) action.

    Scoring: 1.0 if detection is correct, 0.0 otherwise.
    """
    correct = action.has_bug == ground_truth["has_bug"]
    score = 1.0 if correct else 0.0

    if correct:
        if action.has_bug:
            feedback = "Correct! You correctly identified that this snippet contains a bug."
        else:
            feedback = "Correct! You correctly identified that this snippet is bug-free."
    else:
        if ground_truth["has_bug"]:
            feedback = "Incorrect. This snippet contains a bug, but you said it was clean."
        else:
            feedback = "Incorrect. This snippet is clean, but you flagged it as buggy."

    return Reward(
        score=score,
        breakdown={"detection": score},
        feedback=feedback,
        is_correct=correct,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Bug Classification (deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def grade_task2(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """Grade a Task 2 (Bug Classification) action.

    Scoring:
    - +0.4 for correct detection (has_bug)
    - +0.6 for correct classification (bug_type), only if detection is also correct
    """
    detection_score = 0.0
    classification_score = 0.0
    feedback_parts = []

    if action.has_bug == ground_truth["has_bug"]:
        detection_score = 0.4
        feedback_parts.append("Detection correct (+0.4).")

        if action.bug_type == ground_truth["bug_type"]:
            classification_score = 0.6
            feedback_parts.append(
                f"Classification correct: '{action.bug_type}' (+0.6)."
            )
        else:
            if action.has_bug and ground_truth["has_bug"]:
                feedback_parts.append(
                    f"Classification wrong: expected '{ground_truth['bug_type']}', "
                    f"got '{action.bug_type}' (+0.0)."
                )
            else:
                # Both said no_bug — classification matches implicitly
                classification_score = 0.6
                feedback_parts.append("No bug — classification matches (+0.6).")
    else:
        feedback_parts.append(
            f"Detection wrong: expected has_bug={ground_truth['has_bug']}, "
            f"got has_bug={action.has_bug} (+0.0)."
        )

    total = detection_score + classification_score

    return Reward(
        score=total,
        breakdown={
            "detection": detection_score,
            "classification": classification_score,
        },
        feedback=" ".join(feedback_parts),
        is_correct=total >= 0.7,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Full Code Review (deterministic + LLM judge)
# ──────────────────────────────────────────────────────────────────────────────

def grade_task3(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """Grade a Task 3 (Full Code Review) action.

    Scoring:
    - 0.15 for detection
    - 0.15 for bug type classification
    - 0.20 for severity assessment
    - 0.50 for fix quality (LLM-judged)
    - Penalty for reward hacking
    """
    detection_score = 0.15 if action.has_bug == ground_truth["has_bug"] else 0.0
    type_score = 0.15 if action.bug_type == ground_truth["bug_type"] else 0.0
    severity_score = 0.20 if action.severity == ground_truth["severity"] else 0.0

    # LLM judge for fix quality (only if there IS a bug and agent detected it)
    if ground_truth["has_bug"] and action.has_bug and action.suggested_fix:
        fix_raw = _llm_judge(action, ground_truth)
    elif not ground_truth["has_bug"] and not action.has_bug:
        # No bug — fix is irrelevant, full marks for this component
        fix_raw = 1.0
    else:
        fix_raw = 0.0

    fix_score = fix_raw * 0.50
    penalty = _penalize_reward_hack(action)

    total = max(
        min(detection_score + type_score + severity_score + fix_score + penalty, 1.0),
        0.0,
    )

    feedback_parts = [
        f"Detection: {'✓' if detection_score > 0 else '✗'} ({detection_score:.2f})",
        f"Type: {'✓' if type_score > 0 else '✗'} ({type_score:.2f})",
        f"Severity: {'✓' if severity_score > 0 else '✗'} ({severity_score:.2f})",
        f"Fix quality: {fix_score:.2f}/0.50",
    ]
    if penalty < 0:
        feedback_parts.append(f"Penalty: {penalty:.2f} (vague / low-effort fix)")

    return Reward(
        score=round(total, 4),
        breakdown={
            "detection": detection_score,
            "classification": type_score,
            "severity": severity_score,
            "fix_quality": round(fix_score, 4),
        },
        feedback=" | ".join(feedback_parts),
        is_correct=total >= 0.7,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM Judge (private)
# ──────────────────────────────────────────────────────────────────────────────

_LLM_JUDGE_SYSTEM = (
    "You are an expert code review evaluator.\n"
    "You will be given:\n"
    "1. A ground truth bug description and correct fix\n"
    "2. An agent's suggested fix\n\n"
    "Evaluate the agent's fix on three criteria:\n"
    "- Factual accuracy (0.0-1.0): Is the fix technically correct?\n"
    "- Completeness (0.0-1.0): Does it cover the issue fully with specifics?\n"
    "- Actionability (0.0-1.0): Can a developer implement this immediately?\n\n"
    'Return ONLY a JSON object with this exact structure:\n'
    '{\n'
    '  "factual_accuracy": 0.0-1.0,\n'
    '  "completeness": 0.0-1.0,\n'
    '  "actionability": 0.0-1.0,\n'
    '  "overall": 0.0-1.0,\n'
    '  "reason": "one sentence explanation"\n'
    '}'
)


def _llm_judge(action: Action, ground_truth: Dict[str, Any]) -> float:
    """Use an LLM to evaluate the quality of a suggested fix.

    Returns a float between 0.0 and 1.0.
    Falls back to 0.3 on any error.
    """
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME")

    if not api_key or not model:
        return 0.3  # Fallback — no API configured

    user_message = (
        f"Ground truth fix: {ground_truth.get('fix', 'N/A')}\n"
        f"Agent's suggested fix: {action.suggested_fix}\n"
        f"Code context: {ground_truth.get('code', 'N/A')}"
    )

    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        result = json.loads(raw)
        overall = float(result.get("overall", 0.3))
        return max(0.0, min(overall, 1.0))

    except Exception:
        return 0.3  # Safe fallback


# ──────────────────────────────────────────────────────────────────────────────
# Reward-hack penalty (private)
# ──────────────────────────────────────────────────────────────────────────────

def _penalize_reward_hack(action: Action) -> float:
    """Return a penalty (negative value) if the action looks like reward hacking.

    Returns -0.1 if the fix is suspiciously low-effort; 0.0 otherwise.
    """
    fix = action.suggested_fix.strip()

    if len(fix) < 10:
        return -0.1
    if fix.lower() in ["add error handling", "fix the bug", "handle the error"]:
        return -0.1
    if "TODO" in fix or "fix later" in fix.lower():
        return -0.1

    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Grader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_grader(task_level: int) -> Callable[[Action, Dict[str, Any]], Reward]:
    """Return the grading function for the given task level.

    Args:
        task_level: 1, 2, or 3.

    Returns:
        A callable (action, ground_truth) -> Reward.
    """
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_level not in graders:
        raise ValueError(f"Invalid task level: {task_level}. Must be 1, 2, or 3.")
    return graders[task_level]
