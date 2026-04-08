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

# ──────────────────────────────────────────────────────────────────────────────
# Task 4 — Code Smell Detection
# ──────────────────────────────────────────────────────────────────────────────

def grade_task4(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    expected_smells = ground_truth.get("code_smells", [])
    suggested = action.suggested_fix.lower()
    
    score = 0.0
    count = 0
    
    if expected_smells and expected_smells != ["none"]:
        score_per_smell = 0.8 / max(len(expected_smells), 1)
        for smell in expected_smells:
            if smell.replace("_", " ") in suggested or smell in suggested:
                score += score_per_smell
                count += 1
        
        if count == len(expected_smells):
            score += 0.20
            
    if action.has_bug != (expected_smells != ["none"]):
        score -= 0.20
        
    score = max(0.0, min(score, 1.0))
    is_correct = score >= 0.6
    
    feedback = f"Found {count}/{len(expected_smells)} smells. Score: {score:.2f}."
    
    return Reward(
        score=score,
        breakdown={"smells_found": count, "total_smells": len(expected_smells), "bonus": 0.20 if count == len(expected_smells) and expected_smells != ["none"] else 0.0},
        feedback=feedback,
        is_correct=is_correct,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Task 5 — Security Audit
# ──────────────────────────────────────────────────────────────────────────────

def _llm_judge_security(action: Action, ground_truth: Dict[str, Any]) -> float:
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME")
    if not api_key or not model: return 0.3
    
    sys_prompt = (
        "Evaluate if the fix correctly addresses the OWASP vulnerability. "
        "Score on: correctness of fix (0-1), specificity (0-1), security best practice alignment (0-1), overall (0-1). "
        "Return exact JSON: {\"overall\": 0.0-1.0}"
    )
    user_msg = f"Expected vulns: {ground_truth.get('owasp_issues')}\nAgent fix: {action.suggested_fix}\nCode: {ground_truth.get('code')}"
    
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}], temperature=0.1, max_tokens=200)
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"): raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"): raw = raw[: raw.rfind("```")]
        res = json.loads(raw.strip())
        return max(0.0, min(float(res.get("overall", 0.3)), 1.0))
    except Exception:
        return 0.3

def grade_task5(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    expected_vulns = ground_truth.get("owasp_issues", ["none"])
    has_vuln = expected_vulns != ["none"]
    
    det_score = 0.30 if action.has_bug == has_vuln else 0.0
    type_score = 0.20 if (has_vuln and action.bug_type == "security_vulnerability") else 0.0
    
    expected_severity = ground_truth.get("severity", "none")
    sev_score = 0.10 if (has_vuln and action.severity == expected_severity) else 0.0
    
    fix_score = _llm_judge_security(action, ground_truth) * 0.40 if has_vuln and action.has_bug else (0.40 if not has_vuln else 0.0)
    
    total = min(det_score + type_score + sev_score + fix_score, 1.0)
    
    return Reward(
        score=total, breakdown={"detection": det_score, "type": type_score, "severity": sev_score, "fix": fix_score},
        feedback=f"Score: {total:.2f}", is_correct=total >= 0.6
    )

# ──────────────────────────────────────────────────────────────────────────────
# Task 6 — Performance Optimization
# ──────────────────────────────────────────────────────────────────────────────

def _llm_judge_performance(action: Action, ground_truth: Dict[str, Any]) -> float:
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME")
    if not api_key or not model: return 0.3
    
    sys_prompt = (
        "Evaluate the performance optimization suggestion. "
        "Score on: correctly identified the bottleneck (0-1), suggested improvement is actually faster (0-1), Big-O improvement stated correctly (0-1), overall (0-1). "
        "Return exact JSON: {\"overall\": 0.0-1.0}"
    )
    user_msg = f"Expected bottleneck: {ground_truth.get('performance_issue')}\nAgent fix: {action.suggested_fix}\nCode: {ground_truth.get('code')}"
    
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}], temperature=0.1, max_tokens=200)
        raw = resp.choices[0].message.content.strip()
        if "```" in raw: raw = raw.replace("```json", "").replace("```", "")
        return max(0.0, min(float(json.loads(raw.strip()).get("overall", 0.3)), 1.0))
    except Exception:
        return 0.3

def grade_task6(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    has_perf_issue = ground_truth.get("performance_issue", "none") != "none"
    det_score = 0.20 if action.has_bug == has_perf_issue else 0.0
    
    expected_comp = ground_truth.get("time_complexity", "O(1)")
    comp_score = 0.30 if (has_perf_issue and expected_comp.lower() in action.suggested_fix.lower()) else 0.0
    
    fix_score = _llm_judge_performance(action, ground_truth) * 0.50 if has_perf_issue and action.has_bug else (0.50 if not has_perf_issue else 0.0)
    
    total = min(det_score + comp_score + fix_score, 1.0)
    
    return Reward(
        score=total, breakdown={"detection": det_score, "complexity": comp_score, "fix": fix_score},
        feedback=f"Score: {total:.2f}", is_correct=total >= 0.6
    )

# ──────────────────────────────────────────────────────────────────────────────
# Task 7 — Test Coverage Review
# ──────────────────────────────────────────────────────────────────────────────

def _llm_judge_tests(action: Action, ground_truth: Dict[str, Any]) -> float:
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME")
    if not api_key or not model: return 0.3
    
    sys_prompt = (
        "Evaluate the test coverage suggestions. "
        "Score on: correctly identified missing test cases (0-1), test suggestions are specific with inputs and expected outputs (0-1), covers important edge cases (0-1), overall (0-1). "
        "Return exact JSON: {\"overall\": 0.0-1.0}"
    )
    user_msg = f"Expected missing: {ground_truth.get('missing_tests')}\nAgent fix: {action.suggested_fix}\nCode: {ground_truth.get('code')}"
    
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}], temperature=0.1, max_tokens=200)
        raw = resp.choices[0].message.content.strip()
        if "```" in raw: raw = raw.replace("```json", "").replace("```", "")
        return max(0.0, min(float(json.loads(raw.strip()).get("overall", 0.3)), 1.0))
    except Exception:
        return 0.3

def grade_task7(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    exp_testable = ground_truth.get("is_testable", True)
    exp_missing = ground_truth.get("missing_tests", ["none"])
    
    test_score = 0.20 if action.has_bug == (not exp_testable or exp_missing != ["none"]) else 0.0
    
    missing_score = 0.0
    if len(exp_missing) > 0 and exp_missing != ["none"]:
        for t in exp_missing:
            if t.lower() in action.suggested_fix.lower():
                missing_score += 0.30 / len(exp_missing)
                
    fix_score = _llm_judge_tests(action, ground_truth) * 0.50 if action.has_bug else (0.50 if (exp_testable and exp_missing == ["none"]) else 0.0)
    
    total = min(test_score + missing_score + fix_score, 1.0)
    
    return Reward(
        score=total, breakdown={"testability": test_score, "missing": missing_score, "fix": fix_score},
        feedback=f"Score: {total:.2f}", is_correct=total >= 0.6
    )

# ──────────────────────────────────────────────────────────────────────────────
# Shared LLM judge for advanced tasks (8-15)
# ──────────────────────────────────────────────────────────────────────────────

def _llm_judge_advanced(action: Action, ground_truth: Dict[str, Any], rubric: str) -> float:
    """Generic LLM judge for advanced tasks 8-15."""
    api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    if not api_key or not model:
        return 0.3
    sys_prompt = (
        f"{rubric}\n"
        "Return ONLY JSON: {\"overall\": 0.0-1.0, \"reason\": \"one sentence\"}"
    )
    user_msg = f"Code:\n{ground_truth.get('code', '')}\n\nAgent analysis:\n{action.suggested_fix}"
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
            temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "")
        return max(0.0, min(float(json.loads(raw.strip()).get("overall", 0.3)), 1.0))
    except Exception:
        return 0.3


# ── Task 8: Refactoring ─────────────────────────────────────────────────────

def grade_task8(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    has_opps = ground_truth.get("has_refactoring_opportunities", True)
    det = 0.25 if action.has_bug == has_opps else 0.0
    fix = _llm_judge_advanced(action, ground_truth,
        "Evaluate refactoring analysis. Score: found real opportunities (0-1), "
        "specific locations cited (0-1), concrete suggestions given (0-1), overall (0-1).") * 0.75
    total = min(det + fix, 1.0)
    return Reward(score=total, breakdown={"detection": det, "analysis": fix},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 9: SOLID ───────────────────────────────────────────────────────────

def grade_task9(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    violations = ground_truth.get("solid_violations", ["none"])
    has_v = violations != ["none"]
    det = 0.30 if action.has_bug == has_v else 0.0
    # Keyword match for principles mentioned
    fix_text = action.suggested_fix.lower()
    principles = ["srp", "ocp", "lsp", "isp", "dip", "single responsibility",
                  "open/closed", "liskov", "interface segregation", "dependency inversion"]
    keyword_score = min(sum(1 for p in principles if p in fix_text) / max(len(violations), 1), 1.0) * 0.30
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate SOLID principles analysis. Score: correctly identified violated principles (0-1), "
        "named affected components (0-1), explained why violated (0-1), overall (0-1).") * 0.40
    total = min(det + keyword_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "keywords": keyword_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 10: Error Handling ─────────────────────────────────────────────────

def grade_task10(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    poor_handling = ground_truth.get("error_handling_quality", "good") == "poor"
    det = 0.25 if action.has_bug == poor_handling else 0.0
    fix_text = action.suggested_fix.lower()
    terms = ["try", "except", "catch", "logging", "silent", "bare except", "exception"]
    term_score = min(sum(1 for t in terms if t in fix_text) / 3, 1.0) * 0.25
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate error handling review. Score: identified all missing try/except (0-1), "
        "flagged bare excepts (0-1), suggested specific fixes with correct exception types (0-1), overall (0-1).") * 0.50
    total = min(det + term_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "terms": term_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 11: Documentation ──────────────────────────────────────────────────

def grade_task11(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    missing_docs = not ground_truth.get("has_docstrings", True)
    det = 0.25 if action.has_bug == missing_docs else 0.0
    fix_text = action.suggested_fix.lower()
    terms = ["docstring", "param", "return", "type hint", "args", "raises", ":param", "->"]
    term_score = min(sum(1 for t in terms if t in fix_text) / 3, 1.0) * 0.25
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate documentation review. Score: identified all missing docstrings (0-1), "
        "noted missing type hints (0-1), provided a concrete improved docstring example (0-1), overall (0-1).") * 0.50
    total = min(det + term_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "terms": term_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 12: Concurrency ────────────────────────────────────────────────────

def grade_task12(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    has_issues = ground_truth.get("has_concurrency_issues", False)
    det = 0.30 if action.has_bug == has_issues else 0.0
    fix_text = action.suggested_fix.lower()
    terms = ["race condition", "lock", "mutex", "asyncio", "thread", "blocking", "deadlock", "semaphore"]
    term_score = min(sum(1 for t in terms if t in fix_text) / 2, 1.0) * 0.20
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate concurrency analysis. Score: correctly identified race conditions/blocking calls (0-1), "
        "explained the risk clearly (0-1), provided correct fix using locks/asyncio (0-1), overall (0-1).") * 0.50
    total = min(det + term_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "terms": term_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 13: API Design ─────────────────────────────────────────────────────

def grade_task13(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    api_issues = ground_truth.get("api_design_issues", ["none"])
    has_issues = api_issues != ["none"]
    det = 0.25 if action.has_bug == has_issues else 0.0
    fix_text = action.suggested_fix.lower()
    terms = ["parameter", "naming", "validation", "return type", "rest", "snake_case", "camelcase", "endpoint"]
    term_score = min(sum(1 for t in terms if t in fix_text) / 3, 1.0) * 0.25
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate API design review. Score: identified naming/param issues (0-1), "
        "flagged missing validations (0-1), provided improved signature (0-1), overall (0-1).") * 0.50
    total = min(det + term_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "terms": term_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 14: Code Comparison ────────────────────────────────────────────────

def grade_task14(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    v2_has_bugs = ground_truth.get("v2_introduces_bugs", False)
    det = 0.25 if action.has_bug == v2_has_bugs else 0.0
    fix_text = action.suggested_fix.lower()
    terms = ["improvement", "better", "worse", "introduced", "regression", "verdict", "complete", "partial"]
    term_score = min(sum(1 for t in terms if t in fix_text) / 3, 1.0) * 0.25
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate code comparison review. Score: correctly judged if v2 is improvement (0-1), "
        "identified any new bugs introduced (0-1), gave clear verdict with reasoning (0-1), overall (0-1).") * 0.50
    total = min(det + term_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "terms": term_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


# ── Task 15: Dependencies ───────────────────────────────────────────────────

def grade_task15(action: Action, ground_truth: Dict[str, Any]) -> Reward:
    dep_issues = ground_truth.get("dependency_issues", ["none"])
    has_issues = dep_issues != ["none"]
    det = 0.25 if action.has_bug == has_issues else 0.0
    fix_text = action.suggested_fix.lower()
    matched = sum(1 for issue in dep_issues if issue != "none" and issue.lower() in fix_text)
    issue_score = min(matched / max(len([i for i in dep_issues if i != "none"]), 1), 1.0) * 0.35
    llm = _llm_judge_advanced(action, ground_truth,
        "Evaluate dependency review. Score: found all unused imports (0-1), "
        "flagged risky/deprecated packages (0-1), suggested cleaner alternatives (0-1), overall (0-1).") * 0.40
    total = min(det + issue_score + llm, 1.0)
    return Reward(score=total, breakdown={"detection": det, "issues": issue_score, "llm": llm},
                  feedback=f"Score: {total:.2f}", is_correct=total >= 0.6)


def get_grader(task_level: int) -> Callable[[Action, Dict[str, Any]], Reward]:
    """Return the grading function for the given task level (1-15)."""
    graders = {
        1: grade_task1, 2: grade_task2, 3: grade_task3,
        4: grade_task4, 5: grade_task5, 6: grade_task6, 7: grade_task7,
        8: grade_task8, 9: grade_task9, 10: grade_task10, 11: grade_task11,
        12: grade_task12, 13: grade_task13, 14: grade_task14, 15: grade_task15,
    }
    if task_level not in graders:
        raise ValueError(f"No grader for task level {task_level}")
        
    base_grader = graders[task_level]
    
    def clamped_grader(action: Action, ground_truth: Dict[str, Any]) -> Reward:
        reward = base_grader(action, ground_truth)
        # Deep valuation explicitly fails if score == 0.0 or score == 1.0
        # Clamp strictly between 0 and 1
        reward.score = max(0.01, min(0.99, float(reward.score)))
        return reward
        
    return clamped_grader

