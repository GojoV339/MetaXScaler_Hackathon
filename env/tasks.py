# env/tasks.py
"""
Task definitions for CodeReview-Env.
Defines 3 progressive task levels for the code review RL environment.
"""

from env.models import TaskConfig, TaskLevel
from typing import List


# ---------------------------------------------------------------------------
# Task 1 — Bug Detection (Easy)
# ---------------------------------------------------------------------------
TASK_1 = TaskConfig(
    task_id="bug_detection",
    level=1,
    name="Bug Detection",
    description=(
        "Given a code snippet, determine whether it contains a bug. "
        "Return has_bug=True or has_bug=False. "
        "Bug type and severity are not evaluated at this level."
    ),
    action_space={"has_bug": "bool — primary evaluation field"},
)

# ---------------------------------------------------------------------------
# Task 2 — Bug Classification (Medium)
# ---------------------------------------------------------------------------
TASK_2 = TaskConfig(
    task_id="bug_classification",
    level=2,
    name="Bug Classification",
    description=(
        "Identify whether the snippet has a bug AND classify its type. "
        "Choose from: logic_error, security_vulnerability, performance_issue, "
        "syntax_error, no_bug. Partial credit awarded for correct detection "
        "even if classification is wrong."
    ),
    action_space={
        "has_bug": "bool — worth 40% of score",
        "bug_type": "BugType — worth 60% of score",
    },
)

# ---------------------------------------------------------------------------
# Task 3 — Full Code Review (Hard)
# ---------------------------------------------------------------------------
TASK_3 = TaskConfig(
    task_id="full_review",
    level=3,
    name="Full Code Review",
    description=(
        "Complete review: detect bug, classify type, assign severity "
        "(low/medium/high/critical), and provide a specific, actionable "
        "fix in plain English. Fix quality is evaluated by an LLM judge "
        "on accuracy, completeness, and specificity."
    ),
    action_space={
        "has_bug": "bool — worth 15% of score",
        "bug_type": "BugType — worth 15% of score",
        "severity": "Severity — worth 20% of score",
        "suggested_fix": "str — worth 50% of score (LLM-judged)",
    },
)

# All tasks indexed by level
TASK_4 = TaskConfig(
    task_id="code_smell_detection",
    level=4,
    name="Code Smell Detection",
    description=(
        "Identify code quality issues in the snippet. Look for: magic numbers "
        "(hardcoded values that should be constants), dead code (unreachable or "
        "unused code), god functions (functions doing too much), duplicate logic "
        "(copy-pasted blocks), poor naming (single letters, misleading names). "
        "Return a list of smells found. Partial credit per smell correctly identified."
    ),
    action_space={
        "has_bug": "bool — is there a code quality issue?",
        "bug_type": "use 'logic_error' for smells",
        "severity": "low/medium/high based on smell severity",
        "suggested_fix": "List each smell and how to fix it specifically",
    },
    max_score=1.0,
)

TASK_5 = TaskConfig(
    task_id="security_audit",
    level=5,
    name="Security Audit",
    description=(
        "Perform a security audit on the code snippet based on OWASP Top 10. "
        "Identify: SQL injection risks (string concatenation in queries), "
        "XSS vulnerabilities (unsanitized user input in HTML), "
        "hardcoded secrets (API keys, passwords in source), "
        "path traversal risks (unvalidated file paths), "
        "insecure deserialization. Rate the severity and provide a specific fix."
    ),
    action_space={
        "has_bug": "bool — is there a security vulnerability?",
        "bug_type": "security_vulnerability if yes",
        "severity": "critical for injection/XSS, high for hardcoded secrets",
        "suggested_fix": "Specific OWASP-compliant fix with code example",
    },
    max_score=1.0,
)

TASK_6 = TaskConfig(
    task_id="performance_optimization",
    level=6,
    name="Performance Optimization",
    description=(
        "Analyze the time and space complexity of the code. "
        "Identify performance bottlenecks: nested loops on large data (O(n²)+), "
        "N+1 database query patterns, redundant computations inside loops, "
        "sorting inside loops, inefficient data structures. "
        "State the current Big-O complexity and suggest a faster algorithm or approach."
    ),
    action_space={
        "has_bug": "bool — is there a performance issue?",
        "bug_type": "performance_issue if yes",
        "severity": "high if O(n²)+, medium if redundant, low if minor",
        "suggested_fix": "State current O() complexity + suggest specific optimized approach with O() improvement",
    },
    max_score=1.0,
)

TASK_7 = TaskConfig(
    task_id="test_coverage_review",
    level=7,
    name="Test Coverage Review",
    description=(
        "Assess whether the code is testable and identify missing test cases. "
        "Check: are there untested edge cases (null/None inputs, empty collections, "
        "boundary values, negative numbers)? Is the code structured for testability "
        "(no hidden side effects, injectable dependencies)? "
        "Suggest 2-3 specific unit tests that should be written."
    ),
    action_space={
        "has_bug": "bool — does code lack adequate test coverage or testability?",
        "bug_type": "logic_error for untestable design",
        "severity": "based on how critical the missing tests are",
        "suggested_fix": "List 2-3 specific test cases: inputs, expected outputs, what they verify",
    },
    max_score=1.0,
)

_TASKS = {1: TASK_1, 2: TASK_2, 3: TASK_3, 4: TASK_4, 5: TASK_5, 6: TASK_6, 7: TASK_7}


def get_task(level: TaskLevel) -> TaskConfig:
    """Return the task configuration for the given level.

    Args:
        level: Task level (1, 2, or 3).

    Returns:
        TaskConfig for the requested level.

    Raises:
        ValueError: If level is not 1, 2, or 3.
    """
    if level not in _TASKS:
        raise ValueError(
            f"Invalid task level: {level}. Must be one of {list(_TASKS.keys())}."
        )
    return _TASKS[level]


def list_all_tasks() -> List[TaskConfig]:
    """Return all task configurations in order.

    Returns:
        List of all TaskConfigs.
    """
    return [TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, TASK_6, TASK_7]


def get_task_description_for_prompt(level: TaskLevel) -> str:
    """Return a prompt-ready description for the given task level.

    This is used by the inference script to build the system prompt
    so the LLM agent understands what is expected.

    Args:
        level: Task level (1, 2, or 3).

    Returns:
        A multi-line string suitable for inclusion in an LLM prompt.
    """
    task = get_task(level)

    action_lines = "\n".join(
        f"  - {field}: {desc}" for field, desc in task.action_space.items()
    )

    return (
        f"Task: {task.name} (Level {task.level})\n"
        f"Description: {task.description}\n"
        f"Action space:\n{action_lines}\n"
        f"Maximum score: {task.max_score}"
    )
