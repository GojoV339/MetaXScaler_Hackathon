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
_TASKS = {1: TASK_1, 2: TASK_2, 3: TASK_3}


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
    """Return all three task configurations in order.

    Returns:
        List of [Task1, Task2, Task3].
    """
    return [TASK_1, TASK_2, TASK_3]


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
