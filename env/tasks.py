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

TASK_8 = TaskConfig(
    task_id="refactoring_detection",
    level=8,
    name="Refactoring Opportunity Detection",
    description=(
        "Identify specific refactoring opportunities in the code. "
        "Look for: extract_method (long functions), replace_conditional_with_polymorphism, "
        "introduce_parameter_object (too many params), remove_duplicate_code, "
        "decompose_conditional (complex if/else). "
        "In suggested_fix, list each opportunity with: refactor_type, exact location, and suggested change."
    ),
    action_space={
        "has_bug": "bool — does the code have refactoring opportunities?",
        "bug_type": "use 'logic_error' for structural issues",
        "severity": "low/medium/high based on impact",
        "suggested_fix": "JSON list of {refactor_type, location, issue, suggested_change}",
    },
    max_score=1.0,
)

TASK_9 = TaskConfig(
    task_id="solid_violations",
    level=9,
    name="SOLID Principles Violation Detection",
    description=(
        "Detect violations of SOLID principles: "
        "SRP (Single Responsibility — class does too much), "
        "OCP (Open/Closed — not extensible without modification), "
        "LSP (Liskov Substitution — subclass breaks parent contract), "
        "ISP (Interface Segregation — fat interfaces), "
        "DIP (Dependency Inversion — depends on concretions not abstractions). "
        "In suggested_fix, list each violation: violated_principle, component, reason."
    ),
    action_space={
        "has_bug": "bool — are there SOLID violations?",
        "bug_type": "use 'logic_error' for design violations",
        "severity": "high for SRP/DIP, medium for OCP/ISP/LSP",
        "suggested_fix": "JSON list of {violated_principle, component, reason}",
    },
    max_score=1.0,
)

TASK_10 = TaskConfig(
    task_id="error_handling_review",
    level=10,
    name="Error Handling Review",
    description=(
        "Evaluate error handling quality. Check for: "
        "missing try/except around risky operations (I/O, network, parsing), "
        "bare except clauses catching all exceptions, "
        "wrong exception types used, "
        "no logging on failures, "
        "silent failures that swallow errors. "
        "In suggested_fix: provide error_handling_score (0-10), list issues, and fix_suggestions."
    ),
    action_space={
        "has_bug": "bool — is error handling inadequate?",
        "bug_type": "use 'logic_error' for error handling gaps",
        "severity": "high for silent failures, medium for bare except",
        "suggested_fix": "JSON with {error_handling_score, issues: [], fix_suggestions: []}",
    },
    max_score=1.0,
)

TASK_11 = TaskConfig(
    task_id="documentation_review",
    level=11,
    name="Documentation Quality Review",
    description=(
        "Evaluate documentation completeness. Check for: "
        "missing docstrings on public functions/classes, "
        "missing parameter descriptions, "
        "missing return type descriptions, "
        "absent type hints, "
        "misleading or outdated comments. "
        "In suggested_fix: provide doc_quality_score (0-10), list missing items, example improvement."
    ),
    action_space={
        "has_bug": "bool — is documentation inadequate?",
        "bug_type": "use 'logic_error' for documentation gaps",
        "severity": "medium for missing docstrings, low for style issues",
        "suggested_fix": "JSON with {doc_quality_score, missing: [], example_improvement: str}",
    },
    max_score=1.0,
)

TASK_12 = TaskConfig(
    task_id="concurrency_review",
    level=12,
    name="Concurrency & Race Condition Detection",
    description=(
        "Detect concurrency and async issues: "
        "shared mutable state without locks, "
        "missing asyncio locks or semaphores, "
        "blocking calls inside async functions (time.sleep instead of asyncio.sleep), "
        "potential deadlocks from lock ordering, "
        "thread-unsafe operations on shared collections. "
        "In suggested_fix: list each issue with issue_type, affected_code, risk, fix."
    ),
    action_space={
        "has_bug": "bool — are there concurrency issues?",
        "bug_type": "use 'performance_issue' for async, 'logic_error' for race conditions",
        "severity": "critical for race conditions, high for blocking async",
        "suggested_fix": "JSON list of {issue_type, affected_code, risk, fix}",
    },
    max_score=1.0,
)

TASK_13 = TaskConfig(
    task_id="api_design_review",
    level=13,
    name="API Design Review",
    description=(
        "Evaluate API/function design quality: "
        "naming consistency (snake_case vs camelCase), "
        "too many parameters (>5 is a smell), "
        "inconsistent return types, "
        "missing input validation, "
        "poor REST design (wrong HTTP methods, non-RESTful naming). "
        "In suggested_fix: provide api_score (0-10), list issues, improved_signature."
    ),
    action_space={
        "has_bug": "bool — does the API have design issues?",
        "bug_type": "use 'logic_error' for API design problems",
        "severity": "high for validation gaps, medium for naming/params",
        "suggested_fix": "JSON with {api_score, issues: [], improved_signature: str}",
    },
    max_score=1.0,
)

TASK_14 = TaskConfig(
    task_id="code_comparison",
    level=14,
    name="Code Comparison Review",
    description=(
        "Compare two versions of code (v1 vs v2 provided in context). "
        "Determine: is v2 an improvement over v1? "
        "Did the refactor introduce new bugs? "
        "Was the fix complete or partial? "
        "In suggested_fix: provide {improvement: yes/no, new_issues: [], verdict: str, reason: str}."
    ),
    action_space={
        "has_bug": "bool — did v2 introduce new bugs compared to v1?",
        "bug_type": "use appropriate type if new bug introduced",
        "severity": "severity of any new bugs introduced",
        "suggested_fix": "JSON with {improvement, new_issues: [], verdict: str, reason: str}",
    },
    max_score=1.0,
)

TASK_15 = TaskConfig(
    task_id="dependency_review",
    level=15,
    name="Dependency & Import Review",
    description=(
        "Analyze import and dependency quality: "
        "unused imports that add dead weight, "
        "risky/deprecated packages (e.g. pickle for untrusted data, exec/eval usage), "
        "over-importing (importing entire modules when only 1 function needed), "
        "missing __all__ for public API control. "
        "In suggested_fix: provide {unused_imports: [], risky_dependencies: [], cleaner_imports: []}."
    ),
    action_space={
        "has_bug": "bool — are there import/dependency issues?",
        "bug_type": "use 'security_vulnerability' for risky deps, 'logic_error' for unused",
        "severity": "high for security risks, low for unused imports",
        "suggested_fix": "JSON with {unused_imports: [], risky_dependencies: [], cleaner_imports: []}",
    },
    max_score=1.0,
)

_TASKS = {
    1: TASK_1, 2: TASK_2, 3: TASK_3, 4: TASK_4, 5: TASK_5,
    6: TASK_6, 7: TASK_7, 8: TASK_8, 9: TASK_9, 10: TASK_10,
    11: TASK_11, 12: TASK_12, 13: TASK_13, 14: TASK_14, 15: TASK_15,
}


def get_task(level: TaskLevel) -> TaskConfig:
    """Return the task configuration for the given level."""
    if level not in _TASKS:
        raise ValueError(f"Invalid task level: {level}. Must be one of {list(_TASKS.keys())}.")
    return _TASKS[level]


def list_all_tasks() -> List[TaskConfig]:
    """Return all task configurations in order."""
    return [_TASKS[i] for i in sorted(_TASKS.keys())]


def get_task_description_for_prompt(level: TaskLevel) -> str:
    """Return a prompt-ready description for the given task level."""
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

