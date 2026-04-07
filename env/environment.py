# env/environment.py
"""
Core RL environment for CodeReview-Env.
OpenEnv-compliant: supports reset(), step(), and state() interface.
"""

from env.models import Observation, Action, Reward, State, TaskLevel
from env.tasks import get_task, TaskConfig
from env.graders import get_grader
import json
import random
import logging
from typing import Tuple, Dict, Any, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeReviewEnv:
    """
    OpenEnv-compliant RL environment for code review.

    An AI agent learns to review code snippets by detecting bugs,
    classifying them, and suggesting fixes across three difficulty levels.
    """

    MAX_STEPS: int = 10

    def __init__(
        self,
        task_level: TaskLevel = 1,
        snippets_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the CodeReview environment.

        Args:
            task_level: Difficulty level (1=Easy, 2=Medium, 3=Hard).
            snippets_path: Path to the snippets JSON file.
                           Defaults to env/data/snippets.json.
            seed: Optional random seed for reproducibility.
        """
        self.task_level: TaskLevel = task_level

        # Load and filter snippets
        path = snippets_path or str(
            Path(__file__).parent / "data" / "snippets.json"
        )
        all_snippets = self._load_snippets(path)
        self._snippets: List[Dict[str, Any]] = [
            s for s in all_snippets if task_level in s.get("task_levels", list(range(1, 16)))
        ]
        if not self._snippets:
            logger.warning(
                "No snippets found for task_level=%d; using all snippets.", task_level
            )
            self._snippets = all_snippets

        # Random seed
        if seed is not None:
            random.seed(seed)

        # Internal state
        self._state: Optional[State] = None
        self._current_snippet: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._action_history: List[Action] = []
        self._reward_history: List[float] = []
        self._episode_done: bool = True

        # Task configuration and grader
        self._task_config: TaskConfig = get_task(task_level)
        self._grader: Callable[[Action, Dict[str, Any]], Reward] = get_grader(
            task_level
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and start a new episode.

        Returns:
            An Observation containing the new code snippet to review.
        """
        self._current_snippet = random.choice(self._snippets)
        self._step_count = 0
        self._action_history = []
        self._reward_history = []
        self._episode_done = False

        return Observation(
            snippet_id=self._current_snippet["id"],
            code=self._current_snippet["code"],
            language=self._current_snippet["language"],
            task_level=self.task_level,
            step_number=0,
            context_hint=self._build_context_hint(),
        )

    def step(
        self, action: Action
    ) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The agent's code review action.

        Returns:
            Tuple of (observation, reward, done, info).

        Raises:
            RuntimeError: If the episode is already done.
        """
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1

        # Grade the action
        reward = self._grader(action, self._current_snippet)

        # Record history
        self._action_history.append(action)
        self._reward_history.append(reward.score)

        # Episode ends after one step (one snippet per episode)
        done = True
        self._episode_done = True

        info: Dict[str, Any] = {
            "step": self._step_count,
            "snippet_id": self._current_snippet["id"],
            "task_level": self.task_level,
            "difficulty": self._current_snippet["difficulty"],
        }

        return None, reward, done, info

    def state(self) -> State:
        """Return the full internal state for inspection.

        Returns:
            A State object with current environment status.
        """
        return State(
            current_snippet=self._current_snippet,
            action_history=self._action_history,
            reward_history=self._reward_history,
            cumulative_score=sum(self._reward_history),
            episode_done=self._episode_done,
            task_level=self.task_level,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_snippets(self, path: str) -> List[Dict[str, Any]]:
        """Load and validate code snippets from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            List of valid snippet dicts.
        """
        required_fields = {
            "id",
            "language",
            "code",
            "has_bug",
            "bug_type",
            "severity",
            "fix",
            "difficulty",
        }

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Failed to load snippets from %s: %s", path, e)
            raise

        valid: List[Dict[str, Any]] = []
        for i, entry in enumerate(raw):
            missing = required_fields - set(entry.keys())
            if missing:
                logger.warning(
                    "Snippet at index %d is missing fields %s — skipping.", i, missing
                )
                continue
            valid.append(entry)

        logger.info("Loaded %d valid snippets from %s.", len(valid), path)
        return valid

    def _build_context_hint(self) -> Optional[str]:
        """Build an optional context hint based on the current task level.

        Returns:
            A hint string for levels 2-7, or None for level 1.
        """
        hints = {
            1: None,
            2: "Focus on identifying the specific type of bug if one exists.",
            3: "Provide a specific, implementable fix. Vague suggestions score 0.",
            4: "Look for code quality issues: magic numbers, dead code, god functions, duplicate logic, poor naming. List ALL smells you find in your suggested_fix.",
            5: "Perform an OWASP security audit. Check for: SQL injection, XSS, hardcoded secrets, path traversal. State the vulnerability type and provide a specific secure fix.",
            6: "Analyze time complexity. State the current Big-O notation and identify the specific bottleneck. Suggest a faster algorithm with its Big-O in your fix.",
            7: "Assess testability and list missing tests. In suggested_fix, provide 2-3 specific test cases with: input values, expected output, and what edge case they cover.",
            8: "Identify refactoring opportunities. In suggested_fix, return JSON: [{refactor_type, location, issue, suggested_change}]. Be specific about line numbers or code patterns.",
            9: "Detect SOLID principle violations. In suggested_fix, return JSON: [{violated_principle, component, reason}]. Name the exact class/function and which of SRP/OCP/LSP/ISP/DIP is violated.",
            10: "Review error handling quality. In suggested_fix, return JSON: {error_handling_score: 0-10, issues: [], fix_suggestions: []}. Check for missing try/except, bare except, silent failures.",
            11: "Review documentation quality. In suggested_fix, return JSON: {doc_quality_score: 0-10, missing: [], example_improvement: 'full docstring example'}. Check docstrings, type hints, param docs.",
            12: "Detect concurrency issues. In suggested_fix, return JSON: [{issue_type, affected_code, risk, fix}]. Check for race conditions, blocking async calls, missing locks.",
            13: "Review API design. In suggested_fix, return JSON: {api_score: 0-10, issues: [], improved_signature: 'fixed function signature'}. Check naming, params count, validation, return types.",
            14: "Compare v1 vs v2 (both in the code block). In suggested_fix, return JSON: {improvement: 'yes/no', new_issues: [], verdict: 'one sentence', reason: 'explanation'}.",
            15: "Review imports and dependencies. In suggested_fix, return JSON: {unused_imports: [], risky_dependencies: [], cleaner_imports: []}. Flag unused, risky (pickle/eval), and over-imported items.",
        }
        return hints.get(self.task_level)
