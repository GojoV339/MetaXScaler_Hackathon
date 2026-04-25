# env/pr_environment.py
"""
PRReviewEnv — Multi-step Pull Request Review Environment.

Episode structure:
  reset() → loads a PR → returns first file observation
  step(action) × N → grades each file → returns next observation
  step(action) on final_verdict → grades verdict → done=True

Reward formula:
  episode_reward = 0.6 × mean(per_file_scores) + 0.4 × verdict_score
"""
from env.models import (
    PRObservation, PRAction, PRReward, PRState,
    FileObservation,
)
from env.pr_graders import grade_file_step, grade_comparison_step, grade_verdict_step
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional


class PRReviewEnv:
    """Multi-step RL environment for Pull Request code review."""

    def __init__(
        self,
        prs_path: Optional[str] = None,
        snippets_path: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty_filter: Optional[str] = None,
    ):
        if seed is not None:
            random.seed(seed)

        base = Path(__file__).parent

        # Load PRs
        pr_file = prs_path or base / "data" / "prs.json"
        all_prs = json.load(open(pr_file))

        # Curriculum support: filter by difficulty
        if difficulty_filter:
            self._prs = [p for p in all_prs if p.get("difficulty") == difficulty_filter]
            if not self._prs:
                self._prs = all_prs
        else:
            self._prs = all_prs

        # Load snippets as lookup dict
        snip_file = snippets_path or base / "data" / "snippets.json"
        snips = json.load(open(snip_file))
        self._snippets: Dict[str, dict] = {s["id"]: s for s in snips}

        # Episode state
        self._current_pr: Optional[dict] = None
        self._file_index: int = 0
        self._per_file_scores: List[float] = []
        self._findings: List[str] = []
        self._step_number: int = 0
        self._episode_done: bool = False

    def reset(self) -> PRObservation:
        """Start a new PR review episode."""
        self._current_pr = random.choice(self._prs)
        self._file_index = 0
        self._per_file_scores = []
        self._findings = []
        self._step_number = 0
        self._episode_done = False
        return self._build_observation()

    def step(self, action: PRAction) -> Tuple[Optional[PRObservation], PRReward, bool, Dict[str, Any]]:
        """Process one action. Returns (next_obs, reward, done, info)."""
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_number += 1
        pr = self._current_pr
        total_files = len(pr["files"])

        # Final verdict step
        if self._file_index >= total_files:
            reward = grade_verdict_step(action, pr, self._per_file_scores)
            self._episode_done = True

            file_mean = (
                sum(self._per_file_scores) / len(self._per_file_scores)
                if self._per_file_scores else 0.0
            )
            episode_score = round(file_mean * 0.6 + reward.score * 0.4, 4)

            info = {
                "pr_id": pr["pr_id"],
                "step": self._step_number,
                "step_type": "final_verdict",
                "episode_score": episode_score,
                "per_file_scores": self._per_file_scores,
                "verdict": action.verdict,
                "correct_verdict": pr["correct_verdict"],
                "verdict_correct": action.verdict == pr["correct_verdict"],
            }
            return None, reward, True, info

        # File review step
        file_meta = pr["files"][self._file_index]
        snippet = self._snippets.get(file_meta["snippet_id"], {})
        ground_truth = {**snippet, "task_type": file_meta.get("task_type", 1)}

        is_comp = file_meta.get("is_comparison", False)
        if is_comp:
            snip_v2 = self._snippets.get(file_meta.get("snippet_id_v2", ""), {})
            ground_truth["code_v2"] = snip_v2.get("code", "")
            ground_truth["better_version"] = file_meta.get("better_version", "v2")
            ground_truth["comparison_reason"] = file_meta.get("comparison_reason", "")
            reward = grade_comparison_step(action, ground_truth)
        else:
            reward = grade_file_step(action, ground_truth)

        self._per_file_scores.append(reward.score)
        finding = f"{file_meta['file_name']}: {reward.feedback[:80]}"
        self._findings.append(finding)
        self._file_index += 1

        # Build next observation
        if self._file_index >= total_files:
            next_obs = self._build_verdict_observation()
        else:
            next_obs = self._build_observation()

        info = {
            "pr_id": pr["pr_id"],
            "step": self._step_number,
            "step_type": "comparison" if is_comp else "file_review",
            "file_name": file_meta["file_name"],
            "file_score": reward.score,
        }
        return next_obs, reward, False, info

    def state(self) -> PRState:
        """Return full current state of the episode."""
        pr = self._current_pr or {}
        total = len(pr.get("files", []))
        return PRState(
            pr_id=pr.get("pr_id", ""),
            pr_title=pr.get("title", ""),
            files_reviewed=self._file_index,
            total_files=total,
            per_file_scores=self._per_file_scores,
            cumulative_score=round(sum(self._per_file_scores), 4),
            episode_done=self._episode_done,
            findings=self._findings,
            current_step_type=(
                "final_verdict" if self._file_index >= total else "file_review"
            ),
        )

    def _build_observation(self) -> PRObservation:
        pr = self._current_pr
        file_meta = pr["files"][self._file_index]
        snippet = self._snippets.get(file_meta["snippet_id"], {})

        is_comp = file_meta.get("is_comparison", False)
        code_v2 = None
        if is_comp:
            snip_v2 = self._snippets.get(file_meta.get("snippet_id_v2", ""), {})
            code_v2 = snip_v2.get("code", "")

        return PRObservation(
            pr_id=pr["pr_id"],
            pr_title=pr["title"],
            pr_description=pr["description"],
            current_file=FileObservation(
                file_name=file_meta["file_name"],
                code=snippet.get("code", ""),
                language=snippet.get("language", "python"),
                task_type=file_meta.get("task_type", 1),
                is_comparison=is_comp,
                code_v2=code_v2,
            ),
            files_reviewed=self._file_index,
            total_files=len(pr["files"]),
            step_type="comparison" if is_comp else "file_review",
            previous_findings=list(self._findings),
            step_number=self._step_number,
        )

    def _build_verdict_observation(self) -> PRObservation:
        pr = self._current_pr
        return PRObservation(
            pr_id=pr["pr_id"],
            pr_title=pr["title"],
            pr_description=pr["description"],
            current_file=FileObservation(
                file_name="PR Summary",
                code="",
                language="",
                task_type=0,
            ),
            files_reviewed=len(pr["files"]),
            total_files=len(pr["files"]),
            step_type="final_verdict",
            previous_findings=list(self._findings),
            step_number=self._step_number,
        )
