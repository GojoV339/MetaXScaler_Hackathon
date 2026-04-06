# env/models.py
"""
Pydantic v2 models for CodeReview-Env.
Defines all data structures used across the RL environment.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List, Dict, Any

# Type aliases
BugType = Literal[
    "logic_error",
    "security_vulnerability",
    "performance_issue",
    "syntax_error",
    "no_bug",
]
Severity = Literal["low", "medium", "high", "critical", "none"]
TaskLevel = Literal[1, 2, 3]


class Observation(BaseModel):
    """What the agent sees each step — the code snippet and task context."""

    model_config = ConfigDict(frozen=False)

    snippet_id: str = Field(..., description="Unique identifier for the code snippet")
    code: str = Field(..., description="The code snippet to review")
    language: str = Field(
        ..., description="Programming language of the snippet (python / javascript)"
    )
    task_level: TaskLevel = Field(
        ..., description="Current task level: 1=Easy, 2=Medium, 3=Hard"
    )
    step_number: int = Field(
        ..., ge=0, description="Current step number in the episode"
    )
    context_hint: Optional[str] = Field(
        default=None,
        description="Brief hint for harder tasks; None for easy tasks",
    )


class Action(BaseModel):
    """What the agent returns — its code review decision."""

    model_config = ConfigDict(frozen=False)

    has_bug: bool = Field(
        ..., description="Whether the agent believes the snippet contains a bug"
    )
    bug_type: BugType = Field(
        default="no_bug",
        description="Classification of the bug type; 'no_bug' if has_bug is False",
    )
    severity: Severity = Field(
        default="none",
        description="Assessed severity of the bug; 'none' if has_bug is False",
    )
    suggested_fix: str = Field(
        default="",
        description="Suggested fix for the bug; required for task_level 3, empty for levels 1-2",
    )


class Reward(BaseModel):
    """Scoring output — how well the agent performed on this step."""

    model_config = ConfigDict(frozen=False)

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall score between 0.0 and 1.0"
    )
    breakdown: Dict[str, float] = Field(
        ...,
        description="Score breakdown by component, e.g. {'detection': 0.4, 'classification': 0.3}",
    )
    feedback: str = Field(
        ..., description="Human-readable explanation of the score"
    )
    is_correct: bool = Field(
        ..., description="Whether the agent's answer is considered correct"
    )


class State(BaseModel):
    """Full environment state — for inspection and debugging."""

    model_config = ConfigDict(frozen=False)

    current_snippet: Optional[Dict[str, Any]] = Field(
        default=None, description="The current code snippet being reviewed"
    )
    action_history: List[Action] = Field(
        default_factory=list, description="History of all actions taken in this episode"
    )
    reward_history: List[float] = Field(
        default_factory=list, description="History of all scores received in this episode"
    )
    cumulative_score: float = Field(
        default=0.0, description="Sum of all reward scores in this episode"
    )
    episode_done: bool = Field(
        default=False, description="Whether the current episode has ended"
    )
    task_level: TaskLevel = Field(
        default=1, description="Current task level"
    )


class TaskConfig(BaseModel):
    """Task definition — describes one of the three task levels."""

    model_config = ConfigDict(frozen=False)

    task_id: str = Field(..., description="Unique task identifier")
    level: TaskLevel = Field(..., description="Task level: 1, 2, or 3")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Detailed description of the task")
    action_space: Dict[str, str] = Field(
        ..., description="Describes what fields matter for this task and their weights"
    )
    max_score: float = Field(
        default=1.0, description="Maximum achievable score for this task"
    )


class StepResponse(BaseModel):
    """HTTP response payload from the /step endpoint."""

    model_config = ConfigDict(frozen=False)

    observation: Optional[Observation] = Field(default=None, description="Next observation for the agent; None when episode is done")
    reward: Reward = Field(..., description="Reward for the action taken")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the step"
    )
