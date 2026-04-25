# app.py
"""
FastAPI server for CodeReview-Env.
Exposes the RL environment as HTTP endpoints for OpenEnv evaluation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
from env import CodeReviewEnv

# Load .env file
load_dotenv()
from env.models import Action, Observation, State, StepResponse, Reward
from env.models import PRAction, PRObservation, PRStepResponse, PRState
from env.tasks import list_all_tasks, get_task
from env.pr_environment import PRReviewEnv
from typing import Dict, Any, Optional
from pydantic import BaseModel


# ──────────────────────────────────────────────────────────────────────────────
# App initialization
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CodeReview-Env",
    description="OpenEnv-compliant RL environment for AI code review",
    version="1.0.0",
)

# CORS middleware (allow all origins for HF evaluation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env: Optional[CodeReviewEnv] = None
pr_env: Optional[PRReviewEnv] = None


# ──────────────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_level: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment with the given task level.

    Creates a new CodeReviewEnv instance and returns the first observation.
    """
    global env
    try:
        task_level = request.task_level
        if task_level not in range(1, 16):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_level: {task_level}. Must be between 1 and 15.",
            )
        env = CodeReviewEnv(task_level=task_level)
        observation = env.reset()
        return observation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    """Submit an action and receive reward + next observation.

    The action is the agent's code review decision.
    """
    global env
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )

    try:
        state = env.state()
        if state.episode_done:
            raise HTTPException(
                status_code=400,
                detail="Episode done. Call /reset first.",
            )

        observation, reward, done, info = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=State)
async def get_state():
    """Return the current environment state."""
    global env
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    """Return all available task configurations."""
    tasks = list_all_tasks()
    return [t.model_dump() for t in tasks]


@app.get("/health")
async def health():
    """Health check endpoint — pinged by HF validator."""
    return {
        "status": "ok",
        "environment": "CodeReview-Env",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint — environment info and available endpoints."""
    return {
        "name": "CodeReview-Env",
        "description": "RL environment for AI code review",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health",
                     "/pr/reset", "/pr/step", "/pr/state", "/pr/info"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# PR Pipeline Endpoints (additive — existing endpoints unchanged)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/pr/reset")
async def pr_reset(body: dict = {}):
    """Start a new PR review episode."""
    global pr_env
    try:
        difficulty = body.get("difficulty", None)
        pr_env = PRReviewEnv(difficulty_filter=difficulty)
        obs = pr_env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pr/step")
async def pr_step(action: PRAction):
    """Submit action for current step in PR review."""
    global pr_env
    if pr_env is None:
        raise HTTPException(status_code=400, detail="PR environment not initialized. Call /pr/reset first.")
    try:
        obs, reward, done, info = pr_env.step(action)
        return PRStepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pr/state")
async def pr_state():
    """Get current PR review episode state."""
    global pr_env
    if pr_env is None:
        raise HTTPException(status_code=400, detail="PR environment not initialized. Call /pr/reset first.")
    return pr_env.state()


@app.get("/pr/info")
async def pr_info():
    """Metadata about the PR pipeline."""
    return {
        "name": "PR Review Pipeline",
        "episode_structure": "N file_review steps + 1 final_verdict step",
        "reward_formula": "episode = 0.6 * mean(per_file) + 0.4 * verdict_score",
        "verdicts": ["APPROVE", "REQUEST_CHANGES", "REJECT"],
        "anti_hacking": ["always-approve detector", "trivial-fix penalty", "copy-paste penalty"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
