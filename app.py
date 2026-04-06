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
from env.tasks import list_all_tasks, get_task
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
        if task_level not in range(1, 8):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_level: {task_level}. Must be between 1 and 7.",
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
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
