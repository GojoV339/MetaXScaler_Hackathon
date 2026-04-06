---
title: CodeReview-Env
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - code-review
license: mit
---

# CodeReview-Env

> An OpenEnv-compliant reinforcement learning environment for AI code review agents.

## Environment Description

CodeReview-Env simulates a real-world code review workflow. An AI agent is given
code snippets and must act as a senior software engineer — detecting bugs,
classifying their type, assessing severity, and suggesting specific fixes.

This environment is built to train and evaluate agents on a task that software
engineers perform hundreds of times per week, making it directly applicable to
real-world software quality automation.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| snippet_id | string | Unique identifier for the code snippet |
| code | string | The code to review |
| language | string | Programming language (python / javascript) |
| task_level | int | 1=Easy, 2=Medium, 3=Hard |
| step_number | int | Current step in the episode |
| context_hint | string? | Optional hint for harder tasks |

## Action Space

| Field | Type | Required For |
|-------|------|-------------|
| has_bug | bool | All tasks |
| bug_type | enum | Tasks 2 & 3 |
| severity | enum | Task 3 only |
| suggested_fix | string | Task 3 only |

## Task Descriptions

### Task 1 — Bug Detection (Easy)
Detect whether a code snippet contains a bug. Binary yes/no decision.
Scoring: 1.0 if correct, 0.0 if wrong.

### Task 2 — Bug Classification (Medium)
Identify the bug AND classify its type (logic_error, security_vulnerability,
performance_issue, syntax_error, no_bug).
Scoring: +0.4 for correct detection, +0.6 for correct classification.

### Task 3 — Full Code Review (Hard)
Complete review: detection + classification + severity + specific fix.
Scoring: 0.15 detection + 0.15 type + 0.20 severity + 0.50 fix quality (LLM-judged).

## Reward Design

- **Deterministic scoring** for Tasks 1-2 (same input = same score)
- **LLM-as-judge** for Task 3 fix quality evaluation
- **Anti-hack penalties**: -0.1 for vague/low-effort fixes (<10 chars, generic phrases like "fix the bug", TODO placeholders)
- **Partial credit**: agents get proportional scores for partially correct answers

## Baseline Scores

Scores produced by running `python inference.py` with `Qwen/Qwen2.5-Coder-32B-Instruct`:

| Task | Name | Mean Score | Min | Max |
|------|------|-----------|-----|-----|
| 1 | Bug Detection (Easy) | 0.400 | — | — |
| 2 | Bug Classification (Medium) | 0.880 | — | — |
| 3 | Full Code Review (Hard) | 0.500 | — | — |

**Overall Mean Score: 0.593**

To reproduce:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_token"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
python inference.py
```

## Setup & Usage

### Environment Variables
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token-here"
```

### Docker
```bash
docker build -t codereview-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  codereview-env
```

### Run Inference
```bash
python inference.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (`{"task_level": 1}`) |
| `/step` | POST | Submit action, receive reward |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all 3 tasks |
| `/health` | GET | Health check |

## Project Structure

```
.
├── inference.py              ← Agent runner (MANDATORY in root)
├── app.py                    ← FastAPI server
├── openenv.yaml              ← OpenEnv spec metadata
├── Dockerfile                ← Containerization
├── requirements.txt
├── README.md
└── env/
    ├── __init__.py
    ├── environment.py        ← Core RL environment
    ├── models.py             ← Pydantic models
    ├── tasks.py              ← Task definitions
    ├── graders.py            ← Scoring logic
    └── data/
        └── snippets.json     ← 30 labeled code snippets
```

## Author

Palle Venkata Dharaneswara Reddy
