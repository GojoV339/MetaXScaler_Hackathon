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

🚀 Live Demo: https://dharaneswarreddy-codereview-env.hf.space

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
| task_level | int | 1=Bug Detection, 2=Bug Classification, 3=Full Review, 4=Code Smell, 5=Security, 6=Performance, 7=Test Coverage |
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

### Task 4 — Code Smell Detection (Medium)
Agent identifies code quality issues: magic numbers, dead code, god functions,
duplicate logic, poor naming. Partial credit per smell correctly identified.
Scoring: +0.80 distributed across smells found + 0.20 completion bonus.

### Task 5 — Security Audit (Hard)
Agent performs OWASP Top 10 audit: SQL injection, XSS, hardcoded secrets,
path traversal. Fix quality evaluated by LLM judge on security best practices.
Scoring: 0.30 detection + 0.20 type + 0.10 severity + 0.40 fix (LLM-judged).

### Task 6 — Performance Optimization (Hard)
Agent identifies Big-O complexity issues and bottlenecks. Must state current
O() notation and suggest a faster algorithm with its O() improvement.
Scoring: 0.20 detection + 0.30 complexity match + 0.50 fix (LLM-judged).

### Task 7 — Test Coverage Review (Hard)
Agent assesses testability and lists 2-3 specific missing unit tests with
inputs, expected outputs, and edge cases covered.
Scoring: 0.20 testability + 0.30 missing tests + 0.50 suggestions (LLM-judged).

### Task 8 — Refactoring Opportunity Detection (Medium)
Agent identifies specific refactoring patterns: extract_method, introduce_parameter_object,
remove_duplicate_code, decompose_conditional. Must cite exact locations.
Scoring: 0.25 detection + 0.75 fix quality (LLM-judged on specificity of recommendations).

### Task 9 — SOLID Principles Violation Detection (Hard)
Agent detects violations of SRP, OCP, LSP, ISP, DIP in class/module design.
Must name the violated principle, affected component, and explain why violated.
Scoring: 0.30 detection + 0.30 keyword match + 0.40 fix (LLM-judged).

### Task 10 — Error Handling Review (Medium)
Agent evaluates error handling: bare excepts, silent failures, missing try/except,
no logging. Returns error_handling_score (0-10) + issues + fix_suggestions.
Scoring: 0.25 detection + 0.25 term match + 0.50 fix (LLM-judged).

### Task 11 — Documentation Quality Review (Easy-Medium)
Agent assesses docstrings, type hints, parameter docs. Returns doc_quality_score (0-10)
+ missing items + a concrete improved docstring example.
Scoring: 0.25 detection + 0.25 term match + 0.50 fix (LLM-judged).

### Task 12 — Concurrency & Race Condition Detection (Hard)
Agent detects race conditions, missing locks, blocking async calls, deadlock risks.
Returns list of {issue_type, affected_code, risk, fix}.
Scoring: 0.30 detection + 0.20 term match + 0.50 fix (LLM-judged).

### Task 13 — API Design Review (Hard)
Agent evaluates naming consistency, param count, input validation, REST design.
Returns api_score (0-10) + issues + improved_signature.
Scoring: 0.25 detection + 0.25 term match + 0.50 fix (LLM-judged).

### Task 14 — Code Comparison Review (Hard)
Agent compares v1 vs v2 code: determines if v2 is an improvement, identifies new bugs,
returns {improvement, new_issues, verdict, reason}.
Scoring: 0.25 detection + 0.25 term match + 0.50 fix (LLM-judged).

### Task 15 — Dependency & Import Review (Medium)
Agent identifies unused imports, risky packages (pickle, eval), over-importing.
Returns {unused_imports, risky_dependencies, cleaner_imports}.
Scoring: 0.25 detection + 0.35 issue match + 0.40 fix (LLM-judged).

## Reward Design

- **Deterministic scoring** for Tasks 1-2 (same input = same score)
- **LLM-as-judge** for Tasks 3-15 fix quality evaluation
- **Structured JSON output** required for Tasks 8-15 (refactoring, SOLID, error handling, etc.)
- **Anti-hack penalties**: -0.1 for vague/low-effort fixes (<10 chars, TODO placeholders)
- **Partial credit**: agents get proportional scores for partially correct answers

## Dataset
- 120 labeled code snippets (Python and JavaScript)
- 30 original snippets covering bugs (Tasks 1-3)
- 70 extended snippets covering smells, security, performance, testability (Tasks 4-7)
- 20 advanced snippets covering refactoring, SOLID, error handling, docs, concurrency, API, comparison, dependencies (Tasks 8-15)
- Distribution: 5 refactoring, 3 SOLID, 3 error handling, 2 docs, 2 concurrency, 2 API design, 2 comparison, 2 dependency

## Baseline Scores

Scores produced by running `uv run python inference.py` with `llama-3.3-70b-versatile` via Groq:

| Task | Name | Mean Score |
|------|------|-----------|
| 1 | Bug Detection (Easy) | 0.400 |
| 2 | Bug Classification (Medium) | 0.600 |
| 3 | Full Code Review (Hard) | 0.566 |
| 4 | Code Smell Detection (Medium) | 0.360 |
| 5 | Security Audit (Hard) | 0.684 |
| 6 | Performance Optimization (Hard) | 0.640 |
| 7 | Test Coverage Review (Hard) | 0.380 |
| 8 | Refactoring Opportunity Detection (Medium) | 1.000 |
| 9 | SOLID Principles Violation Detection (Hard) | 1.000 |
| 10 | Error Handling Review (Medium) | 0.500 |
| 11 | Documentation Quality Review (Easy-Medium) | 1.000 |
| 12 | Concurrency & Race Condition Detection (Hard) | 0.850 |
| 13 | API Design Review (Hard) | 0.350 |
| 14 | Code Comparison Review (Hard) | 0.942 |
| 15 | Dependency & Import Review (Medium) | 0.518 |

**Overall Mean Score: 0.653**

To reproduce:
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export HF_TOKEN="your_groq_api_key"
export MODEL_NAME="llama-3.3-70b-versatile"
uv run python inference.py
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
| `/tasks` | GET | List all 15 tasks |
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
        └── snippets.json     ← 120 labeled code snippets
```

## Author

Palle Venkata Dharaneswara Reddy
