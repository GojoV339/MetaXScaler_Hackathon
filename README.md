# CodeReview-Env: AI-Powered Code Review RL Environment

CodeReview-Env is a state-of-the-art Reinforcement Learning (RL) environment designed to train and evaluate AI agents on the complex task of software engineering code review. It simulates a realistic "PR culture" where an agent must not only detect bugs but also classify them, suggest fixes, and ultimately decide the fate of a Pull Request.

---

## 🚀 How it Works: Step-by-Step Process

The environment operates in two primary modes: **Single Snippet Evaluation** and the **Multi-Step PR Pipeline**.

### 1. Initialization
- **Environment Reset**: The agent calls `/reset` (or `/pr/reset`).
- **Data Loading**: The environment randomly selects a code snippet (from 120 total) or a Pull Request (from 30 total).
- **Observation**: The agent receives an `Observation` object containing:
    - The code to review.
    - The programming language (Python/JavaScript).
    - A "hint" or context (for harder tasks).
    - PR metadata (title, description).

### 2. The Review Loop
- **Agent Action**: The agent returns an `Action` object (JSON) specify:
    - `has_bug`: Boolean.
    - `bug_type`: Classification (e.g., `security_vulnerability`).
    - `severity`: Assessment (from `low` to `critical`).
    - `suggested_fix`: A specific, actionable code correction.
- **Environment Step**: The agent submits the action to `/step`.
- **Feedback**: The environment returns a `Reward` object containing a score (0.01 to 0.99) and detailed human-readable feedback.

### 3. Graduation (PR Pipeline)
In the PR mode, the agent reviews 3-5 files sequentially and ends with a **Final Verdict** (APPROVE, REQUEST_CHANGES, or REJECT).

---

## ⚖️ The Grading System: Who is Judging?

We use a "Hybrid Judge" system to ensure evaluations are both accurate and nuanced.

| Task Level | Judge Type | What is measured? |
| :--- | :--- | :--- |
| **1–2 (Easy)** | **Deterministic** | Simple correctness: Did you find the bug? Did you get the category right? |
| **3–7 (Standard)** | **LLM Judge** | The quality of your `suggested_fix`. Does it actually solve the problem without creating new ones? |
| **8–15 (Advanced)** | **Advanced LLM Judge** | Evaluation of SOLID principles, concurrency, API design, and refactoring quality. |
| **PR Pipeline** | **PR Master Judge** | Consistency across multiple files and the "Technical Leadership" shown in the final verdict. |

### What the LLM Judges specifically look for:
- **Correctness**: Does the fix address the root cause?
- **Security Check**: Did you catch that SQL injection or Path Traversal?
- **Performance**: Did you identify the O(n²) loop or the N+1 query?
- **Anti-Hacking**: The graders penalize "always-approve" behavior or vague, copy-pasted summaries.

---

## 🧠 The Training Process: GRPO + Curriculum

We use **Group Relative Policy Optimization (GRPO)** via the `Unsloth` and `TRL` frameworks for training.

1.  **Curriculum Learning**: The agent starts on **Level 1 (Direct Bug Detection)**. Once it reaches a >50% hit rate, the environment automatically "unlocks" more complex tasks (Level 2 → Level 3 → PR Pipeline).
2.  **Relative Scoring**: In each training step, the agent generates multiple different reviews for the same code. GRPO compares these reviews against each other and rewards the ones that get higher scores from the environment's graders.
3.  **Low-Rank Adaptation (LoRA)**: We train using 4-bit quantization, allowing the process to run efficiently on a single consumer GPU (like a Colab T4).

---

## 📁 File Structure: What does each file do?

- `app.py`: The heart of the system. A FastAPI server that exposes the RL environment to the world.
- `inference.py`: The evaluation "engine." It runs full baselines to see how your model is performing.
- `env/models.py`: Defines the "language" (Pydantic models) used by the agent and the environment to communicate.
- `env/environment.py`: The logic for individual snippet tasks.
- `env/pr_environment.py`: The state machine for the multi-step PR pipeline.
- `env/graders.py`: The logic for scoring single snippet reviews.
- `env/pr_graders.py`: Complex scoring for PR verdicts and code comparisons.
- `env/data/`: Houses `snippets.json` (120 snippets) and `prs.json` (30 PRs).
- `training/train_grpo.py`: The script that actually trains the brain of the agent.

---

## 🌎 Real-world Relatability

### Is this exactly how they review in the real world?
**Yes and No.**

- **The "Automated" Part (Standard Tasks)**: This mirrors the real-world use of **Static Analysis tools** (like SonarQube or Snyk). Humans don't usually manually check for simple syntax errors; tools do. We train the agent to be as good as these tools.
- **The "PR Pipeline" Part**: This is **exactly** how senior engineers and tech leads operate. 
    - They don't just look at one line; they look at how a change in `db_utils.py` might break the `user_auth.js` flow. 
    - They compare a "v1" approach vs. a "v2" approach (which we simulate in our `is_comparison` tasks).
    - They provide a `verdict_summary` that guides the junior dev on what to fix first. 

**This environment is designed to bridge the gap between "Tool-based checking" and "Human-level architectural review."**

---

## 🛠️ The Tasks we are doing

1.  **Bug Detection**: Pure binary identification.
2.  **Classification**: Identifying the *nature* of the flaw (Security vs. Logic).
3.  **Suggested Fix**: Proposing a code correction.
4.  **Refactoring**: Improving code quality without changing behavior.
5.  **SOLID Principles**: Ensuring high-level design patterns.
6.  **Security Audit**: Deep sweeps for OWASP vulnerabilities.
7.  **Performance Tuning**: Identifying time/space complexity issues.
8.  **PR Verdict**: Synthesizing multiple reviews into one business decision.

---

## 🏁 Getting Started

1.  **Configure**: Add your `MOONSHOT_API_KEY` to `.env`.
2.  **Run Server**: `python app.py`
3.  **Run Baseline**: `python inference.py pr`
4.  **Train**: Use the `training/colab_notebook.ipynb` to start a GRPO run.
