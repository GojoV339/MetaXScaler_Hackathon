---
title: CodeReview-Env
emoji: 🦀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# CodeReview-Env: AI-Powered Code Review RL Environment

CodeReview-Env is a state-of-the-art Reinforcement Learning (RL) pipeline designed to train and evaluate AI agents on the complex task of software engineering code review.

> 📖 **[Read the full Blog Post →](BLOG.md)**
> | 🤗 **[Live Environment on HuggingFace →](https://dharaneswarreddy-codereview-env.hf.space)**
> | 🤖 **[Trained Model →](https://huggingface.co/DharaneswarReddy/codereview-agent)**

Using **Unsloth**, **TRL (Transformer Reinforcement Learning)**, and **GRPO (Group Relative Policy Optimization)**, this project trains a small 1.5B parameter model (Qwen2.5-1.5B) to review code like a Senior Software Engineer. The environment simulates a realistic workflow where an agent must detect bugs, classify them, suggest actionable fixes, and avoid common AI pitfalls like "reward hacking."

---

## 🏗️ Architecture Overview

The system operates across two main architectural boundaries:
1. **The Environment Server (HuggingFace Space)**: A FastAPI server that acts as the "world." It holds the dataset, dishes out coding problems, and uses an LLM-as-a-Judge (powered by Groq) to evaluate the accuracy of the agent's code reviews.
2. **The GRPO Trainer (Colab/Local)**: The training loop running `Unsloth` and `TRL`. It fetches problems from the environment, asks the local model to generate reviews, and updates the model's weights based on a **5-Component Reward System**.

### GRPO Training Pipeline — End-to-End Workflow

![GRPO Training Pipeline — End-to-End Workflow](res/Work_Flow.svg)

*The complete 4-phase pipeline: Pre-fetching → Generation → 5-Component Scoring → GRPO Policy Update.*

---

## 🔄 Detailed Workflow: What is happening?

Here is the exact lifecycle of a single training step:

### 1. Where do the inputs come from?
All raw code snippets live inside `env/data/snippets.json` (120 snippets) and multi-file pull requests in `env/data/prs.json` (30 PRs). These contain code with actual bugs, along with the "ground truth" of what the bug is.

### 2. The GRPO Generation Phase
1. **Reset**: The trainer (`train_grpo.py`) calls the `POST /reset` endpoint on the environment.
2. **Observation**: The environment selects a snippet based on the current *Curriculum Level* and returns the raw `code`, `language`, and a `context_hint`. 
3. **Prompting**: The trainer formats this into a system prompt tailored to the curriculum level and passes it to the `Qwen2.5-1.5B` model.
4. **Generation**: The model generates **4 different variations** of a code review concurrently.

### 3. The Evaluation & Scoring Phase
1. **Action Parsing**: The trainer extracts the raw JSON from the model's text (extracting `has_bug`, `bug_type`, `severity`, and `suggested_fix`).
2. **Environment Step**: The trainer sends these JSON objects individually to the environment's `POST /step` endpoint.
3. **The Grader**: The FastAPI server routes the action to `env/graders.py`. It uses a **Groq LLM-as-a-judge** to determine if the agent actually found the correct bug. It sends back a breakdown of detection and classification accuracy.

### 4. The **5-Component Reward System**
To prevent the model from blindly guessing or "reward hacking", the trainer calculates a highly granular reward signal locally before passing it to GRPO:
- **Format (15%)**: Did the model output perfect, parsable JSON with all required keys?
- **Detection (30%)**: Did the environment confirm the `has_bug` flag was correct?
- **Classification (20%)**: Did the environment confirm the `bug_type` matched the ground truth?
- **Confidence (15%)**: Is the model staying balanced? (Penalizes the model if it guesses "true" 100% of the time).
- **Quality (20%)**: Is the `suggested_fix` actually a detailed code replacement, or is it just generic boilerplate like "fix the bug"?
- **Anti-Hack Penalty**: Heavy negative score if the model repeats the exact same review 10 times in a row.

### 5. Policy Update
GRPO compares the 4 generated reviews. The reviews that scored the highest overall reward (e.g., proper JSON, detailed fix, correctly detected) push the model weights in that direction. 

---

## 📈 Curriculum Learning

The training script automatically scales the difficulty of the prompts:
- **Level 1**: Only asks the model to output `has_bug` (True/False).
- **Level 2**: Asks the model to detect the bug AND classify the `bug_type`.
- **Level 3+**: Asks for a full review including a `suggested_fix`.

The trainer advances to the next level automatically once the rolling average reward hits `0.45`.

---

## 📁 File Structure: What does each file do?

### The Training Engine (Client-Side)
- `training/train_grpo.py`: Core Unsloth/TRL GRPO training loop. Handles curriculum logic, parses model outputs, hits the environment API, and computes the 5-component reward.
- `training/colab_notebook.ipynb`: Optimized setup script for running the training pipeline on a free Google Colab T4 GPU instance.

### The Environment Server (Host-Side)
- `app.py`: The FastAPI server entry point. Exposes `/reset`, `/step` and handles the PR pipelines.
- `env/environment.py`: State management for dispensing individual `snippets.json`.
- `env/pr_environment.py`: Complex state machine for dispensing multi-file PR reviews.
- `env/graders.py`: The deterministic and LLM-as-a-judge scoring functions. Uses Groq to grade complex `suggested_fixes`.
- `env/models.py`: Pydantic data schemas (e.g., `Action`, `Reward`, `Observation`) to enforce strict type checking across the API boundary.

### Data & Evaluation
- `env/data/`: Houses the actual coding problems: `snippets.json` and `prs.json`.
- `inference.py`: Local benchmarking script. Used to test your newly trained model (or base models like GPT-OSS) against the environment to get a baseline score before training.

---

## 🏁 Getting Started & Secrets

You need two secret keys to run the pipeline:
1. **`GROQ_API_KEY`**: Sourced from [console.groq.com/keys](https://console.groq.com/keys). Used by the environment to run the LLM judges, and also by `inference.py` to run baselines.
2. **`HF_TOKEN`**: Sourced from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Must be a 'write' token). Used by `train_grpo.py` at the very end of training to upload your fine-tuned CodeReview agent to the HuggingFace Hub.

### Local Setup
1. Clone the repo and install requirements.
2. Put both your keys into a `.env` file at the root of the project:
   ```env
   GROQ_API_KEY=gsk_...
   HF_TOKEN=hf_...
   ```
3. Run the environment: `python app.py`

### Training on Colab
1. Open `training/colab_notebook.ipynb`. 
2. Open the **Secrets** tab (Key icon) on the left sidebar. Add both `GROQ_API_KEY` and `HF_TOKEN`.
3. Hit **Runtime > Run All**. The script will automatically clone your repo, pull the secrets, train the model for 300 steps, graph the reward curve, and push the model to HuggingFace!

---

## 🚀 The Realistic Roadmap: Future Work

The current version proves the core concept. The architecture, curriculum, and composable rubric system are sound. We are one data upgrade away from a genuinely publishable, production-grade reviewer. Here is what we are going to do next:

1. **Replace Synthetic Data**: Swap `snippets.json` with 5,000 real PR diffs from open-source GitHub repositories (`rust-lang`, `Django`, etc.). The ground truth triple will become `(buggy diff, reviewer comment, fixed code)`.
2. **Add Semantic Reward**: Introduce embedding-based reward. Stop rewarding models that just game the label, and start rewarding models that understand the issue using semantic similarity to real reviewer comments.
3. **Execution-based Reward**: Implement testing verification for PRs with test coverage. If the model's suggested fix passes the test suite, `Pass = reward`. This is the strongest training signal and anchors the whole model.
4. **Deploy as GitHub Action**: Switch output format to inline comments matching GitHub's Review API. Fine-tune on the new action space, deploy as a GitHub Action, and collect real feedback from developers.
