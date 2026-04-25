# training/train_grpo.py
"""
CodeReview-Env — GRPO Training Script
======================================
Unsloth + HF TRL GRPOTrainer
Curriculum learning: easy tasks first, unlock harder tasks progressively
Multiple independent reward signals (7 task graders)

Install (run in Colab first):
  !pip install unsloth trl transformers requests pydantic matplotlib -q
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

Usage:
  python training/train_grpo.py
"""

import os
import json
import re
import sys
import time
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Unsloth must be imported before other heavy libraries
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
from unsloth import FastLanguageModel
import torch

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "https://dharaneswarreddy-codereview-env.hf.space")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SAVE_REPO = os.getenv("SAVE_REPO", "")

# Training config — tuned for free Colab T4 (15GB VRAM)
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
NUM_TRAIN_STEPS = 200
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 5e-6
NUM_GENERATIONS = 4

# Curriculum thresholds
CURRICULUM_THRESHOLD = 0.5
CURRICULUM_WINDOW = 20

# ── Reward tracking ───────────────────────────────────────────────────────────
reward_history: List[float] = []
curriculum_level: int = 1
level_rewards: Dict[int, List[float]] = {1: [], 2: [], 3: []}

# ── Environment API ───────────────────────────────────────────────────────────

def env_reset(task_level: int = 1) -> dict:
    """Reset single-snippet env."""
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_level": task_level}, timeout=30)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        print(f"[env_reset error] {e}")
        return {}


def env_step(action: dict) -> dict:
    """Submit action to single-snippet env."""
    try:
        r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
        return r.json() if r.status_code == 200 else {"reward": {"score": 0.0, "feedback": "error"}, "done": True}
    except Exception as e:
        print(f"[env_step error] {e}")
        return {"reward": {"score": 0.0, "feedback": "error"}, "done": True}


def parse_action(text: str) -> dict:
    """Parse LLM JSON output. Never crashes."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"has_bug": False, "bug_type": "no_bug", "severity": "none", "suggested_fix": ""}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    1: """You are an expert code reviewer. Detect if code has a bug.
Look for: divide-by-zero, null references, off-by-one, SQL injection, wrong operators.
Return ONLY this JSON:
{"has_bug": true/false, "bug_type": "no_bug", "severity": "none", "suggested_fix": ""}""",

    2: """You are an expert code reviewer. Detect bugs AND classify their type.
Bug types: logic_error, security_vulnerability, performance_issue, syntax_error, no_bug
Return ONLY this JSON:
{"has_bug": true/false, "bug_type": "TYPE_HERE", "severity": "none", "suggested_fix": ""}""",

    3: """You are a senior software engineer. Give a complete code review.
Detect the bug, classify it, assign severity, and write a SPECIFIC fix.
Return ONLY this JSON:
{"has_bug": true/false, "bug_type": "TYPE", "severity": "low|medium|high|critical|none", "suggested_fix": "SPECIFIC FIX HERE"}"""
}


def make_prompt(obs: dict, tokenizer, level: int) -> str:
    """Format observation as a model prompt."""
    code = obs.get("code", "")
    lang = obs.get("language", "python")
    hint = obs.get("context_hint") or ""

    user = (
        f"Review this {lang} code:\n\n"
        f"```{lang}\n{code}\n```\n\n"
        f"{hint}\n"
        "Return ONLY the JSON."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[level]},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── Curriculum reward function ────────────────────────────────────────────────

def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """GRPO reward function. Calls live env for scoring."""
    global curriculum_level
    rewards = []

    for completion in completions:
        obs = env_reset(task_level=curriculum_level)
        if not obs or not obs.get("code"):
            rewards.append(0.0)
            continue

        action = parse_action(completion)

        # Anti reward hacking
        fix = action.get("suggested_fix", "")
        if curriculum_level == 3 and len(fix) < 10:
            rewards.append(0.0)
            print(f"  [HACK DETECTED] trivial fix: '{fix}'")
            continue

        result = env_step(action)
        score = float(result.get("reward", {}).get("score", 0.0))
        rewards.append(score)

        if curriculum_level in level_rewards:
            level_rewards[curriculum_level].append(score)

    reward_history.extend(rewards)
    mean = sum(rewards) / max(len(rewards), 1)
    print(f"  Level {curriculum_level} | Batch mean: {mean:.3f} | Rewards: {[f'{r:.2f}' for r in rewards]}")

    # Curriculum progression check
    recent = level_rewards.get(curriculum_level, [])[-CURRICULUM_WINDOW:]
    if len(recent) >= CURRICULUM_WINDOW:
        recent_mean = sum(recent) / len(recent)
        if recent_mean >= CURRICULUM_THRESHOLD and curriculum_level < 3:
            curriculum_level += 1
            print(f"\n  [CURRICULUM] Advancing to level {curriculum_level}\n")

    return rewards

# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(tokenizer, num_samples: int = 400):
    """Build training dataset from live env."""
    from datasets import Dataset

    samples = []
    print(f"Building dataset ({num_samples} samples)...")

    for i in range(num_samples):
        if i < num_samples * 0.5:
            level = 1
        elif i < num_samples * 0.8:
            level = min(2, curriculum_level)
        else:
            level = min(3, curriculum_level)

        obs = env_reset(task_level=level)
        if not obs or not obs.get("code"):
            continue

        prompt = make_prompt(obs, tokenizer, level)
        samples.append({"prompt": prompt, "level": level})

        if (i + 1) % 50 == 0:
            print(f"  Sampled {i+1}/{num_samples}")

    print(f"Dataset: {len(samples)} samples")
    return Dataset.from_list(samples)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load small training model with Unsloth (4-bit)."""
    print(f"Loading {TRAIN_MODEL} with Unsloth (4-bit)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"Model loaded. Trainable params: {model.num_parameters(only_trainable=True):,}")
    return model, tokenizer

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, num_episodes: int = 20, level: int = 1) -> float:
    """Evaluate trained agent. Returns mean score."""
    import torch
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    scores = []
    print(f"\nEvaluating {num_episodes} episodes at level {level}...")

    for i in range(num_episodes):
        obs = env_reset(task_level=level)
        if not obs:
            continue

        prompt = make_prompt(obs, tokenizer, level)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action(generated)
        result = env_step(action)
        score = float(result.get("reward", {}).get("score", 0.0))
        scores.append(score)

        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1}: score={score:.3f}")

    mean = sum(scores) / max(len(scores), 1)
    print(f"Mean score: {mean:.3f}")
    return mean

# ── Reward curve plotting ────────────────────────────────────────────────────

def plot_reward_curve(rewards: List[float], before: float, after: float, save_path: str = "reward_curve.png"):
    """Plot reward curve with before/after markers."""
    if not rewards:
        print("No rewards to plot")
        return

    window = 15
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(sum(rewards[start:i+1]) / (i - start + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rewards, alpha=0.25, color="#7C3AED", linewidth=0.8, label="Raw reward")
    ax.plot(smoothed, color="#7C3AED", linewidth=2.5, label=f"Rolling mean (w={window})")
    ax.axhline(y=before, color="#E11D48", linewidth=1.5, linestyle="--", label=f"Before: {before:.3f}")
    ax.axhline(y=after, color="#16A34A", linewidth=1.5, linestyle="--", label=f"After: {after:.3f}")

    # Curriculum level transitions
    level_2_start = len(level_rewards.get(1, []))
    level_3_start = level_2_start + len(level_rewards.get(2, []))
    if level_2_start > 0:
        ax.axvline(x=level_2_start, color="#D97706", linewidth=1, linestyle=":", alpha=0.7)
        ax.text(level_2_start + 2, 0.05, "Level 2", color="#D97706", fontsize=9)
    if level_3_start > level_2_start:
        ax.axvline(x=level_3_start, color="#0D9488", linewidth=1, linestyle=":", alpha=0.7)
        ax.text(level_3_start + 2, 0.05, "Level 3", color="#0D9488", fontsize=9)

    ax.set_xlabel("Training episode", fontsize=12)
    ax.set_ylabel("Reward (0.0 – 1.0)", fontsize=12)
    ax.set_title("CodeReview-Env — Agent Learning Curve (GRPO + Curriculum)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nReward curve saved: {save_path}")

    if len(rewards) >= 30:
        early = rewards[:20]
        late = rewards[-20:]
        print(f"\nTraining summary:")
        print(f"  Early mean (first 20):  {sum(early)/len(early):.3f}")
        print(f"  Late mean (last 20):    {sum(late)/len(late):.3f}")
        print(f"  Improvement:            {(sum(late)/len(late)) - (sum(early)/len(early)):+.3f}")
        print(f"  Before eval:            {before:.3f}")
        print(f"  After eval:             {after:.3f}")
        print(f"  Total improvement:      {after - before:+.3f}")

# ── Main training loop ────────────────────────────────────────────────────────

def train():
    from trl import GRPOTrainer, GRPOConfig

    # Test environment connection
    print("Testing environment connection...")
    test = env_reset(task_level=1)
    if not test.get("snippet_id"):
        print(f"ERROR: Cannot connect to {ENV_URL}")
        print("Check ENV_URL and ensure your HF Space is running.")
        sys.exit(1)
    print(f"Environment OK — snippet: {test['snippet_id']}\n")

    # Load model
    model, tokenizer = load_model()

    # Evaluate BEFORE training
    print("\n=== BEFORE TRAINING ===")
    score_before = evaluate(model, tokenizer, num_episodes=20, level=1)

    # Build dataset
    dataset = build_dataset(tokenizer, num_samples=NUM_TRAIN_STEPS * NUM_GENERATIONS)

    # GRPO config
    training_args = GRPOConfig(
        output_dir="./codereview_grpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_steps=NUM_TRAIN_STEPS,
        num_generations=NUM_GENERATIONS,
        logging_steps=10,
        save_steps=100,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_function,
        processing_class=tokenizer,
    )

    print(f"\n=== TRAINING ({NUM_TRAIN_STEPS} steps, curriculum start level 1) ===")
    print(f"Environment: {ENV_URL}")
    print(f"Training model: {TRAIN_MODEL}")
    trainer.train()

    # Evaluate AFTER training
    print("\n=== AFTER TRAINING ===")
    score_after = evaluate(model, tokenizer, num_episodes=20, level=curriculum_level)

    # Plot reward curve
    plot_reward_curve(reward_history, score_before, score_after, "reward_curve.png")

    # Save model — proper LoRA merge via Unsloth
    print("\nSaving model...")
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Model saved to ./trained_model")

    if SAVE_REPO and HF_TOKEN:
        print(f"\nPushing to HuggingFace: {SAVE_REPO}")
        model.push_to_hub(SAVE_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(SAVE_REPO, token=HF_TOKEN)
        print("Model pushed to HF Hub!")

    print(f"\n{'='*60}")
    print(f"  Before training:  {score_before:.3f}")
    print(f"  After training:   {score_after:.3f}")
    print(f"  Improvement:      {score_after - score_before:+.3f}")
    print(f"  Curriculum level reached: {curriculum_level}")
    print(f"{'='*60}")

    return model, tokenizer, score_before, score_after


if __name__ == "__main__":
    train()
