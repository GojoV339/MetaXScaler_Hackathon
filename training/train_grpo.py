# training/train_grpo.py
"""
CodeReview-Env — GRPO Training Script (v2: Multi-Component Rewards)
====================================================================
Unsloth + HF TRL GRPOTrainer

Key improvements over v1:
- 5 independent reward components (format, detection, classification, confidence, quality)
- Anti-reward hacking: repetition, boilerplate, diversity checks
- Per-component monitoring columns
- Separated HF_TOKEN from Groq API key
- Graceful HuggingFace push with error handling

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
from collections import Counter

# Unsloth must be imported before other heavy libraries
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
from unsloth import FastLanguageModel
import torch

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "https://dharaneswarreddy-codereview-env.hf.space")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TRAIN_MODEL = os.getenv("TRAIN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SAVE_REPO = os.getenv("SAVE_REPO", "")

# Training config — tuned for free Colab T4 (15GB VRAM)
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
NUM_TRAIN_STEPS = int(os.environ.get("NUM_TRAIN_STEPS", 300))
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 5e-6
NUM_GENERATIONS = 4

# Curriculum thresholds
CURRICULUM_THRESHOLD = 0.45
CURRICULUM_WINDOW = 20

# Reward weights (must sum to 1.0)
REWARD_WEIGHTS = {
    "format":         0.15,
    "detection":      0.30,
    "classification": 0.20,
    "confidence":     0.15,
    "quality":        0.20,
}

# ── Tracking state ────────────────────────────────────────────────────────────
reward_history: List[float] = []
component_history: Dict[str, List[float]] = {k: [] for k in REWARD_WEIGHTS}
curriculum_level: int = 1
level_rewards: Dict[int, List[float]] = {1: [], 2: [], 3: []}

# Anti-hacking state
recent_outputs: List[str] = []          # Last N raw outputs for repetition check
recent_predictions: List[bool] = []     # Last N has_bug predictions for bias check
hack_count: int = 0
timeout_count: int = 0
RECENT_WINDOW = 40

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
        return r.json() if r.status_code == 200 else {"reward": {"score": 0.0, "breakdown": {}, "feedback": "error"}, "done": True}
    except Exception as e:
        print(f"[env_step error] {e}")
        return {"reward": {"score": 0.0, "breakdown": {}, "feedback": "error"}, "done": True}


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

# ── Multi-Component Reward Functions ──────────────────────────────────────────

BOILERPLATE_FIXES = {
    "add error handling", "fix the bug", "handle the error",
    "refactor the code", "improve the code", "fix this",
    "needs fixing", "should be fixed", "todo", "fix later",
    "add validation", "use proper error handling",
}


def is_valid_json_output(text: str) -> bool:
    """Check if the raw text contains valid JSON."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text).strip()
    try:
        json.loads(text)
        return True
    except Exception:
        # Try to find JSON in the text
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                json.loads(m.group())
                return True
            except Exception:
                pass
    return False


def score_format(completion: str, action: dict) -> float:
    """R_format: Score format compliance (0.0 - 1.0)."""
    score = 0.0

    # Valid JSON structure
    if is_valid_json_output(completion):
        score += 0.40

    # Required fields present
    if "has_bug" in action and isinstance(action.get("has_bug"), bool):
        score += 0.20
    if "bug_type" in action and isinstance(action.get("bug_type"), str):
        score += 0.15
    if "severity" in action and isinstance(action.get("severity"), str):
        score += 0.10
    if "suggested_fix" in action and isinstance(action.get("suggested_fix"), str):
        score += 0.15

    return min(score, 1.0)


def score_detection(env_result: dict) -> float:
    """R_detect: Detection accuracy from env breakdown (0.0 or 1.0 for task1, 0.0-0.4 for task2+)."""
    breakdown = env_result.get("reward", {}).get("breakdown", {})
    det = float(breakdown.get("detection", 0.0))

    # Normalize: task1 returns 1.0 or 0.0, task2 returns 0.4 or 0.0
    if det >= 0.4:
        return 1.0
    elif det > 0.0:
        return det / 0.4  # Normalize partial scores
    return 0.0


def score_classification(env_result: dict, level: int) -> float:
    """R_classify: Classification accuracy from env breakdown."""
    if level < 2:
        # Level 1 doesn't require classification — give baseline credit if format is OK
        return 0.5

    breakdown = env_result.get("reward", {}).get("breakdown", {})
    cls = float(breakdown.get("classification", 0.0))

    # Normalize: task2 returns 0.6 or 0.0
    if cls >= 0.6:
        return 1.0
    elif cls > 0.0:
        return cls / 0.6
    return 0.0


def score_confidence(action: dict) -> float:
    """R_confidence: Penalize prediction bias (always True or always False)."""
    global recent_predictions
    prediction = action.get("has_bug", False)
    recent_predictions.append(prediction)

    if len(recent_predictions) > RECENT_WINDOW:
        recent_predictions = recent_predictions[-RECENT_WINDOW:]

    if len(recent_predictions) < 5:
        return 0.5  # Not enough data yet

    true_ratio = sum(recent_predictions) / len(recent_predictions)

    # Ideal: ~50% true, ~50% false (balanced)
    # Penalty increases as ratio diverges from 0.5
    # At 100% same answer: score = 0.0
    # At 50/50: score = 1.0
    balance = 1.0 - 2.0 * abs(true_ratio - 0.5)
    return max(0.0, balance)


def score_quality(action: dict, level: int) -> float:
    """R_quality: Fix specificity and quality heuristics."""
    fix = action.get("suggested_fix", "") or ""
    fix = str(fix).strip()

    if level == 1:
        # Level 1 doesn't require fixes — just needs correct has_bug
        # Give partial credit based on any analysis provided
        if len(fix) > 0:
            return min(len(fix) / 50.0, 1.0)  # Up to 1.0 for 50+ char fixes
        return 0.3  # Baseline for level 1

    score = 0.0

    # Length-based (non-trivial response)
    if len(fix) >= 15:
        score += 0.25
    if len(fix) >= 40:
        score += 0.15
    if len(fix) >= 80:
        score += 0.10

    # Contains actionable language
    actionable_keywords = ["replace", "change", "add", "remove", "use", "instead",
                           "should be", "fix by", "wrap", "check for", "validate"]
    matches = sum(1 for k in actionable_keywords if k in fix.lower())
    score += min(matches / 3.0, 0.25) * 0.25 / 0.25  # Normalize

    # Not boilerplate
    is_boilerplate = fix.lower().strip() in BOILERPLATE_FIXES or len(fix) < 10
    if not is_boilerplate:
        score += 0.25

    return min(score, 1.0)


def compute_anti_hack_penalty(completion: str, action: dict) -> float:
    """Returns a negative penalty (or 0.0) for detected reward hacking."""
    global recent_outputs, hack_count
    penalty = 0.0

    # Track recent outputs
    output_hash = completion.strip()[:100]  # First 100 chars as fingerprint
    recent_outputs.append(output_hash)
    if len(recent_outputs) > RECENT_WINDOW:
        recent_outputs = recent_outputs[-RECENT_WINDOW:]

    # Repetition check: >50% identical outputs in window
    if len(recent_outputs) >= 10:
        counter = Counter(recent_outputs[-20:])
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count > len(recent_outputs[-20:]) * 0.5:
            penalty -= 0.15
            hack_count += 1

    # Trivial/empty fix at higher levels
    fix = str(action.get("suggested_fix", "") or "")
    if fix.lower().strip() in BOILERPLATE_FIXES:
        penalty -= 0.10
        hack_count += 1

    # Suspiciously short output (likely not thinking)
    if len(completion.strip()) < 20:
        penalty -= 0.10

    return penalty


def compute_multi_reward(completion: str, env_result: dict, level: int) -> Dict[str, float]:
    """Compute all 5 reward components + anti-hacking penalty.

    Returns dict with per-component scores and final weighted reward.
    """
    action = parse_action(completion)

    components = {
        "format":         score_format(completion, action),
        "detection":      score_detection(env_result),
        "classification": score_classification(env_result, level),
        "confidence":     score_confidence(action),
        "quality":        score_quality(action, level),
    }

    penalty = compute_anti_hack_penalty(completion, action)

    # Weighted combination
    final = sum(components[k] * REWARD_WEIGHTS[k] for k in REWARD_WEIGHTS)
    final = final + penalty
    final = max(0.0, min(final, 1.0))

    return {
        "final": final,
        "components": components,
        "penalty": penalty,
    }


# ── GRPO Reward Function ─────────────────────────────────────────────────────

def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """GRPO reward function with multi-component scoring."""
    global curriculum_level
    rewards = []
    batch_components = {k: [] for k in REWARD_WEIGHTS}

    for completion in completions:
        obs = env_reset(task_level=curriculum_level)
        if not obs or not obs.get("code"):
            rewards.append(0.0)
            for k in batch_components:
                batch_components[k].append(0.0)
            continue

        action = parse_action(completion)

        # Get env score (for detection/classification breakdown)
        result = env_step(action)

        # Compute multi-component reward
        reward_info = compute_multi_reward(completion, result, curriculum_level)
        rewards.append(reward_info["final"])

        # Track per-component
        for k in REWARD_WEIGHTS:
            val = reward_info["components"][k]
            batch_components[k].append(val)
            component_history[k].append(val)

        if curriculum_level in level_rewards:
            level_rewards[curriculum_level].append(reward_info["final"])

    reward_history.extend(rewards)
    mean = sum(rewards) / max(len(rewards), 1)

    # Enhanced logging with component breakdown
    comp_means = {k: sum(v) / max(len(v), 1) for k, v in batch_components.items()}
    comp_str = " | ".join(f"{k[:3]}={v:.2f}" for k, v in comp_means.items())
    print(f"  L{curriculum_level} mean={mean:.3f} | {comp_str} | hacks={hack_count} | R={[f'{r:.2f}' for r in rewards]}")

    # Save detailed JSON log for post-training plotting
    import json as _json
    log_entry = {
        "step": len(reward_history),
        "mean_reward": round(mean, 4),
        "curriculum_level": curriculum_level,
        "hack_count": hack_count,
        "timeout_count": timeout_count,
        "components": {k: round(v, 4) for k, v in comp_means.items()},
        "rewards": [round(r, 4) for r in rewards],
    }
    with open("training_log.jsonl", "a") as _f:
        _f.write(_json.dumps(log_entry) + "\n")

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

        # Use multi-component reward for eval too
        reward_info = compute_multi_reward(generated, result, level)
        score = reward_info["final"]
        scores.append(score)

        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1}: score={score:.3f}")

    mean = sum(scores) / max(len(scores), 1)
    print(f"Mean score: {mean:.3f}")
    return mean

# ── Enhanced Reward Curve ─────────────────────────────────────────────────────

def plot_reward_curve(rewards: List[float], before: float, after: float, save_path: str = "reward_curve.png"):
    """Plot reward curve with component breakdown and before/after markers."""
    if not rewards:
        print("No rewards to plot")
        return

    window = 15

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 2])

    # ── Top plot: Overall reward ──
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(sum(rewards[start:i+1]) / (i - start + 1))

    ax1.plot(rewards, alpha=0.20, color="#7C3AED", linewidth=0.8, label="Raw reward")
    ax1.plot(smoothed, color="#7C3AED", linewidth=2.5, label=f"Rolling mean (w={window})")
    ax1.axhline(y=before, color="#E11D48", linewidth=1.5, linestyle="--", label=f"Before: {before:.3f}")
    ax1.axhline(y=after, color="#16A34A", linewidth=1.5, linestyle="--", label=f"After: {after:.3f}")

    # Curriculum level transitions
    level_2_start = len(level_rewards.get(1, []))
    level_3_start = level_2_start + len(level_rewards.get(2, []))
    if level_2_start > 0 and level_2_start < len(rewards):
        ax1.axvline(x=level_2_start, color="#D97706", linewidth=1, linestyle=":", alpha=0.7)
        ax1.text(level_2_start + 2, 0.95, "Level 2", color="#D97706", fontsize=9)
    if level_3_start > level_2_start and level_3_start < len(rewards):
        ax1.axvline(x=level_3_start, color="#0D9488", linewidth=1, linestyle=":", alpha=0.7)
        ax1.text(level_3_start + 2, 0.95, "Level 3", color="#0D9488", fontsize=9)

    ax1.set_ylabel("Overall Reward (0.0 – 1.0)", fontsize=12)
    ax1.set_title("CodeReview-Env — Multi-Component GRPO Training", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)

    # ── Bottom plot: Per-component breakdown ──
    comp_colors = {
        "format": "#3B82F6",
        "detection": "#EF4444",
        "classification": "#F59E0B",
        "confidence": "#10B981",
        "quality": "#8B5CF6",
    }

    for comp_name, color in comp_colors.items():
        hist = component_history.get(comp_name, [])
        if not hist:
            continue
        # Smooth each component
        comp_smoothed = []
        for i in range(len(hist)):
            start = max(0, i - window + 1)
            comp_smoothed.append(sum(hist[start:i+1]) / (i - start + 1))
        ax2.plot(comp_smoothed, color=color, linewidth=1.5, alpha=0.8,
                 label=f"{comp_name} ({REWARD_WEIGHTS[comp_name]:.0%})")

    ax2.set_xlabel("Training episode", fontsize=12)
    ax2.set_ylabel("Component Score", fontsize=12)
    ax2.legend(fontsize=9, loc="upper left", ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nReward curve saved: {save_path}")

    # Training summary
    if len(rewards) >= 30:
        early = rewards[:20]
        late = rewards[-20:]
        print(f"\n{'='*60}")
        print(f"  TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"  Early mean (first 20):  {sum(early)/len(early):.3f}")
        print(f"  Late mean (last 20):    {sum(late)/len(late):.3f}")
        print(f"  Improvement:            {(sum(late)/len(late)) - (sum(early)/len(early)):+.3f}")
        print(f"  Before eval:            {before:.3f}")
        print(f"  After eval:             {after:.3f}")
        print(f"  Total improvement:      {after - before:+.3f}")
        print(f"  Hack detections:        {hack_count}")
        print(f"  Curriculum reached:     Level {curriculum_level}")

        # Per-component summary
        print(f"\n  Component breakdown (last 50 episodes):")
        for comp_name in REWARD_WEIGHTS:
            hist = component_history.get(comp_name, [])
            if hist:
                last_50 = hist[-50:]
                print(f"    {comp_name:15s}: {sum(last_50)/len(last_50):.3f} (weight={REWARD_WEIGHTS[comp_name]:.0%})")
        print(f"{'='*60}")

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

    # Show reward config
    print("Reward components:")
    for k, v in REWARD_WEIGHTS.items():
        print(f"  {k:15s}: {v:.0%}")
    print()

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
    print(f"Reward: 5-component (format/detect/classify/confidence/quality)")
    trainer.train()

    # Evaluate AFTER training
    print("\n=== AFTER TRAINING ===")
    score_after = evaluate(model, tokenizer, num_episodes=20, level=curriculum_level)

    # Plot reward curve
    plot_reward_curve(reward_history, score_before, score_after, "reward_curve.png")

    # Save model locally
    print("\nSaving model...")
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Model saved to ./trained_model")

    # Push to HuggingFace (graceful)
    if SAVE_REPO and HF_TOKEN and HF_TOKEN.startswith("hf_"):
        try:
            print(f"\nPushing to HuggingFace: {SAVE_REPO}")
            model.push_to_hub(SAVE_REPO, token=HF_TOKEN)
            tokenizer.push_to_hub(SAVE_REPO, token=HF_TOKEN)
            print("Model pushed to HF Hub!")
        except Exception as e:
            print(f"\n[WARNING] HF push failed: {e}")
            print("Model is still saved locally at ./trained_model")
            print("You can push manually later with:")
            print(f"  huggingface-cli login && huggingface-cli upload {SAVE_REPO} ./trained_model")
    elif SAVE_REPO:
        print(f"\n[INFO] Skipping HF push — HF_TOKEN not set or not a valid HF token.")
        print(f"  Get one from: https://huggingface.co/settings/tokens")
        print(f"  Set: os.environ['HF_TOKEN'] = 'hf_...'")

    print(f"\n{'='*60}")
    print(f"  Before training:  {score_before:.3f}")
    print(f"  After training:   {score_after:.3f}")
    print(f"  Improvement:      {score_after - score_before:+.3f}")
    print(f"  Curriculum level: {curriculum_level}")
    print(f"  Hack detections:  {hack_count}")
    print(f"{'='*60}")

    return model, tokenizer, score_before, score_after


if __name__ == "__main__":
    train()
