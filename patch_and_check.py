"""Patch README baseline scores and run all validation checks."""
import re, json, requests, os, subprocess

# ── 1. Patch README ───────────────────────────────────────────────────────────
readme = open("README.md").read()
lines = readme.split("\n")

# Find the table and replace from line 145 to 163
start = next(i for i, l in enumerate(lines) if "| Task | Name | Mean Score |" in l)
end = next(i for i in range(start, len(lines)) if lines[i].startswith("**Overall Mean Score"))

new_table = [
    "| Task | Name | Mean Score |",
    "|------|------|-----------|",
    "| 1 | Bug Detection (Easy) | 0.400 |",
    "| 2 | Bug Classification (Medium) | 0.600 |",
    "| 3 | Full Code Review (Hard) | 0.566 |",
    "| 4 | Code Smell Detection (Medium) | 0.360 |",
    "| 5 | Security Audit (Hard) | 0.684 |",
    "| 6 | Performance Optimization (Hard) | 0.640 |",
    "| 7 | Test Coverage Review (Hard) | 0.380 |",
    "| 8 | Refactoring Opportunity Detection (Medium) | 1.000 |",
    "| 9 | SOLID Principles Violation Detection (Hard) | 1.000 |",
    "| 10 | Error Handling Review (Medium) | 0.500 |",
    "| 11 | Documentation Quality Review (Easy-Medium) | 1.000 |",
    "| 12 | Concurrency & Race Condition Detection (Hard) | 0.850 |",
    "| 13 | API Design Review (Hard) | 0.350 |",
    "| 14 | Code Comparison Review (Hard) | 0.942 |",
    "| 15 | Dependency & Import Review (Medium) | 0.518 |",
    "",
    "**Overall Mean Score: 0.653**",
]

new_lines = lines[:start] + new_table + lines[end + 1:]
open("README.md", "w").write("\n".join(new_lines))
print("✅ README updated with 15-task scores")

# ── 2. Run all validation checks ─────────────────────────────────────────────
BASE = "https://dharaneswarreddy-codereview-env.hf.space"
checks = []

# File checks
for f in ["inference.py", "Dockerfile", "openenv.yaml", "README.md", "requirements.txt"]:
    checks.append((os.path.exists(f), f + " exists"))

# Snippet check (we have 120 now)
s = json.load(open("env/data/snippets.json"))
checks.append((len(s) >= 100, str(len(s)) + " snippets (≥100)"))

# Live endpoint checks
print("\n=== LIVE ENDPOINT CHECKS ===")
try:
    r = requests.get(f"{BASE}/health", timeout=15)
    print(f"/health → {r.status_code}: {r.text}")
    checks.append((r.status_code == 200, "health 200"))

    r = requests.get(f"{BASE}/tasks", timeout=15)
    tasks = r.json()
    print(f"/tasks → {len(tasks)} tasks: {[t['name'] for t in tasks]}")
    checks.append((len(tasks) >= 7, str(len(tasks)) + " tasks (≥7)"))

    r = requests.post(f"{BASE}/reset", json={"task_level": 1}, timeout=15)
    obs = r.json()
    print(f"/reset → snippet_id: {obs.get('snippet_id')}")
    checks.append(("snippet_id" in obs, "reset returns observation"))

    r = requests.post(f"{BASE}/step", json={
        "has_bug": True, "bug_type": "logic_error",
        "severity": "high", "suggested_fix": "Add null check before dividing to prevent ZeroDivisionError"
    }, timeout=15)
    d = r.json()
    score = d["reward"]["score"]
    print(f"/step → score={score}, feedback={d['reward']['feedback'][:60]}")
    checks.append((0.0 <= score <= 1.0, "score=" + str(round(score, 3)) + " in range"))

    r = requests.get(f"{BASE}/state", timeout=15)
    state = r.json()
    print(f"/state → task_level={state['task_level']}, done={state['episode_done']}")
    checks.append(("task_level" in state, "state has task_level"))

except Exception as e:
    checks.append((False, "Live endpoint error: " + str(e)))

# No hardcoded real tokens
for f in ["inference.py", "app.py"]:
    c = open(f).read()
    has_real_token = bool(re.search(r"hf_[A-Za-z0-9]{20,}", c))
    checks.append((not has_real_token, "no hardcoded key in " + f))

print("\n" + "=" * 50)
passed = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    print(("PASS" if ok else "FAIL") + " - " + name)
print("=" * 50)
print(str(passed) + "/" + str(len(checks)) + " passed")
if passed == len(checks):
    print("READY TO SUBMIT ✅")
else:
    print("FIX FAILURES FIRST ❌")
