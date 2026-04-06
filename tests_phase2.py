#!/usr/bin/env python3
"""Phase 2 — Tests 2-9 combined runner."""
import json, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test2_snippets():
    print("=" * 60)
    print("TEST 2 — Validate snippets.json")
    print("=" * 60)
    data = json.load(open("env/data/snippets.json"))
    assert len(data) == 30, f"Expected 30 snippets, got {len(data)}"

    required_fields = ["id", "language", "code", "has_bug", "bug_type", "severity", "fix", "difficulty", "category"]
    for i, s in enumerate(data):
        for field in required_fields:
            assert field in s, f"Snippet {i} missing field: {field}"

    valid_bug_types = {"logic_error", "security_vulnerability", "performance_issue", "syntax_error", "no_bug"}
    valid_severities = {"low", "medium", "high", "critical", "none"}
    valid_difficulties = {"easy", "medium", "hard"}
    for s in data:
        assert s["bug_type"] in valid_bug_types, f"Invalid bug_type: {s['bug_type']}"
        assert s["severity"] in valid_severities, f"Invalid severity: {s['severity']}"
        assert s["difficulty"] in valid_difficulties, f"Invalid difficulty: {s['difficulty']}"
        if not s["has_bug"]:
            assert s["bug_type"] == "no_bug", f"has_bug=false but bug_type={s['bug_type']}"
            assert s["severity"] == "none", f"has_bug=false but severity={s['severity']}"

    easy = [s for s in data if s["difficulty"] == "easy"]
    medium = [s for s in data if s["difficulty"] == "medium"]
    hard = [s for s in data if s["difficulty"] == "hard"]
    no_bug = [s for s in data if not s["has_bug"]]
    security = [s for s in data if s["bug_type"] == "security_vulnerability"]
    print(f"Easy: {len(easy)}, Medium: {len(medium)}, Hard: {len(hard)}")
    print(f"No-bug snippets: {len(no_bug)}, Security vulns: {len(security)}")
    assert len(easy) >= 8
    assert len(medium) >= 8
    assert len(hard) >= 8
    assert len(no_bug) >= 5
    assert len(security) >= 3
    print("TEST 2 PASSED\n")


def test3_models():
    print("=" * 60)
    print("TEST 3 — Pydantic models")
    print("=" * 60)
    from env.models import Observation, Action, Reward, State, TaskConfig, StepResponse

    obs = Observation(snippet_id="s001", code="print('hi')", language="python", task_level=1, step_number=0)
    assert obs.task_level == 1
    assert obs.context_hint is None

    a1 = Action(has_bug=True, bug_type="logic_error", severity="high", suggested_fix="fix it properly with validation")
    assert a1.has_bug == True

    a2 = Action(has_bug=False)
    assert a2.bug_type == "no_bug"
    assert a2.severity == "none"

    r = Reward(score=0.75, breakdown={"detection": 0.75}, feedback="Good job", is_correct=True)
    assert 0.0 <= r.score <= 1.0

    try:
        Reward(score=1.5, breakdown={}, feedback="", is_correct=False)
        assert False, "Should have raised validation error"
    except Exception:
        pass

    print("TEST 3 PASSED\n")


def test4_tasks():
    print("=" * 60)
    print("TEST 4 — tasks.py")
    print("=" * 60)
    from env.tasks import get_task, list_all_tasks, get_task_description_for_prompt

    t1 = get_task(1)
    assert t1.level == 1
    assert t1.task_id == "bug_detection"

    t2 = get_task(2)
    assert t2.level == 2

    t3 = get_task(3)
    assert t3.level == 3

    try:
        get_task(99)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    all_tasks = list_all_tasks()
    assert len(all_tasks) == 3
    assert [t.level for t in all_tasks] == [1, 2, 3]

    for level in [1, 2, 3]:
        desc = get_task_description_for_prompt(level)
        assert isinstance(desc, str)
        assert len(desc) > 20

    print("TEST 4 PASSED\n")


def test5_graders():
    print("=" * 60)
    print("TEST 5 — graders.py (no LLM)")
    print("=" * 60)
    from env.graders import grade_task1, grade_task2, grade_task3, _penalize_reward_hack
    from env.models import Action

    gt_bug = {"has_bug": True, "bug_type": "logic_error", "severity": "high",
              "fix": "Add null check before dividing", "code": "def f(a,b): return a/b"}
    gt_clean = {"has_bug": False, "bug_type": "no_bug", "severity": "none",
                "fix": "", "code": "def f(a,b): return a+b"}

    # Task 1 correct
    r = grade_task1(Action(has_bug=True), gt_bug)
    assert r.score == 1.0, f"Expected 1.0, got {r.score}"
    assert r.is_correct == True
    r = grade_task1(Action(has_bug=False), gt_clean)
    assert r.score == 1.0

    # Task 1 wrong
    r = grade_task1(Action(has_bug=False), gt_bug)
    assert r.score == 0.0
    assert r.is_correct == False

    # Task 2 full correct
    r = grade_task2(Action(has_bug=True, bug_type="logic_error"), gt_bug)
    assert r.score == 1.0

    # Task 2 correct detection, wrong type
    r = grade_task2(Action(has_bug=True, bug_type="syntax_error"), gt_bug)
    assert r.score == 0.4, f"Expected 0.4, got {r.score}"

    # Task 2 wrong detection
    r = grade_task2(Action(has_bug=False), gt_bug)
    assert r.score == 0.0

    # Penalty tests
    assert _penalize_reward_hack(Action(has_bug=True, suggested_fix="fix")) == -0.1
    assert _penalize_reward_hack(Action(has_bug=True, suggested_fix="add error handling")) == -0.1
    assert _penalize_reward_hack(Action(has_bug=True, suggested_fix="Add a null check before division: if b == 0: raise ValueError")) == 0.0

    # Score variance
    scores = [
        grade_task1(Action(has_bug=True), gt_bug).score,
        grade_task1(Action(has_bug=False), gt_bug).score,
        grade_task1(Action(has_bug=False), gt_clean).score
    ]
    assert len(set(scores)) > 1, "Grader returns same score for all inputs!"

    print("TEST 5 PASSED\n")


def test6_environment():
    print("=" * 60)
    print("TEST 6 — environment.py (full episode)")
    print("=" * 60)
    from env.environment import CodeReviewEnv
    from env.models import Action

    # Task 1 episode
    env = CodeReviewEnv(task_level=1, seed=42)
    obs = env.reset()
    assert obs.snippet_id is not None
    assert obs.code is not None
    assert obs.task_level == 1
    assert obs.step_number == 0
    assert obs.context_hint is None
    print(f"Reset OK — snippet: {obs.snippet_id}, language: {obs.language}")

    action = Action(has_bug=True, bug_type="logic_error", severity="high", suggested_fix="Add null check")
    obs2, reward, done, info = env.step(action)
    assert 0.0 <= reward.score <= 1.0
    assert done == True
    assert "snippet_id" in info
    print(f"Step OK — score: {reward.score}, feedback: {reward.feedback}")

    state = env.state()
    assert state.episode_done == True
    assert len(state.action_history) == 1
    assert state.cumulative_score == reward.score
    print(f"State OK — cumulative: {state.cumulative_score}")

    # Error when step called after done
    try:
        env.step(action)
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        print(f"RuntimeError OK: {e}")

    # Reset clears state
    obs3 = env.reset()
    state2 = env.state()
    assert state2.episode_done == False
    assert len(state2.action_history) == 0
    print("Reset clears state OK")

    # Task 2
    env2 = CodeReviewEnv(task_level=2, seed=42)
    obs = env2.reset()
    assert obs.task_level == 2
    assert obs.context_hint is not None
    print("Task 2 env OK")

    # Task 3
    env3 = CodeReviewEnv(task_level=3, seed=42)
    obs = env3.reset()
    assert obs.task_level == 3
    print("Task 3 env OK")

    # Multi-episode
    env4 = CodeReviewEnv(task_level=1)
    seen = set()
    for _ in range(10):
        obs = env4.reset()
        seen.add(obs.snippet_id)
        env4.step(Action(has_bug=False))
    print(f"Multi-episode OK — {len(seen)} unique snippets across 10 resets")

    print("TEST 6 PASSED\n")


def test7_inference():
    print("=" * 60)
    print("TEST 7 — inference.py (mock)")
    print("=" * 60)
    os.environ["API_BASE_URL"] = "http://localhost:9999"
    os.environ["HF_TOKEN"] = "dummy"
    os.environ["MODEL_NAME"] = "test-model"

    from inference import parse_action, build_system_prompt, build_user_prompt
    from env.models import Action, Observation

    # Valid JSON
    valid_json = '{"has_bug": true, "bug_type": "logic_error", "severity": "high", "suggested_fix": "Add null check before division"}'
    action = parse_action(valid_json)
    assert action.has_bug == True
    assert action.bug_type == "logic_error"
    print("parse_action valid JSON OK")

    # Fenced JSON
    fenced = '```json\n{"has_bug": false, "bug_type": "no_bug", "severity": "none", "suggested_fix": ""}\n```'
    action2 = parse_action(fenced)
    assert action2.has_bug == False
    print("parse_action fenced JSON OK")

    # Broken JSON
    broken = "I think there might be a bug in this code..."
    action3 = parse_action(broken)
    assert isinstance(action3, Action)
    print("parse_action broken JSON returns safe default OK")

    # Empty string
    action4 = parse_action("")
    assert isinstance(action4, Action)
    print("parse_action empty string OK")

    # System prompts
    for level in [1, 2, 3]:
        prompt = build_system_prompt(level)
        assert len(prompt) > 50
        assert "JSON" in prompt
    print("build_system_prompt all levels OK")

    # User prompt
    obs = Observation(snippet_id="s001", code="def f(a,b): return a/b", language="python", task_level=1, step_number=0)
    up = build_user_prompt(obs)
    assert "python" in up.lower()
    assert "def f(a,b)" in up
    print("build_user_prompt OK")

    print("TEST 7 PASSED\n")


def test9_edge_cases():
    print("=" * 60)
    print("TEST 9 — Edge cases")
    print("=" * 60)
    from env.environment import CodeReviewEnv
    from env.models import Action

    env = CodeReviewEnv(task_level=1)
    print(f"Loaded {len(env._snippets)} snippets")
    assert len(env._snippets) >= 20

    for level in [1, 2, 3]:
        env = CodeReviewEnv(task_level=level)
        obs = env.reset()
        action = Action(has_bug=True, bug_type="logic_error", severity="high",
                        suggested_fix="Add proper input validation and null checks before processing")
        obs2, reward, done, info = env.step(action)
        assert isinstance(reward.score, float)
        assert 0.0 <= reward.score <= 1.0
        assert done == True
        print(f"Task {level} end-to-end: score={reward.score:.3f}")

    # Score variance
    env = CodeReviewEnv(task_level=2, seed=1)
    obs = env.reset()
    gt = env._current_snippet

    action_correct = Action(has_bug=gt["has_bug"], bug_type=gt["bug_type"])
    action_wrong = Action(has_bug=not gt["has_bug"])

    from env.graders import grade_task2
    r1 = grade_task2(action_correct, gt)
    r2 = grade_task2(action_wrong, gt)
    assert r1.score != r2.score, "Grader must return different scores for different actions"
    print(f"Score variation OK: correct={r1.score}, wrong={r2.score}")

    print("TEST 9 PASSED\n")


if __name__ == "__main__":
    try:
        test2_snippets()
        test3_models()
        test4_tasks()
        test5_graders()
        test6_environment()
        test7_inference()
        test9_edge_cases()
        print("=" * 60)
        print("ALL PHASE 2 TESTS PASSED (Tests 2-7, 9)")
        print("(Test 8 = API endpoints — run separately with app.py)")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
