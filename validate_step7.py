import json
import logging
logging.basicConfig(level=logging.INFO)

# Validate snippet count
data = json.load(open('env/data/snippets.json'))
print(f'Total snippets: {len(data)}')
assert len(data) == 100, f'Expected 100, got {len(data)}'

# Check all new snippets have new fields
new_fields = ['code_smells', 'owasp_issues', 'time_complexity', 'performance_issue', 'is_testable', 'missing_tests']
for i, s in enumerate(data[30:], 31):
    for field in new_fields:
        assert field in s, f'snippet_{i:03d} missing field: {field}'
print('All 100 snippets valid with new fields')

# Test all 7 task levels work
from env.environment import CodeReviewEnv
from env.models import Action
from env.tasks import list_all_tasks

tasks = list_all_tasks()
print(f'Total tasks: {len(tasks)}')
assert len(tasks) == 7

import os
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-Coder-32B-Instruct"

for task in tasks:
    env = CodeReviewEnv(task_level=task.level)
    obs = env.reset()
    action = Action(
        has_bug=True,
        bug_type='logic_error',
        severity='high',
        suggested_fix='Add null check, use parameterized queries, optimize with O(n log n) algorithm, test with boundary values and empty inputs'
    )
    _, reward, done, info = env.step(action)
    assert 0.0 <= reward.score <= 1.0
    print(f'Task {task.level} ({task.name}): score={reward.score:.3f} OK')

print('All 7 tasks working end-to-end')
