import requests, os, json

BASE = 'https://dharaneswarreddy-codereview-env.hf.space'
checks = []

# File checks
for f in ['inference.py', 'Dockerfile', 'openenv.yaml', 'README.md', 'requirements.txt']:
    checks.append((os.path.exists(f), f + ' exists'))

s = json.load(open('env/data/snippets.json'))
checks.append((len(s) == 100, str(len(s)) + '/100 snippets'))

# Live endpoint checks
try:
    checks.append((requests.get(f'{BASE}/health').status_code == 200, 'health 200'))
    tasks = requests.get(f'{BASE}/tasks').json()
    checks.append((len(tasks) == 7, str(len(tasks)) + '/7 tasks'))
    r = requests.post(f'{BASE}/reset', json={'task_level': 1})
    obs = r.json()
    checks.append(('snippet_id' in obs, 'reset returns observation'))
    r2 = requests.post(f'{BASE}/step', json={
        'has_bug': True,
        'bug_type': 'logic_error',
        'severity': 'high',
        'suggested_fix': 'Add null check before division'
    })
    score = r2.json()['reward']['score']
    checks.append((0.0 <= score <= 1.0, 'score=' + str(round(score, 3)) + ' in range'))
except Exception as e:
    checks.append((False, 'Live endpoint error: ' + str(e)))

# No hardcoded secrets
for f in ['inference.py', 'app.py']:
    c = open(f).read()
    checks.append(('hf_ggt' not in c and 'hf_' not in c.lower(), 'no hardcoded HF key in ' + f))

print('=' * 45)
passed = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    status = 'PASS' if ok else 'FAIL'
    print(status + '  ' + name)
print('=' * 45)
print(str(passed) + '/' + str(len(checks)) + ' passed')
if passed == len(checks):
    print('READY TO SUBMIT')
else:
    print('FIX FAILURES FIRST')
