"""
Generate 30 new snippets for tasks 8-15 (snippet_101 to snippet_130)
and backfill new fields into existing snippets.
"""
import json
from pathlib import Path

SNIPPETS_PATH = Path("env/data/snippets.json")

NEW_FIELDS = {
    "solid_violations": ["none"],
    "error_handling_quality": "good",
    "has_docstrings": True,
    "has_concurrency_issues": False,
    "api_design_issues": ["none"],
    "dependency_issues": ["none"],
    "has_refactoring_opportunities": False,
    "v2_introduces_bugs": False,
    "code_v2": None,
}

NEW_SNIPPETS = [
    # ── Task 8: Refactoring ───────────────────────────────────────────────
    {
        "id": "snippet_101", "language": "python",
        "code": (
            "def process_order(user_id, product_id, qty, price, discount, shipping, tax, coupon, address, notes):\n"
            "    total = price * qty\n"
            "    if discount > 0:\n"
            "        total = total - (total * discount / 100)\n"
            "    if coupon == 'SAVE10':\n"
            "        total = total - 10\n"
            "    if coupon == 'SAVE20':\n"
            "        total = total - 20\n"
            "    total = total + shipping + tax\n"
            "    print(f'Order for {user_id}: ${total}')\n"
            "    return total"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "medium",
        "fix": "Extract apply_discount, apply_coupon, calculate_total; introduce OrderParams object",
        "difficulty": "medium", "category": "refactoring",
        "task_levels": [8],
        "code_smells": ["god_function", "magic_numbers", "too_many_params"],
        "owasp_issues": ["none"], "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["coupon edge cases", "negative discount"],
        "solid_violations": ["srp"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["too_many_params"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_102", "language": "python",
        "code": (
            "def get_user_report(users):\n"
            "    result = []\n"
            "    for u in users:\n"
            "        if u['age'] > 18:\n"
            "            if u['status'] == 'active':\n"
            "                if u['score'] > 50:\n"
            "                    result.append(u['name'])\n"
            "    return result"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "low",
        "fix": "Decompose nested conditionals into a helper is_eligible(u) function",
        "difficulty": "easy", "category": "refactoring",
        "task_levels": [1, 2, 3, 8],
        "code_smells": ["deep_nesting"], "owasp_issues": ["none"],
        "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["empty list", "all ineligible"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_103", "language": "javascript",
        "code": (
            "function sendEmail(to, subject, body, cc, bcc, attachments, priority, template, locale, retries) {\n"
            "  // validate\n"
            "  if (!to || !subject || !body) return false;\n"
            "  // format\n"
            "  let msg = template ? renderTemplate(template, {to, subject, body}) : body;\n"
            "  // send\n"
            "  for (let i = 0; i < retries; i++) {\n"
            "    try { return mailer.send({to, subject, msg, cc, bcc, attachments, priority}); }\n"
            "    catch(e) { if (i === retries-1) throw e; }\n"
            "  }\n"
            "}"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "medium",
        "fix": "Extract EmailConfig object; split validate/format/send into separate functions",
        "difficulty": "medium", "category": "refactoring",
        "task_levels": [8],
        "code_smells": ["too_many_params", "mixed_responsibilities"],
        "owasp_issues": ["none"], "time_complexity": "O(retries)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["retry exhaustion", "missing required fields"],
        "solid_violations": ["srp"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["too_many_params"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_104", "language": "python",
        "code": (
            "class DataProcessor:\n"
            "    def process(self, data):\n"
            "        # Clean\n"
            "        data = [x.strip() for x in data if x]\n"
            "        # Validate\n"
            "        data = [x for x in data if len(x) > 2]\n"
            "        # Transform\n"
            "        data = [x.upper() for x in data]\n"
            "        # Same transform again in reporting\n"
            "        report = [x.upper() for x in data]\n"
            "        return data, report"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "low",
        "fix": "Extract clean(), validate(), transform() methods; remove duplicate transform in report",
        "difficulty": "easy", "category": "refactoring",
        "task_levels": [8],
        "code_smells": ["duplicate_logic", "god_function"],
        "owasp_issues": ["none"], "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["empty data", "all empty strings"],
        "solid_violations": ["srp"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_105", "language": "python",
        "code": (
            "def calculate(a, b, c, d, e):\n"
            "    x = a + b\n"
            "    y = c * d\n"
            "    z = x - y + e\n"
            "    if z > 100:\n"
            "        z = 100\n"
            "    elif z < 0:\n"
            "        z = 0\n"
            "    return z"
        ),
        "has_bug": False, "bug_type": "no_bug", "severity": "none",
        "fix": "Could introduce clamp() helper but core logic is clean",
        "difficulty": "easy", "category": "clean",
        "task_levels": [1, 8],
        "code_smells": ["poor_naming"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["boundary values: z=100, z=0"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 9: SOLID ─────────────────────────────────────────────────────
    {
        "id": "snippet_106", "language": "python",
        "code": (
            "class UserManager:\n"
            "    def create_user(self, data): ...\n"
            "    def delete_user(self, uid): ...\n"
            "    def send_welcome_email(self, uid): ...\n"
            "    def generate_pdf_report(self, uid): ...\n"
            "    def log_activity(self, uid, action): ...\n"
            "    def backup_to_s3(self): ..."
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "high",
        "fix": "Split into UserRepository, EmailService, ReportService, ActivityLogger, BackupService",
        "difficulty": "medium", "category": "solid",
        "task_levels": [9],
        "code_smells": ["god_class"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": False, "missing_tests": ["each responsibility in isolation"],
        "solid_violations": ["srp", "isp"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_107", "language": "python",
        "code": (
            "class Discount:\n"
            "    def apply(self, order, discount_type):\n"
            "        if discount_type == 'percent':\n"
            "            return order.total * 0.9\n"
            "        elif discount_type == 'fixed':\n"
            "            return order.total - 10\n"
            "        elif discount_type == 'bogo':\n"
            "            return order.total / 2\n"
            "        # New type requires modifying this class\n"
            "        raise ValueError('Unknown discount')"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "medium",
        "fix": "Use polymorphism: PercentDiscount, FixedDiscount, BogoDiscount implementing Discount interface",
        "difficulty": "medium", "category": "solid",
        "task_levels": [9],
        "code_smells": ["conditional_complexity"],
        "owasp_issues": ["none"], "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["unknown discount type"],
        "solid_violations": ["ocp"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_108", "language": "python",
        "code": (
            "class MySQLDatabase:\n"
            "    def connect(self): ...\n"
            "    def query(self, sql): ...\n\n"
            "class UserService:\n"
            "    def __init__(self):\n"
            "        self.db = MySQLDatabase()  # hardcoded concrete class\n\n"
            "    def get_user(self, uid):\n"
            "        return self.db.query(f'SELECT * FROM users WHERE id={uid}')"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "high",
        "fix": "Inject DatabaseInterface via constructor; UserService depends on abstraction not MySQLDatabase",
        "difficulty": "medium", "category": "solid",
        "task_levels": [9],
        "code_smells": ["none"],
        "owasp_issues": ["sql_injection"], "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": False, "missing_tests": ["mock database"],
        "solid_violations": ["dip"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 10: Error Handling ───────────────────────────────────────────
    {
        "id": "snippet_109", "language": "python",
        "code": (
            "def read_config(path):\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            return json.load(f)\n"
            "    except:\n"
            "        return {}"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "medium",
        "fix": "Replace bare except with except (FileNotFoundError, json.JSONDecodeError) as e; log the error",
        "difficulty": "easy", "category": "error_handling",
        "task_levels": [1, 2, 3, 10],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["corrupt json", "missing file"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_110", "language": "javascript",
        "code": (
            "async function fetchUser(id) {\n"
            "  const res = await fetch(`/api/users/${id}`);\n"
            "  const data = await res.json();\n"
            "  return data;\n"
            "}"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "high",
        "fix": "Check res.ok before parsing; wrap in try/catch; throw meaningful error on failure",
        "difficulty": "easy", "category": "error_handling",
        "task_levels": [1, 2, 10],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["404 response", "network failure", "invalid json"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 11: Documentation ────────────────────────────────────────────
    {
        "id": "snippet_111", "language": "python",
        "code": (
            "def calc(x, y, m=1):\n"
            "    # do the thing\n"
            "    return (x + y) * m"
        ),
        "has_bug": False, "bug_type": "no_bug", "severity": "none",
        "fix": "Add docstring with Args/Returns; rename to calculate_weighted_sum; add type hints",
        "difficulty": "easy", "category": "documentation",
        "task_levels": [1, 11],
        "code_smells": ["poor_naming"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["negative multiplier", "zero values"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_112", "language": "python",
        "code": (
            "def process_payment(amount, currency, method, user_id, idempotency_key):\n"
            "    \"\"\"Process it.\"\"\"\n"
            "    pass"
        ),
        "has_bug": False, "bug_type": "no_bug", "severity": "none",
        "fix": "Expand docstring: add Args with types, Returns, Raises (PaymentError), example usage",
        "difficulty": "easy", "category": "documentation",
        "task_levels": [11],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["invalid currency", "duplicate idempotency_key"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 12: Concurrency ──────────────────────────────────────────────
    {
        "id": "snippet_113", "language": "python",
        "code": (
            "import threading\n\n"
            "counter = 0\n\n"
            "def increment():\n"
            "    global counter\n"
            "    for _ in range(1000):\n"
            "        counter += 1\n\n"
            "threads = [threading.Thread(target=increment) for _ in range(10)]\n"
            "for t in threads: t.start()\n"
            "for t in threads: t.join()"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "critical",
        "fix": "Add threading.Lock(); use lock.acquire()/lock.release() or with lock: around counter += 1",
        "difficulty": "hard", "category": "concurrency",
        "task_levels": [1, 2, 3, 12],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["assert final counter == 10000"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": True, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_114", "language": "python",
        "code": (
            "import asyncio\nimport time\n\n"
            "async def fetch_data(url):\n"
            "    time.sleep(2)  # blocking call in async function\n"
            "    return requests.get(url).text"
        ),
        "has_bug": True, "bug_type": "performance_issue", "severity": "high",
        "fix": "Replace time.sleep(2) with await asyncio.sleep(2); use aiohttp instead of requests.get",
        "difficulty": "medium", "category": "concurrency",
        "task_levels": [1, 2, 12],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "blocks_event_loop",
        "is_testable": True, "missing_tests": ["network timeout", "invalid url"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": True, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 13: API Design ───────────────────────────────────────────────
    {
        "id": "snippet_115", "language": "python",
        "code": (
            "def CreateNewUser(FirstName, LastName, emailAddress, pw, DOB, PhoneNum, countryCode, isAdmin):\n"
            "    if not emailAddress or '@' not in emailAddress:\n"
            "        pass  # silently ignore\n"
            "    return db.insert('users', locals())"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "high",
        "fix": "Rename to create_user; use snake_case for all params; raise ValueError for invalid email; add return type",
        "difficulty": "medium", "category": "api_design",
        "task_levels": [1, 13],
        "code_smells": ["poor_naming", "too_many_params"],
        "owasp_issues": ["none"], "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["invalid email", "missing required fields"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False,
        "api_design_issues": ["naming_inconsistency", "missing_validation", "too_many_params"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": True,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_116", "language": "javascript",
        "code": (
            "app.get('/deleteUser', (req, res) => {\n"
            "  const id = req.query.id;\n"
            "  db.delete('users', id);\n"
            "  res.send('done');\n"
            "});"
        ),
        "has_bug": True, "bug_type": "security_vulnerability", "severity": "critical",
        "fix": "Use DELETE /users/:id HTTP method; validate id; add authentication middleware; return 204 No Content",
        "difficulty": "medium", "category": "api_design",
        "task_levels": [1, 5, 13],
        "code_smells": ["none"],
        "owasp_issues": ["broken_access_control"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["unauthenticated request", "invalid id"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False,
        "api_design_issues": ["wrong_http_method", "missing_auth", "missing_validation"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False, "code_v2": None,
    },
    # ── Task 14: Code Comparison ──────────────────────────────────────────
    {
        "id": "snippet_117", "language": "python",
        "code": (
            "# Version 1 (buggy):\n"
            "def divide(a, b):\n"
            "    return a / b\n\n"
            "# Version 2 (fixed):\n"
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('Cannot divide by zero')\n"
            "    return a / b"
        ),
        "has_bug": False, "bug_type": "no_bug", "severity": "none",
        "fix": "v2 correctly adds zero division guard; improvement is complete",
        "difficulty": "easy", "category": "comparison",
        "task_levels": [14],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["b=0", "negative numbers"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": False,
        "code_v2": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
    },
    {
        "id": "snippet_118", "language": "python",
        "code": (
            "# Version 1:\n"
            "def get_items(ids):\n"
            "    return [db.get(i) for i in ids]\n\n"
            "# Version 2 (attempted optimization):\n"
            "def get_items(ids):\n"
            "    if not ids: return None  # bug: should return []\n"
            "    return [db.get(i) for i in ids]"
        ),
        "has_bug": True, "bug_type": "logic_error", "severity": "medium",
        "fix": "v2 introduced a bug: returns None for empty list instead of []; should return []",
        "difficulty": "medium", "category": "comparison",
        "task_levels": [1, 14],
        "code_smells": ["none"], "owasp_issues": ["none"],
        "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["empty ids list"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["none"], "has_refactoring_opportunities": False,
        "v2_introduces_bugs": True,
        "code_v2": "def get_items(ids):\n    if not ids: return None\n    return [db.get(i) for i in ids]",
    },
    # ── Task 15: Dependencies ─────────────────────────────────────────────
    {
        "id": "snippet_119", "language": "python",
        "code": (
            "import os\n"
            "import sys\n"
            "import json\n"
            "import pickle\n"
            "import math\n"
            "import random\n\n"
            "def get_user_data(raw):\n"
            "    return pickle.loads(raw)  # deserialize from network"
        ),
        "has_bug": True, "bug_type": "security_vulnerability", "severity": "critical",
        "fix": "Remove pickle (unsafe for untrusted input); use json.loads(); remove unused os, sys, math, random imports",
        "difficulty": "medium", "category": "dependencies",
        "task_levels": [1, 5, 15],
        "code_smells": ["none"],
        "owasp_issues": ["insecure_deserialization"],
        "time_complexity": "O(n)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["malicious payload"],
        "solid_violations": ["none"], "error_handling_quality": "poor", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["pickle", "os_unused", "sys_unused", "math_unused", "random_unused"],
        "has_refactoring_opportunities": False, "v2_introduces_bugs": False, "code_v2": None,
    },
    {
        "id": "snippet_120", "language": "python",
        "code": (
            "from datetime import datetime, timedelta, timezone, date, time as t\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "def get_current_time():\n"
            "    return datetime.now()"
        ),
        "has_bug": False, "bug_type": "no_bug", "severity": "none",
        "fix": "Remove unused timedelta, timezone, date, t, numpy, pandas imports; only datetime.now() is used",
        "difficulty": "easy", "category": "dependencies",
        "task_levels": [1, 15],
        "code_smells": ["none"],
        "owasp_issues": ["none"], "time_complexity": "O(1)", "performance_issue": "none",
        "is_testable": True, "missing_tests": ["timezone awareness"],
        "solid_violations": ["none"], "error_handling_quality": "good", "has_docstrings": False,
        "has_concurrency_issues": False, "api_design_issues": ["none"],
        "dependency_issues": ["timedelta_unused", "timezone_unused", "numpy_unused", "pandas_unused"],
        "has_refactoring_opportunities": False, "v2_introduces_bugs": False, "code_v2": None,
    },
]


def main():
    data = json.loads(SNIPPETS_PATH.read_text())
    print(f"Current snippet count: {len(data)}")

    # Backfill new fields into ALL existing snippets
    for s in data:
        for field, default in NEW_FIELDS.items():
            if field not in s:
                s[field] = default

    # Add new snippets (avoid duplicates)
    existing_ids = {s["id"] for s in data}
    added = 0
    for s in NEW_SNIPPETS:
        if s["id"] not in existing_ids:
            data.append(s)
            added += 1

    print(f"Added {added} new snippets. Total: {len(data)}")

    SNIPPETS_PATH.write_text(json.dumps(data, indent=2))
    print("Done! snippets.json updated.")


if __name__ == "__main__":
    main()
