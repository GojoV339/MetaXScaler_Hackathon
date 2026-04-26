import os
import requests
import json
import time

def print_result(name, success, info=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {name:<25} | {info}")

def run_health_checks():
    # URL of your HuggingFace space (from your .env)
    url = os.getenv("ENV_URL", "https://dharaneswarreddy-codereview-env.hf.space").rstrip("/")
    
    print(f"\n🚀 Running Health Checks against: {url}")
    print("-" * 65)

    # Check 1: Root / Docs Endpoint (FastAPI should have /docs)
    try:
        start = time.time()
        res = requests.get(f"{url}/docs", timeout=5)
        ping = round((time.time() - start) * 1000)
        if res.status_code == 200:
            print_result("Server Reachability", True, f"{ping}ms ping to /docs")
        else:
            print_result("Server Reachability", False, f"Status {res.status_code}")
    except Exception as e:
        print_result("Server Reachability", False, str(e))
        print("\n⚠️  Server is unreachable. Ensure your HuggingFace Space is running and built successfully.")
        return

    # Check 2: POST /reset (Task Level 1)
    obs = None
    try:
        res = requests.post(f"{url}/reset", json={"task_level": 1}, timeout=10)
        if res.status_code == 200:
            obs = res.json()
            if "code" in obs and "snippet_id" in obs:
                print_result("Snippet /reset Endpoint", True, f"Snippet ID: {obs['snippet_id']}")
            else:
                print_result("Snippet /reset Endpoint", False, "Missing 'code' in response")
        else:
            print_result("Snippet /reset Endpoint", False, f"Status {res.status_code} - {res.text}")
    except Exception as e:
        print_result("Snippet /reset Endpoint", False, str(e))

    # Check 3: POST /step (Snippet)
    if obs:
        try:
            action = {
                "has_bug": True,
                "bug_type": "security_vulnerability",
                "severity": "high",
                "suggested_fix": "Fix it"
            }
            res = requests.post(f"{url}/step", json=action, timeout=15)
            if res.status_code == 200:
                step_res = res.json()
                if "reward" in step_res:
                    score = step_res["reward"].get("score", "unknown")
                    print_result("Snippet /step Endpoint", True, f"Reward Score: {score}")
                else:
                    print_result("Snippet /step Endpoint", False, "No reward in response")
            else:
                print_result("Snippet /step Endpoint", False, f"Status {res.status_code} - {res.text}")
        except Exception as e:
            print_result("Snippet /step Endpoint", False, str(e))

    # Check 4: POST /pr/reset
    pr_obs = None
    try:
        res = requests.post(f"{url}/pr/reset", json={}, timeout=10)
        if res.status_code == 200:
            pr_obs = res.json()
            if "pr_title" in pr_obs:
                print_result("PR /pr/reset Endpoint", True, f"PR: {pr_obs['pr_id']}")
            else:
                print_result("PR /pr/reset Endpoint", False, "Missing 'pr_title' in response")
        else:
            print_result("PR /pr/reset Endpoint", False, f"Status {res.status_code} - {res.text}")
    except Exception as e:
        print_result("PR /pr/reset Endpoint", False, str(e))

    # Check 5: POST /pr/step
    if pr_obs:
        try:
            pr_action = {
                "inline_comments": [
                    {
                        "file": pr_obs.get("files", [{"filename": "unknown"}])[0].get("filename"),
                        "line": 10,
                        "comment": "Test comment",
                        "suggestion": "Test fix"
                    }
                ],
                "verdict": "REQUEST_CHANGES",
                "summary": "Health check summary"
            }
            res = requests.post(f"{url}/pr/step", json=pr_action, timeout=15)
            if res.status_code == 200:
                step_res = res.json()
                if "reward" in step_res:
                    score = step_res["reward"].get("score", "unknown")
                    print_result("PR /pr/step Endpoint", True, f"Reward Score: {score}")
                else:
                    print_result("PR /pr/step Endpoint", False, "No reward in response")
            else:
                print_result("PR /pr/step Endpoint", False, f"Status {res.status_code} - {res.text}")
        except Exception as e:
            print_result("PR /pr/step Endpoint", False, str(e))
            
    print("-" * 65)
    print("Health Check Complete!\n")

if __name__ == "__main__":
    run_health_checks()
