import sys

def verify():
    import traceback
    try:
        from env.models import PRAction, PRObservation, PRStepResponse, PRState
        print("models imported")
        import env.pr_graders
        print("pr_graders imported")
        import env.pr_environment
        print("pr_environment imported")
        import app
        print("app imported")
        import inference
        print("inference imported")
        import training.train_grpo
        print("train_grpo imported")
        print("ALL_OK")
        
        # Test PRReviewEnv
        from env.pr_environment import PRReviewEnv
        from env.models import PRAction
        env = PRReviewEnv()
        obs = env.reset()
        print(f"PR env reset: {obs.pr_title}")
        print("END_TO_END_OK")
    except Exception as e:
        print("ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    verify()
