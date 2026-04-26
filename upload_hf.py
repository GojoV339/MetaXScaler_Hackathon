import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

def deploy_to_hf():
    print("Loading HuggingFace credentials...")
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in .env file.")
        return
        
    print("Initiating direct API upload to HuggingFace (bypassing Git LFS)...")
    api = HfApi(token=token)
    repo_id = "DharaneswarReddy/codereview-env"
    
    # We upload the folder, deliberately ignoring the heavy image files in res/
    # and local training data/notebooks that the environment doesn't need to run.
    print(f"Syncing local files to Space: {repo_id}...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git/*", 
            "training/*",      # Excludes local training scripts
            ".venv/*",         # EXTREMELY IMPORTANT: Excludes the virtual environment
            "__pycache__/*", 
            "*.pyc",
            "upload_hf.py",
            ".env"             # Make sure we don't accidentally push the secret .env
        ]
    )
    print("✅ Deploy completed successfully! The Space is now building.")

if __name__ == "__main__":
    deploy_to_hf()
