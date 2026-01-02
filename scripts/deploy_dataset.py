import os
from huggingface_hub import HfApi, create_repo

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
ORG = "electricsheepafrica"
REPO_NAME = "chewie-instruct"
REPO_ID = f"{ORG}/{REPO_NAME}"

DATASET_PATH = "data/training_bundle/train.json"
VAL_PATH = "data/training_bundle/val.json"
README_PATH = "data/training_bundle/README.md"

def deploy():
    print(f"🚀 Preparing to deploy to {REPO_ID}...")
    api = HfApi(token=TOKEN)
    
    try:
        # 1. Create Repo (if not exists)
        print("📦 Ensuring repository exists...")
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=TOKEN)
        
        # 2. Upload Dataset
        print(f"📤 Uploading {DATASET_PATH}...")
        api.upload_file(
            path_or_fileobj=DATASET_PATH,
            path_in_repo="train.json",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # 2b. UploadVal
        print(f"📤 Uploading {VAL_PATH}...")
        api.upload_file(
            path_or_fileobj=VAL_PATH,
            path_in_repo="val.json",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # 3. Upload Readme
        print(f"📄 Uploading {README_PATH}...")
        api.upload_file(
            path_or_fileobj=README_PATH,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        print(f"\n✅ Successfully deployed to: https://huggingface.co/datasets/{REPO_ID}")
        
    except Exception as e:
        print(f"❌ Error during deployment: {e}")

if __name__ == "__main__":
    deploy()
