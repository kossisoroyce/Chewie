import os
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
REPO_ID = "electricsheepafrica/chewie-instruct"

VAL_PATH = "data/training_bundle/val.json"
README_PATH = "data/training_bundle/README.md"

def deploy_missing():
    print(f"🚀 Fixing deployment for {REPO_ID}...")
    api = HfApi(token=TOKEN)
    
    try:
        # Upload Readme
        print(f"📄 Uploading {README_PATH}...")
        api.upload_file(
            path_or_fileobj=README_PATH,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        # Upload Val
        print(f"📤 Uploading {VAL_PATH}...")
        api.upload_file(
            path_or_fileobj=VAL_PATH,
            path_in_repo="val.json",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        
        print(f"\n✅ README and Validation set pushed successfully!")
        
    except Exception as e:
        print(f"❌ Error during deployment: {e}")

if __name__ == "__main__":
    deploy_missing()
