import os
from huggingface_hub import HfApi, create_repo

TOKEN = "HF_TOKEN_PLACEHOLDER"

ORG = "electricsheepafrica"
REPO_NAME = "chewie-llama-3b"
REPO_ID = f"{ORG}/{REPO_NAME}"

MODEL_DIR = "africhw-llama-3.2-3b"

def deploy_model():
    print(f"🚀 Preparing to deploy model to {REPO_ID}...")
    api = HfApi(token=TOKEN)
    
    try:
        # 1. Create Repo (if not exists)
        print("📦 Ensuring repository exists...")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)
        
        # 2. Upload Folder
        print(f"📤 Uploading model directory from {MODEL_DIR}...")
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "*.bin"] # Ignoring bin if safetensors exist, and extensive checkpoints to save space/time
        )
        
        print(f"\n✅ Model successfully deployed to: https://huggingface.co/{REPO_ID}")
        
    except Exception as e:
        print(f"❌ Error during deployment: {e}")

if __name__ == "__main__":
    deploy_model()
