import os
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")
REPO_ID = "electricsheepafrica/chewie-llama-3b"
MODEL_DIR = "africhw-llama-3.2-3b"

FILES_TO_UPLOAD = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md"
]

def deploy_individual():
    print(f"🚀 Deploying files individually to {REPO_ID}...")
    api = HfApi(token=TOKEN)
    
    for filename in FILES_TO_UPLOAD:
        file_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(file_path):
            print(f"📤 Uploading {filename}...")
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                print(f"✅ {filename} uploaded.")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

if __name__ == "__main__":
    deploy_individual()
