"""
Upload the fine-tuned FLAN-T5 LoRA model to Hugging Face Hub.

Usage:
    1. pip install huggingface_hub
    2. huggingface-cli login          # paste your HF token (write access)
    3. python upload_to_hf.py

Update HF_USERNAME and REPO_NAME below before running.
"""

from huggingface_hub import HfApi, create_repo

# ── CHANGE THESE ───────────────────────────────────────────────────────────────
HF_USERNAME = "Chanu2003"          # e.g. "johndoe"
REPO_NAME = "flan-t5-dialogue-summarizer" # repo will be <username>/<repo_name>
MODEL_DIR = "./flan-t5-lora-dialogue-summary"
# ───────────────────────────────────────────────────────────────────────────────

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

# Create the repo on HF Hub (if it doesn't exist)
create_repo(repo_id, repo_type="model", exist_ok=True)

# Upload all files from the saved model directory
api = HfApi()
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=repo_id,
    repo_type="model",
)

print(f"\n✅ Model uploaded to: https://huggingface.co/{repo_id}")
print(f"   Use this repo_id in your Gradio app: \"{repo_id}\"")
