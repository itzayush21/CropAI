# Install Hugging Face CLI


# Download the model locally
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="./minilm_model")
