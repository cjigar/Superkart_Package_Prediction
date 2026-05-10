from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "cjigar/SuperKart-Package-Prediction"
repo_type = "space"

# Step 1: Ensure the Space exists before uploading
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' found.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating it now...")
    # 'sdk' is set to 'streamlit' as per your app.py logic
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="streamlit", private=False)
    print(f"Space '{repo_id}' created successfully.")

# Step 2: Upload the deployment folder
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo=""
)
print("Files uploaded successfully to Hugging Face Space.")
