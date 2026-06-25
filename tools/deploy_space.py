"""Create + push the FastAPI backend to a Hugging Face Docker Space (programmatic deploy).

No MCP / no web-UI clicking: uses huggingface_hub with your WRITE token. Auth first with
either `HF_TOKEN=hf_xxx` in the env or `.venv/Scripts/hf auth login`, then run:
    .venv/Scripts/python tools/deploy_space.py [space-name]

Uploads only the deployable unit (Dockerfile, requirements, alpr/, web/, templates.npz, README).
"""
import sys

from huggingface_hub import HfApi

SPACE_NAME = sys.argv[1] if len(sys.argv) > 1 else "macedonian-alpr"
FILES = ["Dockerfile", "requirements.txt", "README.md",
         "alpr/*", "web/*", "data/templates.npz"]


def main():
    api = HfApi()
    user = api.whoami()["name"]                       # fails clearly if not authenticated
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=".",
                      allow_patterns=FILES,
                      ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
                      commit_message="Deploy classical MK ALPR backend")
    print(f"\nSpace: https://huggingface.co/spaces/{repo_id}")
    print(f"API:   https://{user}-{SPACE_NAME}.hf.space/api/health  (after the build finishes)")


if __name__ == "__main__":
    main()
