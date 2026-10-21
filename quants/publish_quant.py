#!/usr/bin/env python3
"""Publish a quantized artifact dir to the HF Hub with its card.

Usage: python quants/publish_quant.py <artifact_dir> <repo_id> <card_md>

- creates repo <repo_id> if needed (public, model)
- uploads everything in <artifact_dir> (including SHA256SUMS)
- uploads <card_md> as README.md
Idempotent: re-running re-uploads changed files only.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    artifact, repo_id, card = sys.argv[1:4]
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=artifact,
        commit_message=f"upload {Path(artifact).name}",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=card,
        path_in_repo="README.md",
        commit_message="card",
    )
    print(f"PUBLISHED https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
