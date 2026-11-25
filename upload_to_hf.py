#!/usr/bin/env python3
"""Upload Philosopher Micro-RSI (1.52M seed checkpoint + starter architecture) to Hugging Face Hub.

Prerequisites:
  1. Run `hf auth login` once in your terminal with a Hugging Face Write token
     (create one at https://huggingface.co/settings/tokens).
  2. Run `python3 upload_to_hf.py` (optionally pass `--repo-name philosopher-micro-rsi`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent

FILES_TO_UPLOAD = [
    ("MODEL_CARD.md", "README.md"),
    ("LICENSE", "LICENSE"),
    ("checkpoints/brain.pt", "brain.pt"),
    ("checkpoints/tokenizer.json", "tokenizer.json"),
    ("philosopher/__init__.py", "philosopher/__init__.py"),
    ("philosopher/brain.py", "philosopher/brain.py"),
    ("philosopher/tokenizer.py", "philosopher/tokenizer.py"),
    ("philosopher/sandbox_verifier.py", "philosopher/sandbox_verifier.py"),
    ("philosopher/rsi_engine.py", "philosopher/rsi_engine.py"),
    ("playground.py", "playground.py"),
    ("run_rsi_session.py", "run_rsi_session.py"),
    ("vault/fixed_exam.json", "vault/fixed_exam.json"),
    ("vault/rsi_state.json", "vault/rsi_state.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default="philosopher-micro-rsi")
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()
    username = user["name"]
    repo_id = f"{username}/{args.repo_name}"

    print(f"Logged into Hugging Face as: {username}")
    print(f"Creating/updating model repo: https://huggingface.co/{repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    for local_rel, hf_path in FILES_TO_UPLOAD:
        local_fp = ROOT / local_rel
        assert local_fp.exists(), f"Missing file: {local_fp}"
        print(f"  Uploading {local_rel} -> {hf_path}")
        api.upload_file(
            path_or_fileobj=str(local_fp),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {hf_path}",
        )

    print(f"\nDONE! Live at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
