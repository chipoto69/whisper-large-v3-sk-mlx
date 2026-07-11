#!/usr/bin/env python3
"""Verify and publish the accepted model artifact to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "a3e72c9c868ef796dc1a3df36639f07d5eb20308d7bfa9ca9f09cbc7e4f0c0ff"
DEFAULT_REPO = "chipoto69/whisper-large-v3-sk-mlx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_plan(model_dir: Path, repository_root: Path) -> list[tuple[Path, str]]:
    plan = [
        (model_dir / "weights.safetensors", "weights.safetensors"),
        (model_dir / "config.json", "config.json"),
        (model_dir / "source_generation_config.json", "source_generation_config.json"),
        (repository_root / "MODEL_CARD.md", "README.md"),
        (repository_root / "LICENSE", "LICENSE"),
        (repository_root / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        (repository_root / "receipts" / "conversion.json", "receipts/conversion.json"),
        (repository_root / "receipts" / "structural.json", "receipts/structural.json"),
        (repository_root / "receipts" / "parity-public.json", "receipts/parity-public.json"),
        (
            repository_root / "receipts" / "parity-private-redacted.json",
            "receipts/parity-private-redacted.json",
        ),
    ]
    missing = [str(local) for local, _ in plan if not local.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upload inputs: {missing}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir.expanduser().resolve()
    plan = upload_plan(model_dir, root)
    weights_hash = sha256_file(model_dir / "weights.safetensors")
    if weights_hash != EXPECTED_SHA256:
        raise ValueError(
            f"Refusing upload: weights SHA-256 {weights_hash} != accepted {EXPECTED_SHA256}"
        )

    manifest = {
        "repo_id": args.repo_id,
        "private": args.private,
        "weights_sha256": weights_hash,
        "files": [{"local": str(local), "remote": remote} for local, remote in plan],
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.whoami()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    for local, remote in plan:
        print(f"Uploading {remote}")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"upload {remote}",
        )
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
