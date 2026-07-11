# Hugging Face upload

The upload helper refuses any weight file whose SHA-256 is not the accepted
release hash.

Authenticate once:

```bash
hf auth login
```

Verify the complete upload plan without network writes:

```bash
python scripts/upload_to_huggingface.py /path/to/mlx-fp16 --dry-run
```

Publish the public model repository:

```bash
python scripts/upload_to_huggingface.py /path/to/mlx-fp16 \
  --repo-id chipoto69/whisper-large-v3-sk-mlx
```

The helper uploads only the accepted weights and config, the source generation
config, the Hugging Face model card, the licenses/notices, and public-safe or
redacted receipts. It never uploads fixture audio or personal transcript text.
