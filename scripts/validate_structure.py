#!/usr/bin/env python3
"""Strictly validate a converted mlx-whisper model and emit a receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(model_dir: Path) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.whisper import ModelDimensions, Whisper

    config_path = model_dir / "config.json"
    weights_path = model_dir / "weights.safetensors"
    generation_config_path = model_dir / "source_generation_config.json"
    if (
        not config_path.is_file()
        or not weights_path.is_file()
        or not generation_config_path.is_file()
    ):
        raise FileNotFoundError(
            "Expected config.json, weights.safetensors, and source_generation_config.json"
        )

    config = json.loads(config_path.read_text())
    if config.pop("model_type", None) != "whisper":
        raise ValueError("config.json must identify model_type=whisper")
    if "quantization" in config:
        raise ValueError("This release must be the unquantized conversion")

    weights = mx.load(str(weights_path))
    model = Whisper(ModelDimensions(**config), mx.float16)
    generation_config = json.loads(generation_config_path.read_text())
    expected_alignment_heads = np.asarray(generation_config["alignment_heads"], dtype=np.int64)
    model.set_alignment_heads(expected_alignment_heads)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())

    actual_alignment_heads = np.asarray(weights["alignment_heads"])
    if not np.array_equal(actual_alignment_heads, expected_alignment_heads):
        raise ValueError("Converted alignment_heads differ from KInIT generation_config.json")
    non_alignment = {key: value for key, value in weights.items() if key != "alignment_heads"}
    if any(value.dtype != mx.float16 for value in non_alignment.values()):
        raise ValueError("Every model tensor must be float16")
    if weights["alignment_heads"].dtype != mx.int64:
        raise ValueError("alignment_heads must be int64")

    tokenizer = get_tokenizer(
        multilingual=True,
        num_languages=model.num_languages,
        language="sk",
        task="transcribe",
    )
    if tokenizer.sot_sequence != (50258, 50298, 50360):
        raise ValueError(f"Unexpected Slovak SOT sequence: {tokenizer.sot_sequence}")

    receipt = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "model_directory": model_dir.name,
        "strict_load": True,
        "tensor_count": len(weights),
        "parameter_count": sum(math.prod(value.shape) for value in non_alignment.values()),
        "tensor_element_count": sum(math.prod(value.shape) for value in weights.values()),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
        "keyset_sha256": hashlib.sha256(("\n".join(sorted(weights)) + "\n").encode()).hexdigest(),
        "dimensions": config,
        "alignment_heads": {
            "shape": list(actual_alignment_heads.shape),
            "dtype": str(weights["alignment_heads"].dtype),
            "matches_source_generation_config": True,
        },
        "tokenizer_languages": model.num_languages,
        "slovak_sot_sequence": list(tokenizer.sot_sequence),
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_whisper": importlib.metadata.version("mlx-whisper"),
        },
    }
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate(args.model_dir)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
