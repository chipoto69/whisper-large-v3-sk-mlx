#!/usr/bin/env python3
"""Compare pinned Transformers and MLX outputs on one <=30s Slovak fixture.

The default receipt contains transcript hashes and distance metrics, not the
transcript text. This makes it safe to publish when the local fixture is not.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE_MODEL = "kinit/whisper-large-v3-sk"
SOURCE_REVISION = "8654a11704f3564ac53a4b266a567bd4f20c8c51"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def duration_seconds(path: Path) -> float | None:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        )
        return float(output.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def transformers_transcribe(
    audio: Path, source: str, revision: str
) -> tuple[str, float, float, float]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    load_started = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source,
        revision=revision,
        dtype=dtype,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started

    kwargs = {
        "chunk_length_s": 30,
        "generate_kwargs": {"language": "slovak", "task": "transcribe"},
    }
    warmup_started = time.perf_counter()
    asr(str(audio), **kwargs)
    warmup_seconds = time.perf_counter() - warmup_started
    started = time.perf_counter()
    result = asr(str(audio), **kwargs)
    elapsed = time.perf_counter() - started
    return result["text"].strip(), load_seconds, warmup_seconds, elapsed


def mlx_transcribe(audio: Path, model_dir: Path) -> tuple[str, float, float]:
    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": str(model_dir),
        "language": "sk",
        "task": "transcribe",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "fp16": True,
        "verbose": None,
    }
    warmup_started = time.perf_counter()
    mlx_whisper.transcribe(str(audio), **kwargs)
    warmup_seconds = time.perf_counter() - warmup_started
    started = time.perf_counter()
    result = mlx_whisper.transcribe(str(audio), **kwargs)
    elapsed = time.perf_counter() - started
    return result["text"].strip(), warmup_seconds, elapsed


def compare(audio: Path, model_dir: Path, source: str, revision: str, include_text: bool) -> dict:
    duration = duration_seconds(audio)
    if duration is not None and duration > 30.1:
        raise ValueError(f"Parity fixture must be <=30 seconds; received {duration:.3f}s")

    hf_text, hf_load_seconds, hf_warmup_seconds, hf_seconds = transformers_transcribe(
        audio, source, revision
    )
    mlx_text, mlx_warmup_seconds, mlx_seconds = mlx_transcribe(audio, model_dir)
    hf_normalized = normalize(hf_text)
    mlx_normalized = normalize(mlx_text)
    reference_words = hf_normalized.split()
    mlx_words = mlx_normalized.split()
    distance = edit_distance(reference_words, mlx_words)

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": source,
        "source_revision": revision,
        "mlx_model_dir": str(model_dir.resolve()),
        "audio_sha256": sha256_file(audio),
        "audio_duration_seconds": duration,
        "decoding": {
            "language": "slovak/sk",
            "task": "transcribe",
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "chunk_seconds": 30,
        },
        "transformers_load_seconds": hf_load_seconds,
        "transformers_warmup_seconds": hf_warmup_seconds,
        "transformers_measured_seconds": hf_seconds,
        "mlx_warmup_including_load_seconds": mlx_warmup_seconds,
        "mlx_measured_seconds": mlx_seconds,
        "speedup": hf_seconds / mlx_seconds if mlx_seconds else None,
        "normalized_exact_match": hf_normalized == mlx_normalized,
        "word_edit_distance": distance,
        "word_error_rate_between_backends": distance / max(1, len(reference_words)),
        "transformers_text_sha256": hashlib.sha256(hf_normalized.encode()).hexdigest(),
        "mlx_text_sha256": hashlib.sha256(mlx_normalized.encode()).hexdigest(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "environment": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_whisper": importlib.metadata.version("mlx-whisper"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    if include_text:
        result["transformers_text"] = hf_text
        result["mlx_text"] = mlx_text
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--source", default=SOURCE_MODEL)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include transcripts in the receipt; off by default for privacy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = compare(
        args.audio,
        args.model_dir,
        args.source,
        args.revision,
        args.include_text,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
