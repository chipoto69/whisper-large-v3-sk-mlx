#!/usr/bin/env python3
"""Transcribe Slovak speech with the published MLX conversion."""

import argparse
import json

import mlx_whisper

MODEL_ID = "chipoto69/whisper-large-v3-sk-mlx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    result = mlx_whisper.transcribe(
        args.audio,
        path_or_hf_repo=args.model,
        language="sk",
        task="transcribe",
        suppress_tokens=[],
        suppress_blank=True,
        max_initial_timestamp=1.0,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=None,
        word_timestamps=False,
        fp16=True,
        verbose=None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
