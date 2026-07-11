# whisper-large-v3-sk-mlx

Reproducible Apple MLX conversion of
[`kinit/whisper-large-v3-sk`](https://huggingface.co/kinit/whisper-large-v3-sk),
KInIT's full-parameter Slovak fine-tune of Whisper Large V3.

The source is pinned to commit
[`8654a11704f3564ac53a4b266a567bd4f20c8c51`](https://huggingface.co/kinit/whisper-large-v3-sk/commit/8654a11704f3564ac53a4b266a567bd4f20c8c51).
This release is an unquantized FP16 architecture conversion for Apple silicon;
it is not a new fine-tune and it does not substitute a generic English model.

- Model weights: [Hugging Face](https://huggingface.co/chipoto69/whisper-large-v3-sk-mlx)
- Conversion source: this repository
- Runtime: [`mlx-whisper==0.4.3`](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- License: Apache-2.0, inherited from the source checkpoint

## Why this checkpoint

KInIT reports substantially better Slovak recognition than the generic
Whisper Large V3 baseline: 9.45% versus 25.09% WER on Common Voice 24 SK, and
8.06% versus 18.72% WER on its internal evaluation. Those are source-model
results; the MLX conversion is separately checked for structural integrity and
backend parity in [`receipts/`](receipts/).

## Install

MLX requires an Apple-silicon Mac.

```bash
python -m pip install "mlx-whisper==0.4.3"
```

## Slovak transcription

Keep the language and task explicit. KInIT fine-tuned the model for Slovak,
and its model card recommends a maximum reliable 30-second segment without
chunking. `mlx_whisper.transcribe` windows longer audio internally.

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "memo.m4a",
    path_or_hf_repo="chipoto69/whisper-large-v3-sk-mlx",
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

print(result["text"])
```

The complete runnable example is
[`examples/transcribe_slovak.py`](examples/transcribe_slovak.py).

## Reproduce the conversion

The converter handles both single-file and sharded Hugging Face safetensors
checkpoints. The pinned source currently has two weight shards.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python scripts/convert_hf_to_mlx.py \
  --source kinit/whisper-large-v3-sk \
  --revision 8654a11704f3564ac53a4b266a567bd4f20c8c51 \
  --dtype float16 \
  --output mlx_model
```

The generated model directory contains:

```text
mlx_model/
|-- config.json
|-- weights.safetensors
|-- conversion_receipt.json
`-- source_generation_config.json
```

Weights are intentionally excluded from GitHub and published through the
Hugging Face model repository.

## Validate

Strict structural validation checks the complete key set and every tensor
shape against the MLX Whisper Large V3 architecture.

```bash
python scripts/validate_structure.py mlx_model \
  --output receipts/structural.json
```

Parity uses the same pinned KInIT model, a Slovak fixture no longer than 30
seconds, deterministic temperature-zero decoding, and explicit Slovak
language/task tokens. Receipts redact transcript text by default.

```bash
python scripts/parity_benchmark.py fixture.wav mlx_model \
  --output receipts/parity.json
```

Run the lightweight converter tests with:

```bash
pytest -q
```

## Release validation

The accepted artifact is 3,083,275,629 bytes with SHA-256
`a3e72c9c868ef796dc1a3df36639f07d5eb20308d7bfa9ca9f09cbc7e4f0c0ff`.
Strict validation loaded all 1,259 tensors and 1,541,570,560 FP16 model
elements. The remaining tensor is KInIT's 10x2 `alignment_heads` coordinate
array in Int64.

The public synthetic Slovak fixture produced an exact normalized transcript
match between the pinned Transformers checkpoint and MLX: word edit distance
0, backend WER 0, and character similarity 1.0. Production MLX decoding with
segment timestamps processed the 12.24-second fixture in 1.60 seconds,
including model load (7.67x real time). See
[`receipts/parity-public.json`](receipts/parity-public.json).

A separate private, redacted three-clip probe covered 86.54 seconds of audio.
Two normal-length clips matched exactly; a 1.04-second mixed-language fragment
differed by one word. MLX processed the set at 5.55x real time versus 2.56x for
the Transformers backend. No private name, filename, audio, or transcript is
included in the receipt.

## Scope and limitations

- This conversion inherits KInIT's catastrophic-forgetting warning: use a
  multilingual base model for non-Slovak audio.
- Strong accents, dialects, proper nouns, code-switching, and unfamiliar
  domains can reduce accuracy.
- Do not rely on automatic language detection for this Slovak-specialized
  checkpoint; pass `language="sk"` explicitly.
- Human review is still required for safety-critical or legally consequential
  transcripts.
- FP16 MLX and the original Transformers backend may not be numerically
  bit-identical. The published parity receipt measures the observed decoding
  difference on the release fixture.
- No personal voice memo, audio fixture, or transcript is published here.

## Alignment-head preservation

The source `generation_config.json` selects ten decoder attention heads for
time alignment. A stock local-Hugging-Face conversion silently retains the
generic 320-head fallback. This converter loads KInIT's exact ten coordinate
pairs before serialization; structural validation rejects any mismatch.

## Attribution

The model weights were created and released by the
[KInIT team](https://huggingface.co/kinit). This repository only provides the
MLX conversion and validation tooling. The key remapping follows Apple's
open-source [`mlx-examples/whisper/convert.py`](https://github.com/ml-explore/mlx-examples/blob/main/whisper/convert.py).

See [`MODEL_CARD.md`](MODEL_CARD.md) for model metadata and inherited
limitations.
