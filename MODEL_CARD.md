---
library_name: mlx
license: apache-2.0
language:
  - sk
base_model: kinit/whisper-large-v3-sk
pipeline_tag: automatic-speech-recognition
tags:
  - mlx
  - mlx-whisper
  - whisper
  - slovak
  - automatic-speech-recognition
---

# KInIT Whisper Large V3 Slovak — MLX FP16

This is an unquantized FP16 MLX conversion of
[`kinit/whisper-large-v3-sk`](https://huggingface.co/kinit/whisper-large-v3-sk)
at source revision
[`8654a11704f3564ac53a4b266a567bd4f20c8c51`](https://huggingface.co/kinit/whisper-large-v3-sk/commit/8654a11704f3564ac53a4b266a567bd4f20c8c51).

It preserves the KInIT Slovak fine-tune's Whisper Large V3 architecture and
converts the weights for inference with `mlx-whisper==0.4.3` on Apple silicon.
It is not a new fine-tune.

## Intended use

Slovak automatic speech recognition across varied recording conditions. Pass
`language="sk"` and `task="transcribe"` explicitly. Long recordings should be
processed in 30-second windows; `mlx_whisper.transcribe` performs that
windowing.

## Source evaluation

The following measurements are reported by KInIT for the source checkpoint,
not newly claimed for this conversion:

| Model | Common Voice 24 SK WER | Internal WER |
|---|---:|---:|
| OpenAI Whisper Large V3 | 25.09% | 18.72% |
| KInIT Whisper Large V3 SK | 9.45% | 8.06% |

Conversion integrity and backend-parity results are recorded separately under
[`receipts/`](receipts/).

## Decoding settings

```python
result = mlx_whisper.transcribe(
    audio,
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
    fp16=True,
)
```

## Limitations

The conversion inherits the source model's limitations:

- non-Slovak performance is substantially degraded by Slovak-only fine-tuning;
- dialectal, heavily accented, noisy, or domain-specific speech may be worse;
- a single unchunked segment should not exceed 30 seconds;
- automatic language detection is unreliable for this Slovak-specialized
  checkpoint, so callers must pass `language="sk"`;
- safety-critical transcription requires human review.

FP16 conversion can introduce small numerical differences from the original
Transformers checkpoint. See the release parity receipt for the measured
effect on a Slovak fixture.

## Training

No training was performed for this repository. KInIT performed the original
full-parameter Slovak fine-tuning. Refer to the
[source model card](https://huggingface.co/kinit/whisper-large-v3-sk) for its
data, training procedure, evaluation, and acknowledgements.

## License and attribution

Apache-2.0. Original model by KInIT. Conversion tooling follows the MLX Whisper
implementation maintained in Apple's `ml-explore/mlx-examples` repository.
