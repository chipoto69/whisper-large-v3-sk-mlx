# Release receipts

The published release includes machine-readable receipts for:

- `conversion.json` — pinned source, conversion format, dtype, tensor count,
  key-set digest, output size, and output SHA-256;
- `structural.json` — strict MLX architecture load and shape validation;
- `parity-public.json` — public synthetic Slovak fixture, backend transcript,
  edit distance, WER between backends, segment timing, and production speed;
- `parity-private-redacted.json` — aggregate and per-fixture metrics with all
  private names, filenames, transcript text, and audio excluded.

Parity transcript text and the audio fixture are excluded to protect privacy.
The receipt identifies the fixture by SHA-256.
