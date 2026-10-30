# nemotron_transcribe

Owns `nemo_speechlm`: NVIDIA Nemotron-3.5-Transcribe-30B-A4B, a speech-augmented Nemotron-H trunk (transcription, translation,
speaker-attributed and rich transcription, audio questions). Pack layout = mlx-audio's `nemotron_transcribe` (Blaizzy/mlx-audio
#1087); converter `scripts/convert_nemotron_transcribe.py` (`--bits 8 --group-size 128`, or `--bf16`). Served on
`/v1/audio/transcriptions` and as chat `input_audio`, both through the checkpoint's own template.

## Files
- `config.zig`: MLX-free claim; the trunk parses through `nemotron_h` (the NeMo `llm_config`); `parseAudio` reads the
  perception module (`perception`, `pe_encoder_config`).
- `plugin.zig`: `nemotron_h`'s table as it is, with `load` and `forward` wrapped (the trunk loads under `llm.backbone`,
  the tower beside it) and `MediaHooks` (`prepare`: the clip's samples, `tower`: the rows).
- `perception.zig`: NeMo log-mels, the ASR expert (32-layer RoPE Transformer, q/k LayerNorm), the speaker expert (31
  layers) streaming through a FIFO and an arrival-order speaker cache (`nemotron_diarization`'s `aoscKeep`), their fusion.
- `proof_test.zig`: the tower and a spliced prefill over the tiny pack (no reference oracle yet).

## Facts
- The perception module stays bf16 in a quantized pack; only `llm.*` quantizes. Mamba `A_log`/`D`/`dt_bias`, the router and
  its correction bias keep f32 in the pack.
- The prompt's rows are `<|audio|>` (the engine requires its `placeholder_ids.audio` in the run), id `vocab_size`: one
  past the embedding table. `forward` swaps those ids for `<SPECIAL_999>` (`config.spliceId`; a pack too small for it
  takes its last id, `parse` refuses a clash) before the trunk gathers them, and the splice (`base.placeholders.audio`)
  overwrites the rows. The engine hands its ids as a device array (`Forward.ids`, `tokens` empty), so the swap covers
  both forms; a test that passes host `tokens` only never runs the engine's path (`docs/gotchas/engine-mlx.md`).
- The speaker expert reads the RAW mels and the ASR expert the normalized ones (`perception.py`).
- A clip is split into 40 s windows with 50 rows of context either side; features are normalized over the WHOLE clip first.

## Rules
- A routed expert's `fc2` reads 1856 columns, which no 128-wide group divides: it quantizes at group size 64 as a per-path
  entry of the config's `quantization` (23 of them); the reader resolves each weight from its own geometry.
- The pack's `tokenizer_config.json` drops its inline `chat_template` (the converter): it is the older template, without
  audio parts, and the host prefers an inline template to `chat_template.jinja`.
- Transcription is the card's recipe (`transcription.nemoPrompt`) as the user turn, thinking off; `language` names the
  language in it and `prompt` takes the contextual recipe, never both.

## Not served
- `language` is the Qwen3-ASR list (the host's `/v1/audio/transcriptions` validation), not the card's 40 languages: a
  language outside it is a named 400; leave `language` out and the model detects it.
- The app's dictation and transcript surfaces pick their pack by `qwen3_asr`, so they ignore a loaded Nemotron Transcribe.
- Word timestamps (the CTC head), `diarized_json` from the embedded speaker model, MTP and the DFlash2 drafter: the pack
  keeps no CTC tensors and `spec` is empty.
