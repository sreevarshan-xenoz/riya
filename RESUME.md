# Resume point (paused 2026-10-11 ~12:20 at the user's request — wait until they say resume)

All stopped: laptop jobs, Colab sessions (none active), background watchers. Everything below is on this PC and in
`G:\My Drive\jarvis\runs`.

## Page 27 Stage 0 (laptop CPU)
- Brains `scratch_runs/bench/train-{v0,v0.1,oracle}` at 563 / 503 / 554 of 1200 episodes (state.pt saved ~12:18).
- Resume: `$env:RIYA_THREADS="5"` (full CPU) or unset (gentle, 2 threads) then `evaluation/stage0_local.ps1`.
  It resumes training, then runs B0–B4 + v0.1 evals on 20 test seeds, then Stage 1 gate fit + gated/MPC evals + scorecard.

## Riya v1 (Colab T4) — pass 1b (compile-free: prebuilt llama.cpp CUDA 12.8 build b11552)
Candidate results so far (`scratch_runs/colab/bench_models/`):
- Huihui LFM2.5-1.2B abliterated: GPU 1004 MB VRAM, TTFT 22 ms, 202 tok/s, 0 refusals, stable long chat → gate PASS;
  CPU-only (game mode, 2 Colab cores): TTFT 446 ms, 9.2 tok/s, 0 refusals → PASS.
- Huihui Qwen3.5-2B abliterated: 1584 MB VRAM (over 1.5 GB budget) → gate FAIL; 108 tok/s, 0 refusals.
- Not yet benched: stock LFM2.5-1.2B, LFM2.5 Heretic.
- Resume: new T4 → upload jarvis.zip + data/voice/synth_requests.jsonl → `colab/bootstrap_t4.py`
  (downloads models + runs pass1b: remaining benches → teacher dialogues → LoRA → GGUF → bench Riya v1).

## Voice
- TTS: Pocket TTS (318 ms first audio on 2-core CPU). STT: Moonshine v2 tiny whole-clip 554 ms, WER 0 (Whisper base 2007 ms);
  streaming-mode numbers were lost when the CPU session was reclaimed — rerun `colab/launch_moonshine_bench.py`.
- `riya/voice.py` now streams STT (Moonshine) + streams TTS (Pocket), with barge-in.

## Lessons (don't repeat)
- Never `pkill -f` with a pattern that may appear inside a pipeline shell's script text (killed pipelines twice).
- Free Colab reclaims sessions often: keep everything resumable and pull artifacts as soon as they exist.
