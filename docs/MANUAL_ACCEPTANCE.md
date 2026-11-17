# Manual acceptance checks

These checks need a T4 GPU, a Windows desktop, provider keys, or voice hardware. CI and the Python/body suites cannot
establish them, so they stay manual. Run them on the target machine and record the outcome (the project logs results in
Notion). This checklist is not a test suite: nothing here runs automatically.

## 1. Gate run — model quality (Kaggle/Colab T4)

Runbook: `colab/riya_v1/README.md`, sections 7–9.

- [ ] Confirm `evaluation.gate.identity()` matches the committed 0.11.1 calibration record (gate version, probes,
      prompt, scorers, embedder, repeat). Verified on 2026-11-17 in this checkout.
- [ ] Check the base GGUF's sha256 and the pinned `bge-small-en-v1.5` embedder with `FastEmbedder(verify=True)` on the
      runner. The run refuses a different embedder before starting.
- [ ] Ship the exact committed code (`git archive HEAD`), not a stale `jarvis/code` copy; compare identity on both sides.
- [ ] Run five fresh base-vs-base runs at `--repeat 4` with the pinned embedder. Time the first one: over 1200 s means
      the gate is too slow — stop before spending the other four.
- [ ] Calibrate from the five bank files on the laptop. `CALIBRATION: PASS` rewrites `evaluation/gate_calibration.json`;
      commit it and re-ship the code. A FAIL is recorded, never tuned against the held-out bank.
- [ ] Run the round-2 candidate verdict, then `evaluation.champion --check-verdict` locally. Adoption is the user's
      decision; `--adopt-verdict` is not run by anyone else.

Local status on 2026-11-17: identity and record match; no bank or verdict artifacts exist in this checkout, so the gate
has **not** been run here. Nothing has been promoted.

## 2. Windows desktop integration

Use `evaluation.health_check`, `evaluation.pet_check` and `evaluation.pet_cpu` where applicable.

- [ ] Welcome: all eight steps, each skippable; name, key (masked), voice, look, eyes, agents; reopening from Settings.
- [ ] Health: one row per component, at most one fix button; outward-facing actions show exactly what they will do and
      require a Yes. Change settings from two windows at once; both stay consistent.
- [ ] Startup shortcut: simulate failure and recovery; the successful patch still applies and the error is private to
      the originating window.
- [ ] Claude Code hooks: only the user runs `python -m riya.agents.install_hooks --apply`; verify hook events reach the
      agents chips and terminal focus returns the right window. The installer is never applied by an agent.
- [ ] Voice: barge-in while a reply is streaming; mic and TTS device selection; no crash when devices disappear.
- [ ] Reduced motion: "Keep her still" and the OS switch suppress walking/falling/confetti but keep poses, labels and
      the needs-you badge.
- [ ] Buddy/pet CPU: `evaluation.pet_cpu --mode idle` (~1%) and `--mode walk` (~3%) with the model server running.

## 3. Provider loose ends

- [ ] For every provider with a key in `providers.env`, check each configured model id against that provider's
      `/models` list. The Gemini, Mistral, SambaNova and NVIDIA ids were guesses when written.
- [ ] `cline auth` before Cline can join the model pool.
- [ ] `agy` stays out until its tools can be locked down; it currently runs with every tool on.
- [ ] OpenCode's free tier shares one daily allowance that resets at 00:00 UTC; watch for pool handovers at that time.

## 4. Live model and voice benchmarks

- [ ] Long-chat stability on the 6 GB GPU with voice running; no refusals; response latency at least as good as the
      current champion.
- [ ] Pocket TTS first-audio latency and Moonshine WER on the target CPU; re-run `colab/launch_moonshine_bench.py` if
      the streaming numbers are missing.
- [ ] Record each candidate's result in Notion; promote only on the user's "keep".

## What the automated suite deliberately does not cover

Opt-in real-embedder tests (`live_bge`), live brains, voice hardware and Windows desktop behavior are outside CI. A
green suite is necessary but not sufficient for any item above.
