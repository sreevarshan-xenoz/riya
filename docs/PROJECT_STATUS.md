# Project review and next steps

Reviewed 2026-11-12. This is a source review and local test report, not a live-model benchmark or a Windows acceptance report.

## Where the project stands

- `jarvis/` contains the research language/world models; `riya/` contains the companion runtime; `body/` is the Electron desktop app.
- The latest checked-in feature plan is v0.11, **She uses what she knows**: local-only memory retrieval for personal requests, a pinned embedder, and retrieval-aware model gating.
- `RESUME.md` records an older, paused training session. Its episode counts and model results are historical, not evidence of currently running jobs.
- The latest ruling in `docs/superpowers/riya-rulings.md` identifies a fresh five-run gate bank, calibration, and a new candidate verdict as the next model-quality work. This review did not inspect external artifacts or establish whether those runs have happened elsewhere.

## Gap fixed in this pass: settings-error privacy

A dictionary-shaped `set_settings` message passed protocol parsing but could fail deeper validation with an error containing the submitted regex or unknown setting name. That error was broadcast to every connection, potentially repeating an accidentally pasted key.

- `riya/core/engine.py` now returns fixed, input-free settings errors only to the originating connection. Autostart exception details are no longer sent over the socket either.
- Successful settings changes still broadcast, including changes applied before a Startup-shortcut failure. In-process callers without a connection callback retain the existing fallback.
- The desktop recognizes the generic validation error as a settings error; its pending-control router still associates it with the submitted control.
- Multi-socket tests cover invalid denylist patterns, unknown keys, invalid values, unchanged rejected settings, continued socket use, successful broadcasts, and private autostart failures.

## Validation from this review

A lightweight `.venv` was created for tests; no model, adapter, user settings, or external training job was changed.

| Check | Result |
| --- | --- |
| New Python regression cases before the fix | All 5 failed, reproducing the unsafe behavior |
| `tests/core/test_app.py`, `test_engine.py`, `test_setup_engine.py` after the fix | 133 passed |
| Full root suite on Python 3.14 (`fastembed==0.9.0` + CPU PyTorch installed) | 3,502 passed, 10 skipped |
| Full root suite on Python 3.12 with only the CI dependency set | 3,502 passed, 10 skipped |
| `npm test` in `body/` (after `npm ci --ignore-scripts`) | 325 passed |
| `evaluation.gate.identity()` against the committed 0.11.1 record | `gate_version`, `probe_sha256`, `prompt_fp`, `scorer_sha256`, `embedder`, `repeat` all match |

The Python runs used `NO_PROXY=127.0.0.1,localhost` and `no_proxy=127.0.0.1,localhost` so local hook tests did not go through the environment's outbound proxy, and completed in about 4.5 minutes. There was also an upstream Starlette warning about its HTTPX test-client integration.

The earlier two dependency failures (`test_installed_fastembed_reads_the_package_metadata` and `test_constructing_text_embedding_directly_is_guarded_too` in `tests/core/test_embed_pin.py`) were environmental: `fastembed==0.9.0` was missing. With it installed they pass. Opt-in real-embedder tests, live brains, voice hardware, and Windows desktop behavior were not validated. Do not treat the suite alone as a complete release gate.

## CI

`.github/workflows/ci.yml` runs two jobs on every push to `main` and every PR:

- **Python (3.12):** CPU PyTorch plus only the modules the tests import, then `python -m pytest -q`. No model weights are downloaded; the conftest guards stay in force.
- **Body (Node 22):** `npm ci --ignore-scripts` (skips Electron's binary) and `npm test`.

The exact dependency set was rehearsed in a clean Python 3.12 venv before committing.

## Recommended next work, in order

1. **Finish the v0.11 quality gate before promoting a brain.** The committed calibration is 0.11.1 and `evaluation.gate.identity()` matches it, so the tree is verdict-ready; what remains is the Kaggle T4 run. Use `colab/riya_v1/README.md`, sections 7–9: gather five fresh base-vs-base runs with the pinned embedder (K3), calibrate using the recorded bank (K4, already passing), then run the round-2 candidate and `evaluation.champion --check-verdict` (K5). Record failures rather than tuning against the held-out bank. Adoption (K6) is the user's decision.
2. **Run the complete Python suite in the supported development environment.** Done in this pass on 3.12 and 3.14: 3,502 passed, 10 skipped. Keep the real-model and real-user-file guards intact; live-embedder tests remain opt-in.
3. **Do the Windows integration pass.** Verify welcome/Health, two simultaneous windows changing settings, Startup-shortcut failure recovery, Claude Code hooks and terminal focus, voice barge-in, and reduced-motion behavior. Use the documented `evaluation.health_check`, `evaluation.pet_check`, and `evaluation.pet_cpu` tools where applicable. Pure tests here cannot establish hardware latency or desktop CPU budgets.
4. **Keep CI green and add the hardware gates separately.** CI now covers the desktop suite and the isolated Python suite. Model/hardware acceptance stays a manual, documented step.

No training was resumed, model promoted, calibration record created, or held-out retrieval behavior changed in this pass.
