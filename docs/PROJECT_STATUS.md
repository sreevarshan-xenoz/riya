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
| `tests/core`, `tests/agents`, `tests/hands`, `tests/learn` | 3,140 passed, 10 skipped, 2 failed because `fastembed==0.9.0` was not installed in the lightweight environment |
| `npm test` in `body/` | 325 passed |

The broader Python run used `NO_PROXY=127.0.0.1,localhost` and `no_proxy=127.0.0.1,localhost` so local hook tests did not go through the environment's outbound proxy. It completed in about 144 seconds. There was also an upstream Starlette warning about its HTTPX test-client integration.

The two dependency failures were `test_installed_fastembed_reads_the_package_metadata` and `test_constructing_text_embedding_directly_is_guarded_too` in `tests/core/test_embed_pin.py`. The complete root Python suite, opt-in real-embedder tests, live brains, voice hardware, and Windows desktop behavior were not validated. Do not treat the partial suite as a complete release gate.

## Recommended next work, in order

1. **Finish the v0.11 quality gate before promoting a brain.** Use `colab/riya_v1/README.md`, sections 7–9: gather five fresh base-vs-base runs with the pinned embedder, calibrate using the recorded bank, then evaluate the candidate and check its verdict locally. Record failures rather than tuning against the held-out bank. Adoption remains the user's decision.
2. **Run the complete Python suite in the supported development environment.** Install the appropriate CPU PyTorch build and the pinned Riya requirements, including `fastembed==0.9.0`. Keep the existing real-model and real-user-file guards intact; live-embedder tests remain opt-in.
3. **Do the Windows integration pass.** Verify welcome/Health, two simultaneous windows changing settings, Startup-shortcut failure recovery, Claude Code hooks and terminal focus, voice barge-in, and reduced-motion behavior. Use the documented `evaluation.health_check`, `evaluation.pet_check`, and `evaluation.pet_cpu` tools where applicable. Pure tests here cannot establish hardware latency or desktop CPU budgets.
4. **Then automate the reproducible checks.** A CI job should run the desktop suite and isolated Python tests without downloading model weights or touching user files. Separate model/hardware acceptance from ordinary unit tests, and document the supported Python version and dependency installation path.

No training was resumed, model promoted, calibration record created, or held-out retrieval behavior changed in this pass.
