# Project review and next steps

Reviewed 2026-11-17. This is a source review and local test report, not a live-model benchmark or a Windows acceptance report.

## Where the project stands

- `jarvis/` contains the research language/world models; `riya/` contains the companion runtime; `body/` is the Electron desktop app.
- `main` is at the docs refresh merge (PR #28), after the nine audit fixes from this review (PRs #19–#27) were merged on 2026-11-17. They were verified together on a local branch first, then each PR's CI passed and the post-merge run on `main` was green. PR #17 added the online model pool.
- The latest feature is v0.12, **Morpheus in Riya** (`riya/core/routine/`). Engine-made routine claims ship disabled (`CLAIMS_ENABLED = False`) because the held-out record failed A3, A5 and A7; hunches still run. Specs, plans and rulings through v0.12 and the 2026-11-13 model pool are checked in under `docs/superpowers/`.
- The Linux laptop is the main dev machine: `bash scripts/start_riya.sh` starts the model server, core and body, also used by the `riya.service` systemd user unit. Windows is the older setup; its path is in `docs/HANDOFF.md`.
- `RESUME.md` and `LOOPS.md` are dated, paused historical records. The versioned specs/plans/rulings are also historical; they record decisions as made and are deliberately not rewritten.

## Audit and fixes from this review

Static audit: syntax compilation clean; `ruff check .` reports 1,080 findings, almost entirely style rules in unconfigured default mode (the repo has no ruff config). The high-signal subset was reviewed by hand: the `F821` in `jarvis/lm/data.py` is a false positive caused by `del arr` after the closure is used, and the `B023`/`PLR0124` hits are loop-local lambdas and a deliberate NaN check.

Four parallel source audits (core runtime, subsystems/agents/hands/eyes/routine, Electron body, research/evaluation) plus a documentation audit produced nine confirmed, reproduced bugs. Each has its own branch, single commit, regression test and PR:

| PR | Severity | Bug | Fix |
| --- | --- | --- | --- |
| #19 | medium | Reduced motion erased the "needs you" wave, hiding the badge/label | Keep the wave marker; animation is already gated on reduced views |
| #20 | medium | `free_vram_mb()` crashed the local server on hosts without `nvidia-smi` | Total probe: missing/failed/malformed telemetry means 0 MB free (CPU) |
| #21 | low/medium | Sprite frame symlinks could point outside the pack | Resolve real paths and reject frames that leave the pack root |
| #22 | medium (privacy) | Bare `hf_`/`glpat-`/`ya29.`/`AIza` tokens and glued markers survived redaction | Cover those prefixes and remove word-boundary anchors for distinctive tokens |
| #23 | medium | `<PROVIDER>_MODELS` overrides marked text-only models as vision-capable | Preserve known image flags; unknown ids default to text-only |
| #24 | low | Short probe check flagged `speak_first` stage notes | Exclude them, matching `holdout.probe_texts()` |
| #25 | low | Text CLI raised `UnboundLocalError` on an empty reply | Report "(no reply)" when no sentence arrived |
| #26 | medium | Voice SFT crashed when the validation split was smaller than 19 batches | Iterate real slices; `nan` for an empty val set |
| #27 | medium | Voice SFT crashed sampling a batch from a train set smaller than `batch_size` | Sample with replacement for small pools; clear error for empty pools |

No private identifiers, keys or user files were touched. The local pre-push hook, the no-hooks-installer rule and the test guards stay in force.

## Validation from this review

All runs used `NO_PROXY=127.0.0.1,localhost` and `no_proxy=127.0.0.1,localhost` so local hook/websocket tests did not go through the outbound proxy.

| Check | Result |
| --- | --- |
| Python suite on `main` before the audit fixes (`.venv`, Python 3.14) | 3,745 passed, 10 skipped |
| Python suite on all nine fixes merged locally before merge, and on `main` after (3.14) | **3,756 passed, 10 skipped** |
| Body suite on all nine fixes merged locally before merge, and on `main` after (`npm test`) | **331 passed** |
| Post-merge CI on `main` (Python 3.12 + Node 22, run 44034402336) | both jobs green |
| Syntax compilation (`compileall`, Python + JS audit) | clean |
| `evaluation.gate.identity()` against the committed 0.11.1 record | `gate_version`, `probe_sha256`, `prompt_fp`, `scorer_sha256`, `embedder`, `repeat` all match |

The only warning is an upstream `StarletteDeprecationWarning` about `httpx` with `starlette.testclient` — not a project issue. Opt-in real-embedder tests, live brains, voice hardware and Windows desktop behavior were not validated. The suite alone is not a release gate.

## CI

`.github/workflows/ci.yml` runs two jobs on every push to `main` and every PR:

- **Python (3.12):** CPU PyTorch plus only the modules the tests import, then `python -m pytest -q`. No model weights are downloaded; the `tests/conftest.py` guards stay in force.
- **Body (Node 22):** `npm ci --ignore-scripts` (skips Electron's binary) and `npm test`.

## Recommended next work, in order

1. **Write the 3D avatar spec**, then build it — the next feature in `docs/HANDOFF.md`. Keep the existing message contract, add sprites as fallback, and stay inside the body's CPU budget.
2. **Finish the v0.11 quality gate before promoting a brain.** The committed calibration is 0.11.1 and `evaluation.gate.identity()` matches it. See `colab/riya_v1/README.md`, sections 7–9: five fresh base-vs-base runs with the pinned embedder, calibrate, then the round-2 candidate and `evaluation.champion --check-verdict`. Record failures; never tune against the held-out bank.
3. **Do the Windows integration pass.** Welcome/Health, two windows changing settings, Startup-shortcut failure recovery, Claude Code hooks and terminal focus, voice barge-in and reduced motion. Use `evaluation.health_check`, `pet_check` and `pet_cpu` where applicable; pure tests cannot establish hardware latency or desktop CPU budgets.
4. **Keep CI green and add hardware gates separately.** Model/hardware acceptance remains a manual, documented step.
