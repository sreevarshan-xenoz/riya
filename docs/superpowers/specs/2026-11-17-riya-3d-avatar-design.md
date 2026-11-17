# Riya's 3D body: a VRM avatar (design)

Date: 2026-11-17. Status: proposed. Follows the route in `docs/HANDOFF.md`; build only after this spec is agreed.

## Why

Her on-screen body is a 2D sprite pack rendered by Electron. The user wants her to look like a real person. VRM is the
standard humanoid avatar format: expressions, blink, visemes and look-at travel with the model, and a VRoid-made or
commissioned avatar can be swapped without touching the runtime. The body must keep working on the Linux laptop's
MX250, which also runs her local model server.

## Goals

- The avatar replaces the sprite *renderer*, not the companion: every core message keeps its current meaning.
- Map all 14 expressions, speech lip-sync, blinking, looking, breathing and the Little Riya poses onto the model.
- Keep sprites as a one-setting fallback, including when WebGL or the model file fails.
- No bundler and no new runtime server: vendored libraries, loaded as plain scripts like the rest of `body/`.
- Stay inside the body's CPU budget (~1% idle, ~3% walking, per `evaluation.pet_cpu`).

## Non-goals

- No new core-to-body messages, no changes to the model server or core.
- No physics, cloth simulation, IK beyond VRM's look-at, or full-body lip-sync beyond the `aa` viseme.
- No shipped character model: VRM/GLB/GLTF files stay git-ignored, like the sprite art.
- No rewriting of `pet-logic.js`'s walking, sitting, sleeping, privacy or reduced-motion rules.

## Anti-goals

- Hiding her when WebGL is missing or the avatar file is bad — that is when sprites must take over.
- Downloading a model at runtime or reading one from a URL. The path is local, like `sprite_pack`.
- Spinning the render loop while nothing changes, or adding load to the laptop during a game.

## Approaches

1. **three.js + `@pixiv/three-vrm` inside the existing renderer (chosen).** Both are copied into `body/vendor/` at exact
   pinned versions and loaded as scripts. Keeps one window, one WebSocket bridge, and the same message contract.
2. A separate 3D viewer process or iframe. More moving parts, a second render surface, and harder to keep the speech
   bubble, badge and pet logic in sync.
3. Replace sprites outright. Leaves no fallback on machines without WebGL and discards working, tested code.

## Design

- **Selection.** A new pair of settings, `avatar: "sprites" | "vrm"` (default `"sprites"`) and `avatar_path` (a local
  file path, empty by default). Add both to the core's validated settings so the existing `set_settings` patch path
  carries them; the body never reads arbitrary files itself. The model file lives outside the repo.
- **Fallback.** On start, and whenever the setting or path changes, try to create a WebGL context and load the VRM.
  Failure of any step (no WebGL, bad file, load error) falls back to the sprite renderer and reports why in Health.
  Only one renderer is active at a time; the bubble, badge and pet state are shared.
- **Pure mapping in `body/src/avatar-logic.js`.** Tested with `node --test`, the same way `sprite-logic.js` is:
  - the 14 core expressions to VRM expression presets, with `neutral` as the reset;
  - `mouth.level` (0..1) to the `aa` viseme with smoothing matching the current `MouthGate`;
  - the blink timer to the `blink` expression (and never a permanently shut eye after a failed frame);
  - `look.side` / `look.watching` to the VRM look-at target;
  - breathing and lean to small spine/chest bone offsets, suppressed while reduced motion is on;
  - the reduced-motion gate: expressions, labels and the needs-you wave marker survive; bone motion does not.
- **Message contract (unchanged).** `expression` selects the preset; `mouth` drives `aa`; `state` sets the base
  expression and presence; `look` moves the eyes; `say`/`say_end` drive the speech bubble as today; `agent_activity`
  still produces the badge and, in buddy mode, the wave. No core change is needed for any of these.
- **Little Riya.** `pet-logic.js` remains the owner of where she is and what she is doing. Walk/sit/sleep become VRM
  animation clips (VRMA or retargeted Mixamo) when the pack supplies them; until then an idle pose is used and the
  existing sprite-style motion still drives the timers. Reduced motion and "Keep her still" keep their current meaning.
- **Budget.** Cap rendering at about 30 fps; render only when a mapped value changed or while an animation is active;
  stop while `document.hidden`. Use a small canvas and a light scene. Check idle and walking CPU with
  `evaluation.pet_cpu` on the Linux laptop before calling the feature done; if the budget is missed, drop bloom,
  shadow and the animation loop before dropping expressions.
- **Files.** `body/src/avatar-logic.js` and tests; a thin renderer adapter next to `renderer.js`; two pinned vendor
  files in `body/vendor/`; two new validated settings; one Health line. Existing `.gitignore` already excludes
  `*.vrm`, `*.glb`, `*.gltf`, `*.fbx`, `*.blend`.

## Validation

1. `node --test` for every mapping in `avatar-logic.js`, including unknown expression names, mouth extremes, blink
   failure and reduced motion.
2. Manual visual pass with a local VRM: all 14 expressions, lip-sync while speaking, blink, look, breathing, and the
   sprite fallback with WebGL disabled.
3. `evaluation.pet_cpu --mode idle` and `--mode walk` against the documented budgets, on the Linux laptop with her
   model server running.
4. Full Python and body suites stay green; the core's protocol and settings tests are unchanged except for the two new
   settings.

## Left to the builder

- The exact per-model expression preset names (they vary by avatar) and whether to ship a mapping editor in Settings.
- Viseme smoothing constants and the bone offsets that read as "breathing" on the chosen model.
- Which walk/sit/sleep clips to use, and how to retarget them for a VRoid model.
- Who supplies the avatar and its licence; nothing is committed or downloaded by the app.
