# Handoff: continuing on the Windows laptop (2026-11-13)

Next job: make Riya's on-screen body look like a real person (a 3D model) instead of the 2D sprite pack.

## Where things stand

- `main` is at the model-pool merge (PR #17). CI is green, and the repo is public.
- **Online brain:** a pool of free models that take over for each other: opencode's free models first, then any
  keyed providers in `providers.env`, then Kilo. Spec: `docs/superpowers/specs/2026-11-13-riya-model-pool-design.md`.
- **Small talk:** runs on the local LFM2.5 1.2B GGUF in `~/models/riya`.
- **v0.12 routine:** ships with `CLAIMS_ENABLED = False` (hunches only), because the held-out record failed A3, A5
  and A7.
- **The Linux laptop is the main dev machine.** Riya runs there as a systemd user service, `riya.service`. Windows is
  the older setup; `scripts/start_riya.ps1` still works.

## Setting up the Windows clone

1. `git pull` on `main`.
2. Check the commit identity. Use the noreply address for both name and email, and never your real name or Gmail:
   ```
   git config user.name sreevarshan-xenoz
   git config user.email 178558691+sreevarshan-xenoz@users.noreply.github.com
   ```
3. Copy the local pre-push hook by hand from the Linux clone's `.git/hooks/pre-push` (USB, or paste it). It is never
   committed, because it lists the real identifiers it blocks. Without it, nothing stops a private string from being
   pushed.
4. Python:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\pip install -r requirements-riya.txt
   ```
5. Body: in `body`, run `npm ci`. Electron's normal install works on Windows.
6. The local model and llama.cpp go in `%USERPROFILE%\models\riya` and `%USERPROFILE%\llama.cpp\llama-server.exe`.
   You only need these to run the whole of Riya; `npm start` in `body` shows the body on its own, with "not connected".
7. Her data (memory db, `secrets.env`, `providers.env`) lives in `%LOCALAPPDATA%\Riya`, not the repo. Provider keys
   aren't synced between laptops: copy `providers.env` by hand if you want them on Windows too.
8. Merging: do it locally (`git merge --no-ff`) and push `main`. GitHub's web merge button stamps the account's
   primary email and real name on the merge commit.

## How the body draws her today (what a 3D model replaces)

The body is an Electron app in `body/`. It is plain scripts, with no bundler and no framework.

- **The renderer is `body/renderer.js`.** It reacts to the core's websocket messages (switch at about line 405):

  | Message | What it carries |
  |---|---|
  | `say` / `say_end` | Speech bubble text |
  | `mouth` | `level` 0..1 while she speaks: drive lip-sync from this |
  | `expression` | `name`: one of the 14 below |
  | `state` | Base expression, presence, feelings |
  | `look` | `side`, `watching`: where she looks |

- **Expressions:** neutral, happy, excited, missed_you, pouty, flustered, teasing, worried, sad, sleepy, curious,
  thinking, surprised, covering_eyes. The list is in `body/sprite-manifest.schema.json`.
- **Frame choice:** `renderFrame()` (around line 528) picks a frame through `frameFor({manifest, expression,
  mouthOpen, blinking})`.
- **Pure logic lives in `body/src/` and is tested with `node --test`:**
  - `sprite-logic.js`: manifest checks, frame fallback, mouth gate, blink, breathing.
  - `stage-logic.js`: which expression wins.
  - `pet-logic.js`: Little Riya walking, sitting and sleeping, plus reduced motion.
- **Art packs:**
  - `pack-path.js` finds the pack: the `sprite_pack` setting, then `sprites/riya-v1`, then `body/placeholder-pack`.
  - The placeholder pack is generated SVGs (`body/tools/make-placeholder-pack.js`).
  - Real character art is private and git-ignored (`/sprites/`, `riya images/`, and now `*.vrm`, `*.glb`, `*.gltf`,
    `*.fbx`, `*.blend`).

## Suggested route to a 3D Riya

1. **Format: VRM.** It is the standard humanoid avatar format, with expressions, blink, visemes and look-at built in.
   You can make one in VRoid Studio on Windows, or buy or commission one. Render it with three.js plus
   `@pixiv/three-vrm`, copied into `body/vendor/` at exact versions so the renderer stays bundler-free.
2. **Keep the same contract.** Map messages to the model like this:

   | Message | VRM equivalent |
   |---|---|
   | each of the 14 expressions | VRM expression presets or custom blendshapes |
   | `mouth.level` | the `aa` viseme (smoothed, like `MouthGate`) |
   | blink timer | `blink` |
   | `look.side` | the look-at target |
   | breathing and lean | small spine and chest bone motion |

   Put the mapping in a pure `body/src/avatar-logic.js` with tests, the same way `sprite-logic.js` is done.
3. **Keep sprites as the fallback.** Add a setting along the lines of `avatar: "sprites" | "vrm"` and a model path
   that lives outside the repo (like `sprite_pack`). If WebGL or the model fails to load, she falls back to sprites.
4. **Budget.** The body has a CPU budget check (`evaluation.pet_cpu`). Render only when something changes or while
   she is animating, cap the frame rate at about 30 fps, use a small canvas, and honour reduced motion. The Linux
   laptop's MX250 is also running the model server, so keep the scene light.
5. **Little Riya (buddy/pet mode)** walks, sits and sleeps through `pet-logic.js`. With a VRM those become animation
   clips (VRMA or Mixamo, retargeted); until then a sprite or idle pose is fine.

Brainstorm and write the spec (`docs/superpowers/specs/`) before building, as with every version so far.

## Still open from the Linux side

- **Provider keys:** once they're in `providers.env`, check each provider's real model ids against its `/models`
  list. The Gemini, Mistral, SambaNova and NVIDIA ids are guesses.
- **Cline:** needs `cline auth` before it can join the pool.
- **agy:** left out until its tools can be locked down (it runs with every tool on).
- **opencode's free tier:** one shared daily limit that resets at 00:00 UTC.
