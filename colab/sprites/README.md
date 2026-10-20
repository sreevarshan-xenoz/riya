# Riya's sprite pack: runbook

Generation runs on a free Colab T4; the pack is built on the laptop. Nothing generated is ever committed: Colab writes
to `MyDrive/jarvis/sprites/` and the laptop to `sprites/` (gitignored). Every step can be re-run after a disconnect and
skips what already exists.

The repo and the `riya images/` folder are uploaded to `MyDrive/jarvis/`. In the notebook, `J` is a Python variable;
IPython expands `$J` inside `!` lines.

## 1. Colab: set up (T4 GPU runtime)

```python
from google.colab import drive
drive.mount("/content/drive")
J = "/content/drive/MyDrive/jarvis"
```

```
!pip install -U diffusers transformers accelerate safetensors peft
%cd /content/drive/MyDrive/jarvis/riya
```

(`riya` here is the repo folder; use whatever name you uploaded it under.)

`--lora <file>.safetensors` (optional, with `--lora-scale`) adds a character LoRA to both generators.

## 2. Colab: outfit candidates, then the pick

```
!python colab/sprites/gen_outfits.py --refs "$J/riya images" --out $J/sprites/outfits
```

Open `$J/sprites/outfits/outfits_sheet.png`, pick one cell (outfit id and seed, e.g. `pink_hoodie #22`), and write it to
`$J/sprites/choice.json`:

```json
{"outfit": "pink_hoodie", "seed": 22}
```

## 3. Colab: check the face box

```
!python colab/sprites/gen_expressions.py --choice $J/sprites/choice.json --refs "$J/riya images" --out $J/sprites/frames --preview-box
```

This copies the picked candidate to `frames/base.png`, writes `frames/base_box.png` and stops. Check it:

- red: the face box, hairline to chin and ear to ear (everything an expression may change);
- magenta: the eye band (repainted for `blink`); yellow: the mouth (repainted for `open`);
- cyan: where the hands go for `covering_eyes`.

If the boxes are off, add `--face-box x0,y0,x1,y1` (pixels on the 832x1216 base) and preview again until they fit.

## 4. Colab: generate the frames

Run the same command without `--preview-box` (keep the `--face-box` you settled on, if any):

```
!python colab/sprites/gen_expressions.py --choice $J/sprites/choice.json --refs "$J/riya images" --out $J/sprites/frames
```

It writes `<expression>_<closed|open|blink>.png` for the 14 expressions (41 frames: `covering_eyes` has no blink, her
hands cover her eyes and the body falls back to closed), `review_sheet.png` and `face_sheet.png`, and
records the outfit, seed, face box and reroll counts in `frames/run.json`. After a disconnect, run the same command
again: it resumes. It refuses (exit 2) if `choice.json` or `--face-box` no longer match `run.json`; `--reset-run`
accepts the change (frames already made are kept; the face box stays the recorded one unless you pass `--face-box`).

## 5. Colab: review, and reroll what's wrong

Look at `frames/review_sheet.png` (whole frames) and `frames/face_sheet.png` (the faces at full resolution: same face
shape and eye colour everywhere?). To redo an expression:

```
!python colab/sprites/gen_expressions.py --choice $J/sprites/choice.json --refs "$J/riya images" --out $J/sprites/frames --reroll pouty,sad
```

That deletes the frames of each named expression (three, two for `covering_eyes`) and regenerates them with the next seed for it. Each `--reroll`
moves on one seed, so if the session drops mid-reroll, resume **without** `--reroll`. `--only pouty,sad` is different:
it only restricts which expressions are generated and never redoes a finished frame. Repeat until every row is right.

Before closing the session, flush Drive so the last writes land:

```python
drive.flush_and_unmount()
```

## 6. Laptop: build the pack and start the body

Pull the whole `MyDrive/jarvis/sprites/frames/` folder, `run.json` included, to `sprites/frames/` (or point `--raw` at
`G:\My Drive\jarvis\sprites\frames`). Then, from the repo root:

```
.venv/Scripts/python.exe -m pip install "rembg[cpu]" jsonschema
.venv/Scripts/python.exe colab/sprites/build_pack.py --raw sprites/frames --out sprites/riya-v1
```

Expect `missing frames: none`. The face box comes from `run.json` (`--face-box` overrides it). Then start the body
(`scripts/start_riya.ps1`, or `npm --prefix body start`); it picks up `sprites/riya-v1/` by itself.

## 7. Manual checks

- Silver hair edges are clean against a dark desktop and against a white window.
- Switching expressions doesn't make her jump, and her outline doesn't flicker.
- The panel crop shows her head and shoulders, and the chat still fits beside it.
- Her mouth moves when she talks, blinks close only her eyes, and her face doesn't shimmer on those swaps.
- `covering_eyes` shows both hands over her eyes, with no cut-off fingers at the mask edge.

## 8. A walk cycle (optional, your step)

Little Riya walks along the bottom of your screen in buddy mode. Without a walk cycle she slides her idle frames along with a
small step bob and sway, which already works, so skip this unless you want real steps.

1. Make 2 to 16 frames of her walking, facing right (her natural side; the body mirrors the frames when she walks left). Keep
   the same canvas as the other frames (768 x 1344 for `riya-v1`), a transparent background, and her feet at the same height
   in every frame so she doesn't hover or sink. Name them `walk_01.png`, `walk_02.png` and so on, and put them in
   `sprites/riya-v1/`, next to the other frames.
2. List them, in order, in `sprites/riya-v1/manifest.json` under a new top-level `motions` key. The cycle loops at 8 frames a
   second:

   ```json
   "motions": { "walk": ["walk_01.png", "walk_02.png", "walk_03.png", "walk_04.png"] }
   ```

   `motions` may also hold `sit` and `sleep` the same way (the body ignores them for now). Only those three names are allowed, each with 1 to 16 file names
   (`body/sprite-manifest.schema.json` is the contract), and every name must be a plain relative path inside the pack: a
   missing or escaping file is skipped.
3. Restart the body. `build_pack.py` never writes `motions`, so if you rebuild the pack, add the key again afterwards.

Under reduced motion (Settings, or Windows' own switch) she doesn't walk at all, so the cycle never plays.
