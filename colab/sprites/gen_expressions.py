"""Riya's base figure and expression frames: one base image, then only the face is inpainted. Run on a Colab T4.

  pip install -U diffusers transformers accelerate safetensors peft
  python colab/sprites/gen_expressions.py --choice <drive>/jarvis/sprites/choice.json
      --refs "<drive>/jarvis/riya images" --out <drive>/jarvis/sprites/frames
      [--preview-box] [--face-box x0,y0,x1,y1] [--only pouty,sad] [--reroll pouty]

base.png is the picked candidate outfit_<id>_<seed>.png copied from --outfits (regenerated with the same prompt and
seed only if that file is missing). Check the face box first: --preview-box writes base_box.png and stops. Then writes
<expression>_<closed|open|blink>.png for the 14 expressions (41 frames: covering_eyes has no blink, her hands cover her
eyes and the body falls back to closed), review_sheet.png (14 rows x 3 columns) and face_sheet.png
(the same grid of face-box crops at native resolution, to judge identity drift). Each closed frame is inpainted from
base.png over the face box; its open and blink frames from that closed frame, over only the mouth or the eye band.

<out>/run.json records the outfit, seed, face box and reroll counts. A resume with a different outfit, seed or face box
is refused unless --reset-run is given; without --face-box the recorded box is used (also with --reset-run). Resumable: finished frames are
skipped. --only restricts which expressions are generated; it never redoes a finished frame. To redo an expression,
pass --reroll <expression>: its three frames are deleted and regenerated with the next seed (stable_seed + 1000 x its
reroll count). choice.json may also carry "rerolls": {"<expression>": N}.
"""
import argparse
import gc
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from anywhere

from PIL import Image, ImageDraw, ImageFilter

from colab.sprites.common import (
    BASE_PROMPT, EXPRESSION_PROMPTS, NEGATIVE, OUTFITS, VARIANT_PROMPTS, existing_pngs, load_pipeline, save_image_atomic,
    save_json_atomic,
)
from colab.sprites.gen_outfits import SIZE, load_refs, outfit_prompt
from colab.sprites.review_sheet import make_sheet

Box = tuple[int, int, int, int]
DEFAULT_FACE_BOX: Box = (256, 60, 576, 420)  # the head on the 832x1216 base
HANDS_EXTENSION = 280  # covering_eyes: the hands mask runs this far below the face box
HANDS_WIDTH = 1.8  # covering_eyes: the hands mask is this many face-box widths wide, centred on the face
MOUTH = (0.25, 0.6, 0.75, 1.0)  # open: x0, y0, x1, y1 as fractions of the face box (lower 40%, middle 50%)
EYES = (0.1, 0.3, 0.9, 0.55)  # blink: the eye band (30%-55% of the height, middle 80%)
FACE_STRENGTH, HANDS_STRENGTH = 0.55, 0.75
FEATHER = 8  # Gaussian radius that softens the mask edge
NOT_GENERATED = {("covering_eyes", "blink")}  # her hands cover her eyes, so the body falls back blink -> closed
RUN_FILE = "run.json"
REROLL_STEP = 1000  # each reroll moves an expression's seed this far


def stable_seed(expression: str, seed: int) -> int:
    """Per-expression seed. crc32, not hash(): str hashes are randomized per process and must not vary between runs."""
    return zlib.crc32(expression.encode()) % 10000 + seed


def frame_seed(expression: str, seed: int, rerolls: dict[str, int]) -> int:
    """The seed an expression's frames are inpainted with: stable_seed moved on by REROLL_STEP per reroll."""
    return stable_seed(expression, seed) + REROLL_STEP * rerolls.get(expression, 0)


def box_mask(size: tuple[int, int], box: Box) -> Image.Image:
    """Inpaint mask (white = repaint) for `box`, clamped to the image and feathered."""
    width, height = size
    x0, y0, x1, y1 = box
    mask = Image.new("L", size, 0)
    mask.paste(255, (max(0, x0), max(0, y0), min(width, x1), min(height, y1)))
    return mask.filter(ImageFilter.GaussianBlur(FEATHER))


def sub_box(box: Box, fractions: tuple[float, float, float, float]) -> Box:
    """The part of `box` given as (x0, y0, x1, y1) fractions of its width and height."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    fx0, fy0, fx1, fy1 = fractions
    return x0 + round(fx0 * w), y0 + round(fy0 * h), x0 + round(fx1 * w), y0 + round(fy1 * h)


def hands_box(box: Box, size: tuple[int, int]) -> Box:
    """covering_eyes hands region: HANDS_WIDTH face widths, centred, from the face top to HANDS_EXTENSION below it."""
    x0, y0, x1, y1 = box
    centre, half = (x0 + x1) / 2, HANDS_WIDTH * (x1 - x0) / 2
    return max(0, round(centre - half)), max(0, y0), min(size[0], round(centre + half)), min(size[1], y1 + HANDS_EXTENSION)


def frame_passes(expression: str, variant: str, box: Box, size: tuple[int, int]) -> list[tuple[Box, float]]:
    """(region, strength) inpaint passes for one frame: closed repaints the face (plus the hands for covering_eyes);
    open repaints only the mouth and blink only the eye band, so talking and blinking don't repaint the whole face."""
    if variant == "open":
        return [(sub_box(box, MOUTH), FACE_STRENGTH)]
    if variant == "blink":
        return [(sub_box(box, EYES), FACE_STRENGTH)]
    passes = [(box, FACE_STRENGTH)]
    if expression == "covering_eyes":
        passes.append((hands_box(box, size), HANDS_STRENGTH))
    return passes


def job_list(expressions: list[str], variants: list[str], existing: set[str]) -> list[tuple[str, str]]:
    """(expression, variant) pairs whose `<expression>_<variant>.png` is not in `existing`, in the given order
    (NOT_GENERATED frames are never listed)."""
    return [(e, v) for e in expressions for v in variants if (e, v) not in NOT_GENERATED and f"{e}_{v}.png" not in existing]


def parse_expressions(text: str) -> list[str]:
    """`--only` / `--reroll` value: comma-separated expression names, each one of EXPRESSION_PROMPTS."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPRESSION_PROMPTS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown expression(s) {', '.join(unknown) or text!r}; choose from: {', '.join(EXPRESSION_PROMPTS)}"
        )
    return names


def check_face_box(values) -> Box:
    """`values` as a face box: four ints with 0 <= x0 < x1 <= 832 and 0 <= y0 < y1 <= 1216 (inside the base)."""
    if not (isinstance(values, (list, tuple)) and len(values) == 4 and all(type(v) is int for v in values)):
        raise ValueError(f"expected x0,y0,x1,y1 as four integers, got {values!r}")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= SIZE[0] and 0 <= y0 < y1 <= SIZE[1]):
        raise ValueError(f"face box must satisfy 0 <= x0 < x1 <= {SIZE[0]} and 0 <= y0 < y1 <= {SIZE[1]}, got {values!r}")
    return x0, y0, x1, y1


def parse_face_box(text: str) -> Box:
    """`--face-box` value: x0,y0,x1,y1 inside the 832x1216 base (see check_face_box)."""
    try:
        values = [int(p) for p in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected x0,y0,x1,y1 as four integers, got {text!r}") from None
    try:
        return check_face_box(values)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def check_rerolls(value, source: str) -> dict[str, int]:
    """`value` as reroll counts {"<expression>": N >= 0}; ValueError naming `source` otherwise."""
    if not isinstance(value, dict):
        raise ValueError(f'{source}: "rerolls" must be an object like {{"pouty": 1}}, got {value!r}')
    unknown = [k for k in value if k not in EXPRESSION_PROMPTS]
    if unknown:
        raise ValueError(f"{source}: unknown expression(s) in rerolls: {', '.join(unknown)}; "
                         f"choose from: {', '.join(EXPRESSION_PROMPTS)}")
    bad = [k for k, n in value.items() if type(n) is not int or n < 0]
    if bad:
        raise ValueError(f"{source}: reroll counts must be whole numbers >= 0 ({', '.join(bad)})")
    return dict(value)


def load_choice(path: Path) -> tuple[str, int, dict[str, int]]:
    """(outfit id, seed, reroll counts) from the picked choice.json; "rerolls" is optional."""
    try:
        choice = json.loads(Path(path).read_text(encoding="utf-8"))
        outfit, seed = choice["outfit"], int(choice["seed"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise SystemExit(f'cannot read {path}: expected {{"outfit": "...", "seed": N}} ({err!r})') from None
    if outfit not in OUTFITS:
        raise SystemExit(f"{path}: unknown outfit {outfit!r}; choose from: {', '.join(OUTFITS)}")
    try:
        return outfit, seed, check_rerolls(choice.get("rerolls", {}), str(path))
    except ValueError as err:
        raise SystemExit(str(err)) from None


def read_run(folder: Path) -> dict | None:
    """The run record <folder>/run.json ({outfit, seed, face_box, rerolls}), or None if there is none.

    Raises ValueError naming the file if it is malformed."""
    path = Path(folder) / RUN_FILE
    if not path.exists():
        return None
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(run, dict) or run.get("outfit") not in OUTFITS or type(run.get("seed")) is not int:
            raise ValueError("needs a known outfit and an integer seed")
        check_face_box(run.get("face_box"))
        run["rerolls"] = check_rerolls(run.get("rerolls", {}), RUN_FILE)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is malformed ({err}); fix it, or delete it to start a new run record") from None
    return run


def run_mismatches(stored: dict, outfit: str, seed: int, box: Box | None) -> list[str]:
    """How this run differs from the recorded one (`box` None: not compared)."""
    now = {"outfit": outfit, "seed": seed, "face_box": None if box is None else list(box)}
    return [f"{key} is {value!r} but {RUN_FILE} has {stored[key]!r}" for key, value in now.items()
            if value is not None and stored[key] != value]


def merge_rerolls(
    stored: dict[str, int], from_choice: dict[str, int], reroll: list[str],
) -> tuple[dict[str, int], list[str]]:
    """(new counts, expressions whose count changed): the larger of the recorded and choice.json counts, plus one for
    each --reroll name. A changed count means that expression's frames must be regenerated."""
    counts = {e: max(stored.get(e, 0), from_choice.get(e, 0)) + (e in reroll) for e in EXPRESSION_PROMPTS}
    changed = [e for e in EXPRESSION_PROMPTS if counts[e] != stored.get(e, 0)]
    return {e: n for e, n in counts.items() if n}, changed


def prepare_base(out: Path, outfits_dir: Path, outfit: str, seed: int) -> bool:
    """Copy the picked outfit candidate to out/base.png (resized to SIZE if needed). False if the candidate is missing."""
    candidate = Path(outfits_dir) / f"outfit_{outfit}_{seed}.png"
    if not candidate.exists():
        return False
    image = _load(candidate)
    if image.size != SIZE:
        image = image.resize(SIZE, Image.LANCZOS)
    save_image_atomic(image, Path(out) / "base.png")
    print("base.png copied from", candidate, flush=True)
    return True


def frame_prompt(outfit: str, expression: str, variant: str) -> str:
    return f"{BASE_PROMPT}, {OUTFITS[outfit]}, {EXPRESSION_PROMPTS[expression]}, {VARIANT_PROMPTS[variant]}"


def _load(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def run_jobs(jobs, out: Path, outfit: str, seed: int, box: Box, inpaint, rerolls: dict[str, int] | None = None) -> None:
    """Render each (expression, variant) frame with `inpaint(image, mask, prompt, strength, seed) -> image`.

    closed starts from base.png and repaints the face (covering_eyes adds a hands pass); open and blink start from that
    expression's closed frame (made first if missing) and repaint only the mouth or the eye band (see frame_passes).
    Each pass is composited back through its own feathered mask, so pixels outside it stay exactly as they were.
    """
    out = Path(out)
    base = _load(out / "base.png")

    def render(expression: str, variant: str) -> None:
        path = out / f"{expression}_{variant}.png"
        if path.exists():
            return
        if variant == "closed":
            image = base
        else:
            render(expression, "closed")
            image = _load(out / f"{expression}_closed.png")
        prompt, this_seed = frame_prompt(outfit, expression, variant), frame_seed(expression, seed, rerolls or {})
        for region, strength in frame_passes(expression, variant, box, base.size):
            mask = box_mask(base.size, region)
            image = Image.composite(inpaint(image, mask, prompt, strength, this_seed).convert("RGB"), image, mask)
        save_image_atomic(image, path)
        print("wrote", path, flush=True)

    for expression, variant in jobs:
        render(expression, variant)


def write_box_preview(out: Path, box: Box) -> Path:
    """base_box.png: base.png with the hands (cyan), face (red), mouth (yellow) and eye-band (magenta) boxes drawn."""
    image = _load(Path(out) / "base.png")
    draw = ImageDraw.Draw(image)
    regions = [(hands_box(box, image.size), (0, 220, 255)), (box, (255, 40, 40)), (sub_box(box, MOUTH), (255, 220, 0)),
               (sub_box(box, EYES), (255, 0, 255))]
    for (x0, y0, x1, y1), colour in regions:
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colour, width=3)
    return save_image_atomic(image, Path(out) / "base_box.png")


def write_review_sheets(out: Path, box: Box) -> list[Path]:
    """review_sheet.png (whole frames, scaled down) and face_sheet.png (the face box at native resolution, to judge
    drift): one row per expression in protocol order, columns closed/open/blink, missing frames blank."""
    done = {p.name for p in existing_pngs(out)}
    cells = [(e, v) for e in EXPRESSION_PROMPTS for v in VARIANT_PROMPTS]
    paths = [out / f"{e}_{v}.png" if f"{e}_{v}.png" in done else None for e, v in cells]
    labels, cols = [f"{e}/{v}" for e, v in cells], len(VARIANT_PROMPTS)
    face_cell = (box[2] - box[0], box[3] - box[1])
    return [make_sheet(paths, labels, out / "review_sheet.png", cols=cols),
            make_sheet(paths, labels, out / "face_sheet.png", cols=cols, cell=face_cell, crop=box)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--choice", type=Path, required=True,
                    help='choice.json: {"outfit": "...", "seed": N} plus optional "rerolls": {"<expression>": N}')
    ap.add_argument("--refs", type=Path, required=True, help="folder of reference images")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--outfits", type=Path, default=None, help="outfit candidates folder (default: <out>/../outfits)")
    ap.add_argument("--lora", type=Path, default=None, help="optional character LoRA (.safetensors)")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--only", type=parse_expressions, default=None, metavar="EXPR[,EXPR]",
                    help="generate only these expressions' missing frames, e.g. pouty,sad (never redoes a finished "
                         "frame: use --reroll for that)")
    ap.add_argument("--reroll", type=parse_expressions, default=[], metavar="EXPR[,EXPR]",
                    help="redo these expressions: delete their three frames and regenerate them with the next seed "
                         "(bumps their count in <out>/run.json; after a disconnect, resume without --reroll)")
    ap.add_argument("--face-box", type=parse_face_box, default=None, metavar="X0,Y0,X1,Y1",
                    help="face box on the 832x1216 base (default: the one in <out>/run.json, else "
                         f"{','.join(map(str, DEFAULT_FACE_BOX))}); check it with --preview-box")
    ap.add_argument("--preview-box", action="store_true",
                    help="write base_box.png (base.png with the face, hands, mouth and eye boxes drawn) and exit "
                         "before any inpainting")
    ap.add_argument("--reset-run", action="store_true",
                    help="accept a changed outfit, seed or face box and rewrite <out>/run.json (frames already made "
                         "are kept: --reroll the ones to redo)")
    args = ap.parse_args(argv)
    outfit, seed, choice_rerolls = load_choice(args.choice)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        stored = read_run(args.out)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    if stored and not args.reset_run:
        diffs = run_mismatches(stored, outfit, seed, None if args.preview_box else args.face_box)
        if diffs:
            print(f"{args.out / RUN_FILE} records a different run: {'; '.join(diffs)}.\nFix choice.json or --face-box, "
                  "or pass --reset-run to record this run instead (the frames already made are kept).", file=sys.stderr)
            raise SystemExit(2)
    if args.face_box:
        box = args.face_box
    elif stored:  # also with --reset-run: it resets the outfit and seed, not the face box you settled on
        box = tuple(stored["face_box"])
    else:
        box = DEFAULT_FACE_BOX

    done = {p.name for p in existing_pngs(args.out)}
    if "base.png" not in done and prepare_base(args.out, args.outfits or args.out.parent / "outfits", outfit, seed):
        done.add("base.png")
    refs = None
    if "base.png" not in done:
        import torch

        refs = load_refs(args.refs)
        pipe = load_pipeline("txt2img", args.lora, args.lora_scale)
        base = pipe(
            prompt=outfit_prompt(outfit),  # exactly the candidate's prompt and seed, so it reproduces it
            negative_prompt=NEGATIVE,
            ip_adapter_image=[refs],
            width=SIZE[0],
            height=SIZE[1],
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        save_image_atomic(base, args.out / "base.png")
        print("wrote", args.out / "base.png", flush=True)
        del pipe  # a T4 can't hold both pipelines; diffusers pipelines hold reference cycles, so collect first
        gc.collect()
        torch.cuda.empty_cache()
    if args.preview_box:
        print("box preview:", write_box_preview(args.out, box))
        return

    rerolls, changed = merge_rerolls(stored["rerolls"] if stored else {}, choice_rerolls, args.reroll)
    run = {"outfit": outfit, "seed": seed, "face_box": list(box), "rerolls": rerolls}
    if run != stored:
        save_json_atomic(run, args.out / RUN_FILE)
    for expression in changed:
        for variant in VARIANT_PROMPTS:
            (args.out / f"{expression}_{variant}.png").unlink(missing_ok=True)
            done.discard(f"{expression}_{variant}.png")
        print(f"reroll {expression} (#{rerolls.get(expression, 0)}): regenerating its three frames", flush=True)
    wanted = list(args.only or EXPRESSION_PROMPTS)
    wanted += [e for e in changed if e not in wanted]
    jobs = job_list(wanted, list(VARIANT_PROMPTS), done)
    if jobs:
        import torch

        refs = refs or load_refs(args.refs)
        pipe = load_pipeline("inpaint", args.lora, args.lora_scale)

        def inpaint(image, mask, prompt, strength, frame_seed):
            return pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                image=image,
                mask_image=mask,
                strength=strength,
                ip_adapter_image=[refs],
                width=image.width,
                height=image.height,
                generator=torch.Generator("cuda").manual_seed(frame_seed),
            ).images[0]

        run_jobs(jobs, args.out, outfit, seed, box, inpaint, rerolls)

    for sheet in write_review_sheets(args.out, box):
        print("sheet:", sheet)


if __name__ == "__main__":
    main()
