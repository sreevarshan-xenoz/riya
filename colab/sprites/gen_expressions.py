"""Riya's base figure and expression frames: one base image, then only the face is inpainted. Run on a Colab T4.

  python colab/sprites/gen_expressions.py --choice <drive>/jarvis/sprites/choice.json
      --refs "<drive>/jarvis/riya images" --out <drive>/jarvis/sprites/frames [--only pouty,sad] [--face-box x0,y0,x1,y1]

base.png is the picked candidate outfit_<id>_<seed>.png copied from --outfits (regenerated with the same prompt and
seed only if that file is missing). Then writes <expression>_<closed|open|blink>.png for the 14 expressions,
and review_sheet.png (14 rows x 3 columns). Resumable: existing files are skipped; delete a frame and re-run (or pass --only) to redo it.
Each expression's closed frame is inpainted from base.png; its open and blink frames from that closed frame.
"""
import argparse
import gc
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from anywhere

from PIL import Image, ImageFilter

from colab.sprites.common import (
    BASE_PROMPT, EXPRESSION_PROMPTS, NEGATIVE, OUTFITS, VARIANT_PROMPTS, existing_pngs, load_pipeline, save_image_atomic,
)
from colab.sprites.gen_outfits import SIZE, load_refs, outfit_prompt
from colab.sprites.review_sheet import make_sheet

Box = tuple[int, int, int, int]
DEFAULT_FACE_BOX: Box = (256, 60, 576, 420)  # the head on the 832x1216 base
HANDS_EXTENSION = 280  # covering_eyes: the mask runs this far below the face box to take in the hands
FACE_STRENGTH, HANDS_STRENGTH = 0.55, 0.75
FEATHER = 8  # Gaussian radius that softens the mask edge


def face_box(width: int, height: int, override: Box | None) -> Box:
    """The inpaint box: `override` if given, else the fixed head box (the base is always 832x1216)."""
    return override or DEFAULT_FACE_BOX


def stable_seed(expression: str, seed: int) -> int:
    """Per-expression seed. crc32, not hash(): str hashes are randomized per process and must not vary between runs."""
    return zlib.crc32(expression.encode()) % 10000 + seed


def face_mask(size: tuple[int, int], box: Box, extend_down: int = 0) -> Image.Image:
    """Inpaint mask (white = repaint) for `box`, extended down by `extend_down` px, clamped to the image and feathered."""
    width, height = size
    x0, y0, x1, y1 = box
    mask = Image.new("L", size, 0)
    mask.paste(255, (max(0, x0), max(0, y0), min(width, x1), min(height, y1 + extend_down)))
    return mask.filter(ImageFilter.GaussianBlur(FEATHER))


def job_list(expressions: list[str], variants: list[str], existing: set[str]) -> list[tuple[str, str]]:
    """(expression, variant) pairs whose `<expression>_<variant>.png` is not in `existing`, in the given order."""
    return [(e, v) for e in expressions for v in variants if f"{e}_{v}.png" not in existing]


def parse_only(text: str) -> list[str]:
    """`--only` value: comma-separated expression names, each one of EXPRESSION_PROMPTS."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPRESSION_PROMPTS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown expression(s) {', '.join(unknown) or text!r}; choose from: {', '.join(EXPRESSION_PROMPTS)}"
        )
    return names


def parse_face_box(text: str) -> Box:
    """`--face-box` value: x0,y0,x1,y1 with x0 < x1 and y0 < y1."""
    try:
        x0, y0, x1, y1 = (int(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected x0,y0,x1,y1 as four integers, got {text!r}") from None
    if x0 >= x1 or y0 >= y1:
        raise argparse.ArgumentTypeError(f"face box must satisfy x0 < x1 and y0 < y1, got {text!r}")
    return x0, y0, x1, y1


def load_choice(path: Path) -> tuple[str, int]:
    """(outfit id, seed) from the picked choice.json."""
    try:
        choice = json.loads(Path(path).read_text(encoding="utf-8"))
        outfit, seed = choice["outfit"], int(choice["seed"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise SystemExit(f'cannot read {path}: expected {{"outfit": "...", "seed": N}} ({err!r})') from None
    if outfit not in OUTFITS:
        raise SystemExit(f"{path}: unknown outfit {outfit!r}; choose from: {', '.join(OUTFITS)}")
    return outfit, seed


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


def run_jobs(jobs, out: Path, outfit: str, seed: int, box: Box, inpaint) -> None:
    """Render each (expression, variant) frame with `inpaint(image, mask, prompt, strength, seed) -> image`.

    closed starts from base.png; open and blink start from that expression's closed frame (made first if missing).
    covering_eyes gets a second, hands pass over the extended mask. Each pass is composited back through its own
    feathered mask, so pixels outside the mask stay exactly as they were (no VAE round-trip drift in the body).
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
        passes = [(face_mask(base.size, box), FACE_STRENGTH)]
        if expression == "covering_eyes":
            passes.append((face_mask(base.size, box, HANDS_EXTENSION), HANDS_STRENGTH))
        prompt, frame_seed = frame_prompt(outfit, expression, variant), stable_seed(expression, seed)
        for mask, strength in passes:
            image = Image.composite(inpaint(image, mask, prompt, strength, frame_seed).convert("RGB"), image, mask)
        save_image_atomic(image, path)
        print("wrote", path, flush=True)

    for expression, variant in jobs:
        render(expression, variant)


def write_review_sheet(out: Path) -> Path:
    """review_sheet.png: one row per expression in protocol order, columns closed/open/blink, missing frames blank."""
    done = {p.name for p in existing_pngs(out)}
    cells = [(e, v) for e in EXPRESSION_PROMPTS for v in VARIANT_PROMPTS]
    paths = [out / f"{e}_{v}.png" if f"{e}_{v}.png" in done else None for e, v in cells]
    return make_sheet(paths, [f"{e}/{v}" for e, v in cells], out / "review_sheet.png", cols=len(VARIANT_PROMPTS))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--choice", type=Path, required=True, help='choice.json: {"outfit": "...", "seed": N}')
    ap.add_argument("--refs", type=Path, required=True, help="folder of reference images")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--outfits", type=Path, default=None, help="outfit candidates folder (default: <out>/../outfits)")
    ap.add_argument("--lora", type=Path, default=None, help="optional character LoRA (.safetensors)")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--only", type=parse_only, default=None, help="comma-separated expressions to generate, e.g. pouty,sad")
    ap.add_argument("--face-box", type=parse_face_box, default=None, metavar="X0,Y0,X1,Y1",
                    help=f"face inpaint box (default {','.join(map(str, DEFAULT_FACE_BOX))}); set it after looking at base.png")
    args = ap.parse_args(argv)
    outfit, seed = load_choice(args.choice)
    args.out.mkdir(parents=True, exist_ok=True)
    box = face_box(*SIZE, args.face_box)

    done = {p.name for p in existing_pngs(args.out)}
    if "base.png" not in done and prepare_base(args.out, args.outfits or args.out.parent / "outfits", outfit, seed):
        done.add("base.png")
    jobs = job_list(args.only or list(EXPRESSION_PROMPTS), list(VARIANT_PROMPTS), done)
    need_base = "base.png" not in done
    if need_base or jobs:
        import torch

        refs = load_refs(args.refs)
        if need_base:
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
        if jobs:
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

            run_jobs(jobs, args.out, outfit, seed, box, inpaint)

    sheet = write_review_sheet(args.out)
    print("sheet:", sheet)


if __name__ == "__main__":
    main()
