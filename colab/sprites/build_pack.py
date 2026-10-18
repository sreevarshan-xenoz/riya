"""Turns the raw generated frames into Riya's sprite pack: transparent PNGs on one canvas plus manifest.json.

  python colab/sprites/build_pack.py --raw <pulled frames dir> --out sprites/riya-v1 [--face-box x0,y0,x1,y1]

Reads <expression>_<closed|open|blink>.png from --raw (base.png, the review sheet and anything else are ignored),
removes the background, fits every frame onto one 768x1344 canvas with the same bottom-centred transform (so she never
jumps between expressions), and writes the frames plus a manifest that validates against body/sprite-manifest.schema.json.
Frames that are absent are omitted from the manifest and printed as `<expression>.<variant>`. Re-running overwrites the
pack and removes frames that are no longer in --raw. rembg is imported only when the real remover is used.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from anywhere

from PIL import Image

from colab.sprites.common import EXPRESSION_PROMPTS, VARIANT_PROMPTS, existing_pngs, save_image_atomic
from colab.sprites.gen_expressions import Box, face_box, parse_face_box

CANVAS = (768, 1344)
RAW_SIZE = (832, 1216)  # what gen_expressions generates; the face box is in these coordinates
CROP_WIDTH, CROP_ABOVE, CROP_BELOW = 2.0, 0.15, 0.7  # head and shoulders, in face-box widths / heights
SCHEMA = Path(__file__).resolve().parents[2] / "body" / "sprite-manifest.schema.json"


def _fit(size: tuple[int, int], canvas: tuple[int, int]) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """(scale, scaled size, paste offset) for `size` fitted inside `canvas` and bottom-centred."""
    scale = min(canvas[0] / size[0], canvas[1] / size[1])
    scaled = (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))
    return scale, scaled, ((canvas[0] - scaled[0]) // 2, canvas[1] - scaled[1])


def normalise(img: Image.Image, canvas: tuple[int, int] = CANVAS) -> Image.Image:
    """Scale `img` to fit inside `canvas` (aspect kept), pasted bottom-centred onto a transparent canvas."""
    _, scaled, offset = _fit(img.size, canvas)
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    out.paste(img.convert("RGBA").resize(scaled, Image.LANCZOS), offset)
    return out


def upper_crop(raw_size: tuple[int, int], box: Box, canvas: tuple[int, int] = CANVAS) -> list[int]:
    """[x, y, w, h] in canvas pixels framing head and shoulders: centred on the face box at CROP_WIDTH x its width,
    from CROP_ABOVE x its height above it to CROP_BELOW x its height below it, mapped through the normalise transform
    and clamped to the canvas."""
    scale, _, (ox, oy) = _fit(raw_size, canvas)
    x0, y0, x1, y1 = box
    fw, fh, cx = x1 - x0, y1 - y0, (x0 + x1) / 2
    rx0, rx1 = cx - CROP_WIDTH * fw / 2, cx + CROP_WIDTH * fw / 2
    ry0, ry1 = y0 - CROP_ABOVE * fh, y1 + CROP_BELOW * fh
    left, top = max(0, math.floor(rx0 * scale + ox)), max(0, math.floor(ry0 * scale + oy))
    right, bottom = min(canvas[0], math.ceil(rx1 * scale + ox)), min(canvas[1], math.ceil(ry1 * scale + oy))
    return [left, top, max(1, right - left), max(1, bottom - top)]


def default_remover() -> Callable[[Image.Image], Image.Image]:
    """rembg with the anime model and alpha matting (clean silver hair edges). rembg loads here, not at import."""
    from rembg import new_session, remove

    session = new_session("isnet-anime")
    return lambda img: remove(img, session=session, alpha_matting=True)


def _write_json_atomic(data: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _validate(manifest: dict) -> None:
    from jsonschema import Draft7Validator

    errors = list(Draft7Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).iter_errors(manifest))
    if errors:
        raise ValueError("manifest does not match sprite-manifest.schema.json: " + "; ".join(e.message for e in errors))


def build_pack(
    raw_dir: Path, out_dir: Path, remove_bg: Callable[[Image.Image], Image.Image], name: str = "riya-v1",
    face_override: Box | None = None,
) -> list[str]:
    """Build the pack in `out_dir` from `raw_dir`; returns the missing `<expression>.<variant>` names (never written out).

    Raises ValueError if neutral_closed.png is missing or a raw frame's size differs from the others, before writing anything.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    found = {p.name for p in existing_pngs(raw_dir)}
    cells = [(e, v) for e in EXPRESSION_PROMPTS for v in VARIANT_PROMPTS]
    present = [(e, v) for e, v in cells if f"{e}_{v}.png" in found]
    if ("neutral", "closed") not in present:
        raise ValueError(f"{raw_dir / 'neutral_closed.png'} is missing: the pack needs at least neutral_closed.png")
    sizes = {}
    for e, v in present:
        with Image.open(raw_dir / f"{e}_{v}.png") as im:
            sizes[f"{e}_{v}.png"] = im.size
    size = sizes["neutral_closed.png"]
    odd = [f"{n} is {w}x{h}" for n, (w, h) in sizes.items() if (w, h) != size]
    if odd:
        raise ValueError(f"every frame must be {size[0]}x{size[1]} like neutral_closed.png (one transform for all): " + ", ".join(odd))

    box = face_box(*RAW_SIZE, face_override)
    manifest = {
        "name": name, "version": 1, "canvas": {"width": CANVAS[0], "height": CANVAS[1]},
        "crops": {"upper": upper_crop(size, box)}, "expressions": {},
    }
    for e, v in present:
        manifest["expressions"].setdefault(e, {})[v] = f"{e}_{v}.png"
    _validate(manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    for e, v in present:
        with Image.open(raw_dir / f"{e}_{v}.png") as im:
            save_image_atomic(normalise(remove_bg(im.convert("RGB"))), out_dir / f"{e}_{v}.png")
        print("wrote", out_dir / f"{e}_{v}.png", flush=True)
    _write_json_atomic(manifest, out_dir / "manifest.json")
    frame_names = {f"{e}_{v}.png" for e, v in cells}
    keep = {f"{e}_{v}.png" for e, v in present}
    for stale in out_dir.glob("*.png"):  # frames of an earlier build, and leftovers of an interrupted one
        if stale.name.endswith(".tmp.png") or (stale.name in frame_names and stale.name not in keep):
            stale.unlink()
    return [f"{e}.{v}" for e, v in cells if (e, v) not in present]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder of <expression>_<variant>.png frames")
    ap.add_argument("--out", type=Path, required=True, help="pack folder to write, e.g. sprites/riya-v1")
    ap.add_argument("--name", default="riya-v1", help="pack name in the manifest")
    ap.add_argument("--face-box", type=parse_face_box, default=None, metavar="X0,Y0,X1,Y1",
                    help="face box in the 832x1216 raw frames (use the one you passed to gen_expressions)")
    args = ap.parse_args(argv)
    missing = build_pack(args.raw, args.out, default_remover(), args.name, args.face_box)
    print(f"pack written to {args.out}")
    print("missing frames:", ", ".join(missing) if missing else "none")


if __name__ == "__main__":
    main()
