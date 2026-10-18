"""Outfit candidates for Riya: SDXL anime model + IP-Adapter from the reference images. Run on a Colab T4.

  python colab/sprites/gen_outfits.py --refs "<drive>/jarvis/riya images" --out <drive>/jarvis/sprites/outfits

Resumable: existing outfit_<id>_<seed>.png files are skipped. Writes outfits_sheet.png for picking.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from anywhere

from colab.sprites.common import BASE_PROMPT, NEGATIVE, OUTFITS, existing_pngs, load_pipeline, save_image_atomic
from colab.sprites.review_sheet import make_sheet

SIZE = (832, 1216)  # (width, height)
REF_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def square_crop(image):
    """Square crop that keeps the head: top-anchored (0, 0, w, w) for portrait images, centred for landscape ones.
    The IP-Adapter's CLIP processor centre-crops to a square, which would cut the head off a portrait reference."""
    w, h = image.size
    if h > w:
        return image.crop((0, 0, w, w))
    left = (w - h) // 2
    return image.crop((left, 0, left + h, h))


def load_refs(folder: Path) -> list:
    """Every reference image in `folder` sorted by name (names may contain spaces and emoji), as square_crop RGB images."""
    from PIL import Image

    images = []
    for p in sorted(q for q in Path(folder).iterdir() if q.suffix.lower() in REF_SUFFIXES):
        with Image.open(p) as im:
            images.append(square_crop(im.convert("RGB")))
    if not images:
        raise SystemExit(f"no reference images (.jpg/.jpeg/.png/.webp) in {folder}")
    return images


def outfit_prompt(outfit_id: str) -> str:
    return f"{BASE_PROMPT}, {OUTFITS[outfit_id]}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", type=Path, required=True, help="folder of reference images")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--lora", type=Path, default=None, help="optional character LoRA (.safetensors)")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--seeds", default="11,22", help="comma-separated seeds")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(oid, seed, args.out / f"outfit_{oid}_{seed}.png") for oid in OUTFITS for seed in seeds]
    done = existing_pngs(args.out, "outfit_*.png")
    todo = [j for j in jobs if j[2] not in done]
    if todo:
        import torch

        refs = load_refs(args.refs)
        pipe = load_pipeline("txt2img", args.lora, args.lora_scale)
        for oid, seed, path in todo:
            image = pipe(
                prompt=outfit_prompt(oid),
                negative_prompt=NEGATIVE,
                ip_adapter_image=[refs],  # one adapter, all reference images
                width=SIZE[0],
                height=SIZE[1],
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            save_image_atomic(image, path)
            print("wrote", path, flush=True)

    make_sheet([j[2] for j in jobs], [f"{oid} #{seed}" for oid, seed, _ in jobs], args.out / "outfits_sheet.png",
               cols=max(1, len(seeds)))
    print("sheet:", args.out / "outfits_sheet.png")


if __name__ == "__main__":
    main()
