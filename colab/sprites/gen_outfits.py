"""Outfit candidates for Riya: SDXL anime model + IP-Adapter from the reference images. Run on a Colab T4.

  python colab/sprites/gen_outfits.py --refs "<drive>/jarvis/riya images" --out <drive>/jarvis/sprites/outfits

Resumable: existing outfit_<id>_<seed>.png files are skipped. Writes outfits_sheet.png for picking.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run as a script from anywhere

from colab.sprites.common import BASE_PROMPT, NEGATIVE, OUTFITS
from colab.sprites.review_sheet import make_sheet

MODEL = "cagliostrolab/animagine-xl-3.1"
IP_ADAPTER = ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin")
SIZE = (832, 1216)  # (width, height)
IP_SCALE = 0.6
REF_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_refs(folder: Path) -> list:
    """Every reference image in `folder`, sorted by name (names may contain spaces and emoji)."""
    from PIL import Image

    images = []
    for p in sorted(q for q in Path(folder).iterdir() if q.suffix.lower() in REF_SUFFIXES):
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    if not images:
        raise SystemExit(f"no reference images (.jpg/.jpeg/.png/.webp) in {folder}")
    return images


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
    todo = [j for j in jobs if not j[2].exists()]
    if todo:
        import torch
        from diffusers import StableDiffusionXLPipeline

        refs = load_refs(args.refs)
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
        pipe.load_ip_adapter(IP_ADAPTER[0], subfolder=IP_ADAPTER[1], weight_name=IP_ADAPTER[2])
        pipe.set_ip_adapter_scale(IP_SCALE)
        extra = {}
        if args.lora:
            pipe.load_lora_weights(str(args.lora.parent), weight_name=args.lora.name)
            extra["cross_attention_kwargs"] = {"scale": args.lora_scale}
        for oid, seed, path in todo:
            image = pipe(
                prompt=f"{BASE_PROMPT}, {OUTFITS[oid]}",
                negative_prompt=NEGATIVE,
                ip_adapter_image=[refs],  # one adapter, all reference images
                width=SIZE[0],
                height=SIZE[1],
                generator=torch.Generator("cuda").manual_seed(seed),
                **extra,
            ).images[0]
            image.save(path)
            print("wrote", path, flush=True)

    make_sheet([j[2] for j in jobs], [f"{oid} #{seed}" for oid, seed, _ in jobs], args.out / "outfits_sheet.png",
               cols=max(1, len(seeds)))
    print("sheet:", args.out / "outfits_sheet.png")


if __name__ == "__main__":
    main()
