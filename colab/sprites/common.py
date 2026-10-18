"""Prompts, pipeline loading and safe output writing for Riya's sprite pack (shared by every generator).

The module imports without torch or diffusers; they load inside load_pipeline only.
"""
import os
from pathlib import Path
from typing import Literal

BASE_PROMPT = (
    "1girl, solo, alya, long silver hair, blue eyes, red hair ribbon, upper body to thighs, standing, "
    "facing viewer, simple grey background, anime style, masterpiece, best quality"
)
NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, text, watermark, signature, multiple girls, cropped head, nsfw"

OUTFITS = {
    "sweater_plaid": "black off-shoulder ribbed sweater, grey plaid skirt",
    "pink_hoodie": "oversized pale pink hoodie, black leggings",
    "jacket_skirt": "black and red track jacket worn off shoulders, white sleeveless top, pink skirt, black thighhighs",
    "white_tee": "white loose t-shirt, denim shorts",
}

EXPRESSION_PROMPTS = {
    "neutral": "calm gentle expression",
    "happy": "happy smile",
    "excited": "excited, sparkling eyes, wide smile",
    "missed_you": "happy tears, teary eyes, relieved smile, blush",
    "pouty": "pouting, puffed cheeks, looking away, blush",
    "flustered": "flustered, heavy blush, embarrassed, wavy mouth",
    "teasing": "smug, teasing smirk, half-closed eyes",
    "worried": "worried, furrowed brows, concerned",
    "sad": "sad, downcast eyes, frown",
    "sleepy": "sleepy, drowsy half-closed eyes, yawning",
    "curious": "curious, head tilt, raised eyebrow",
    "thinking": "thinking, looking up, finger on chin",
    "surprised": "surprised, wide eyes, small open mouth",
    "covering_eyes": "covering eyes with both hands, peeking, blush",
}

VARIANT_PROMPTS = {"closed": "closed mouth", "open": "open mouth, talking", "blink": "closed eyes, closed mouth"}

MODEL = "cagliostrolab/animagine-xl-3.1"
IP_ADAPTER = ("h94/IP-Adapter", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.safetensors")
IP_ENCODER_FOLDER = "models/image_encoder"  # the plus adapter needs its ViT-H encoder
IP_SCALE = 0.6


def load_pipeline(kind: Literal["txt2img", "inpaint"], lora: Path | None = None, lora_scale: float = 0.8):
    """SDXL anime pipeline on CUDA with the IP-Adapter (plus, ViT-H) and an optional LoRA. Heavy imports stay in here."""
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline

    cls = {"txt2img": StableDiffusionXLPipeline, "inpaint": StableDiffusionXLInpaintPipeline}[kind]
    try:
        pipe = cls.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16")
    except (ValueError, OSError):  # no fp16 variant published
        pipe = cls.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.to("cuda")
    repo, subfolder, weight_name = IP_ADAPTER
    pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name, image_encoder_folder=IP_ENCODER_FOLDER)
    pipe.set_ip_adapter_scale(IP_SCALE)
    if lora:
        pipe.load_lora_weights(str(Path(lora).parent), weight_name=Path(lora).name, adapter_name="riya")
        pipe.set_adapters(["riya"], adapter_weights=[lora_scale])
    return pipe


def save_image_atomic(image, path: Path) -> Path:
    """Save as PNG via a .tmp.png sibling then os.replace, so an interrupted run never leaves a truncated file."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.png")
    try:
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def existing_pngs(folder: Path, pattern: str = "*.png") -> set[Path]:
    """Finished outputs in `folder` matching `pattern`; leftover .tmp.png files don't count."""
    return {p for p in Path(folder).glob(pattern) if not p.name.endswith(".tmp.png")}
