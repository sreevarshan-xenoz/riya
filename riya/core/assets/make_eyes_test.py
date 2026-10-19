"""Writes eyes_test.png, the picture the Health page's eye test sends instead of the screen: 640x200, white, with
"Riya eye test 42" centred in black. Deterministic; the PNG is committed, so this only runs again if the text changes.

  .venv\\Scripts\\python riya/core/assets/make_eyes_test.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TEXT = "Riya eye test 42"
SIZE = (640, 200)
OUT = Path(__file__).with_name("eyes_test.png")


def make(out: Path = OUT) -> Path:
    im = Image.new("L", SIZE, 255)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=64)
    left, top, right, bottom = draw.textbbox((0, 0), TEXT, font=font)
    xy = ((SIZE[0] - (right - left)) / 2 - left, (SIZE[1] - (bottom - top)) / 2 - top)
    draw.text(xy, TEXT, fill=0, font=font)
    im.save(out, format="PNG", optimize=True)
    return out


if __name__ == "__main__":
    print(make())
