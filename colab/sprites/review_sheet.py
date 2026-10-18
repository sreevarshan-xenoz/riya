"""Labelled contact sheet of generated images, for picking an outfit or reviewing expressions."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H, LABEL_H, PAD = 256, 372, 28, 8
BG, FG = (32, 32, 36), (235, 235, 240)


def make_sheet(paths: list[Path | None], labels: list[str], out: Path, cols: int = 4) -> Path:
    """Tile `paths` into a grid under their `labels` and save it to `out`; a None path leaves its cell blank."""
    if len(paths) != len(labels):
        raise ValueError("paths and labels must be the same length")
    rows = max(1, -(-len(paths) // cols))
    sheet = Image.new("RGB", (cols * (CELL_W + PAD) + PAD, rows * (CELL_H + LABEL_H + PAD) + PAD), BG)
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for i, (path, label) in enumerate(zip(paths, labels)):
        x = PAD + (i % cols) * (CELL_W + PAD)
        y = PAD + (i // cols) * (CELL_H + LABEL_H + PAD)
        if path is not None:
            with Image.open(path) as src:
                img = src.convert("RGB")
            img.thumbnail((CELL_W, CELL_H))
            sheet.paste(img, (x + (CELL_W - img.width) // 2, y + (CELL_H - img.height) // 2))
        draw.text((x + 4, y + CELL_H + 8), label, fill=FG, font=font)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
