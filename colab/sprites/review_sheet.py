"""Labelled contact sheet of generated images, for picking an outfit or reviewing expressions."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H, LABEL_H, PAD = 256, 372, 28, 8
BG, FG = (32, 32, 36), (235, 235, 240)


def make_sheet(paths: list[Path | None], labels: list[str], out: Path, cols: int = 4,
               cell: tuple[int, int] = (CELL_W, CELL_H), crop: tuple[int, int, int, int] | None = None) -> Path:
    """Tile `paths` into a grid of `cell`-sized cells under their `labels` and save it to `out`; a None path leaves its
    cell blank. `crop` (x0, y0, x1, y1) cuts each image first; images larger than `cell` are shrunk to fit it."""
    if len(paths) != len(labels):
        raise ValueError("paths and labels must be the same length")
    cell_w, cell_h = cell
    rows = max(1, -(-len(paths) // cols))
    sheet = Image.new("RGB", (cols * (cell_w + PAD) + PAD, rows * (cell_h + LABEL_H + PAD) + PAD), BG)
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for i, (path, label) in enumerate(zip(paths, labels)):
        x = PAD + (i % cols) * (cell_w + PAD)
        y = PAD + (i // cols) * (cell_h + LABEL_H + PAD)
        if path is not None:
            with Image.open(path) as src:
                img = src.convert("RGB")
            if crop:
                img = img.crop(crop)
            img.thumbnail((cell_w, cell_h))
            sheet.paste(img, (x + (cell_w - img.width) // 2, y + (cell_h - img.height) // 2))
        draw.text((x + 4, y + cell_h + 8), label, fill=FG, font=font)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
