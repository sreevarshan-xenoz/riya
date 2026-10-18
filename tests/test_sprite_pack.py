import json
import subprocess
import sys
from pathlib import Path

import pytest

from PIL import Image

from colab.sprites import gen_outfits
from colab.sprites.common import (
    EXPRESSION_PROMPTS, OUTFITS, VARIANT_PROMPTS, existing_pngs, load_pipeline, save_image_atomic,
)
from colab.sprites.review_sheet import make_sheet


def protocol_expressions():
    return json.loads(Path("docs/protocol.schema.json").read_text(encoding="utf-8"))["expressions"]


def test_prompts_cover_protocol_expressions_exactly():
    assert sorted(EXPRESSION_PROMPTS) == sorted(protocol_expressions())
    assert set(VARIANT_PROMPTS) == {"closed", "open", "blink"} and len(OUTFITS) == 4


def test_make_sheet_draws_a_labelled_grid(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"t{i}.png"
        Image.new("RGB", (80, 120), (40 * i, 90, 160)).save(p)
        paths.append(p)
    out = make_sheet(paths, [f"label {i}" for i in range(5)], tmp_path / "sheet.png", cols=2)
    assert out == tmp_path / "sheet.png"
    with Image.open(out) as sheet:
        assert sheet.width > 160 and sheet.height > 3 * 120  # 2 cols x 3 rows plus label strips


def test_save_image_atomic_writes_png_without_leftovers(tmp_path):
    out = save_image_atomic(Image.new("RGB", (8, 8), "red"), tmp_path / "a.png")
    assert out == tmp_path / "a.png" and Image.open(out).size == (8, 8)
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


def test_save_image_atomic_failure_leaves_no_partial_file(tmp_path):
    class Dying:
        def save(self, path, format=None):
            Path(path).write_bytes(b"half a file")
            raise OSError("session dropped")

    with pytest.raises(OSError):
        save_image_atomic(Dying(), tmp_path / "a.png")
    assert list(tmp_path.iterdir()) == []


def test_existing_pngs_ignores_tmp_leftovers(tmp_path):
    for name in ("outfit_a_1.png", "outfit_b_1.tmp.png", "other.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
    assert existing_pngs(tmp_path, "outfit_*.png") == {tmp_path / "outfit_a_1.png"}


def test_gen_outfits_resumes_without_torch_when_everything_exists(tmp_path):
    for oid in OUTFITS:
        for seed in (11, 22):
            Image.new("RGB", (32, 48)).save(tmp_path / f"outfit_{oid}_{seed}.png")
    gen_outfits.main(["--refs", str(tmp_path / "no refs needed"), "--out", str(tmp_path)])
    assert (tmp_path / "outfits_sheet.png").exists()


def test_common_imports_without_torch_or_diffusers():
    code = (
        "import sys, colab.sprites.common as c, colab.sprites.gen_outfits;"
        "assert callable(c.load_pipeline) and 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])
