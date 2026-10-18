import json
from pathlib import Path

from PIL import Image

from colab.sprites.common import EXPRESSION_PROMPTS, OUTFITS, VARIANT_PROMPTS
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
