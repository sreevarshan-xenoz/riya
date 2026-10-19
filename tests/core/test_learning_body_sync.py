"""The body's dial bands and twin kinds (body/src/learning-logic.js) must agree with the core's."""
import re
from pathlib import Path

from riya.core.learn import twin
from riya.core.learn.dials import BANDS

JS = Path(__file__).resolve().parents[2] / "body" / "src" / "learning-logic.js"
DIAL_ROW = re.compile(r'\{\s*name:\s*"(\w+)",\s*label:\s*"[^"]+",\s*lo:\s*([\d.]+),\s*hi:\s*([\d.]+)\s*\}')
KIND_ROW = re.compile(r'\{\s*kind:\s*"(\w+)",\s*label:\s*"[^"]+"\s*\}')


def _block(name: str) -> str:
    text = JS.read_text(encoding="utf-8")
    return text.split(f"const {name} = [", 1)[1].split("];", 1)[0]


def test_js_dials_match_core_bands():
    rows = [(m.group(1), (float(m.group(2)), float(m.group(3)))) for m in DIAL_ROW.finditer(_block("DIALS"))]
    assert len(rows) == len({n for n, _ in rows}) == len(BANDS), "every dial exactly once"
    assert dict(rows) == BANDS
    assert [n for n, _ in rows] == list(BANDS), "the core's order"


def test_js_twin_kinds_match_core_kinds():
    kinds = [m.group(1) for m in KIND_ROW.finditer(_block("TWIN_KINDS"))]
    assert tuple(kinds) == twin.KINDS
