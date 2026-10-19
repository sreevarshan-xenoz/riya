"""The body's Health lists (body/src/health-logic.js) must agree with the core's rows, fixes and confirm gate."""
import re
from pathlib import Path

from riya.core import health, setup

JS = Path(__file__).resolve().parents[2] / "body" / "src" / "health-logic.js"
PAIR = re.compile(r'^\s*(\w+):\s*"([^"]+)",?\s*$', re.M)


def _text() -> str:
    return JS.read_text(encoding="utf-8")


def _block(name: str) -> str:
    m = re.search(r"const " + name + r" = \{\n(.*?)\n\};", _text(), re.S)
    assert m, f"health-logic.js declares {name} one pair per line"
    return m.group(1)


def js_row_labels() -> dict[str, str]:
    """ROW_LABELS from health-logic.js, in source order."""
    return dict(PAIR.findall(_block("ROW_LABELS")))


def js_confirm_gated() -> list[str]:
    m = re.search(r"const CONFIRM_GATED = \[([^\]]*)\]", _text())
    assert m, "health-logic.js declares CONFIRM_GATED"
    return re.findall(r'"(\w+)"', m.group(1))


def js_fix_labels() -> dict[str, str]:
    return dict(PAIR.findall(_block("FIX_LABELS")))


def test_js_confirm_gated_matches_the_core():
    assert js_confirm_gated() == list(setup.CONFIRM_GATED)


def test_js_row_labels_match_the_cores_rows():
    labels = js_row_labels()
    assert all(labels.values())
    assert list(labels) == list(health.ROWS)


def test_js_fix_labels_cover_the_cores_fixes():
    assert set(js_fix_labels()) | {"autostart"} == set(health.FIXES)


def test_health_check_labels_match():
    """The live check's LABELS is the third copy of the row labels: it can't drift from the core or the body."""
    from evaluation import health_check
    assert health_check.LABELS == js_row_labels()
    assert list(health_check.LABELS) == list(health.ROWS)
