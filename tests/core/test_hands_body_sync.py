"""The body's tool list (body/src/hands-logic.js) must agree with the core's tiers."""
import re
from pathlib import Path

from riya.core.hands.tools import DEFAULT_OFF, TIERS

JS = Path(__file__).resolve().parents[2] / "body" / "src" / "hands-logic.js"
ROW = re.compile(r'\{\s*name:\s*"(\w+)",\s*tier:\s*(\d),\s*label:\s*"[^"]+"\s*\}')


def _js_tools() -> list[tuple[str, int]]:
    text = JS.read_text(encoding="utf-8")
    block = text.split("const TOOLS = [", 1)[1].split("];", 1)[0]
    return [(m.group(1), int(m.group(2))) for m in ROW.finditer(block)]


def test_js_tools_match_core_tiers():
    tools = _js_tools()
    assert len(tools) == len({n for n, _ in tools}) == 17, "every tool exactly once"
    assert dict(tools) == TIERS


def test_js_default_off_matches_core():
    text = JS.read_text(encoding="utf-8")
    m = re.search(r"const DEFAULT_OFF = \[([^\]]*)\]", text)
    assert m, "hands-logic.js declares DEFAULT_OFF"
    assert set(re.findall(r'"(\w+)"', m.group(1))) == set(DEFAULT_OFF)


def test_js_limits_match_core_settings():
    from riya.core import settings as s
    text = JS.read_text(encoding="utf-8")
    consts = dict(re.findall(r"const (MAX_\w+) = (\d+);", text))
    assert {k: int(v) for k, v in consts.items()} == {
        "MAX_APPS": s.MAX_APPS, "MAX_COMMANDS": s.MAX_COMMANDS, "MAX_HANDS_NAME": s.MAX_HANDS_NAME,
        "MAX_TARGET": s.MAX_TARGET, "MAX_ARGV": s.MAX_ARGV, "MAX_PLACE": s.MAX_PLACE}
    m = re.search(r"const TIMEOUT_RANGE = \[(\d+), (\d+)\]", text)
    assert m and (int(m.group(1)), int(m.group(2))) == s.TIMEOUT_RANGE
