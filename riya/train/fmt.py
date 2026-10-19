"""The row formats of Riya v1's fine-tuning data: the requests the dataset builder writes (and the teacher answers), and
JSON-lines reading and writing. Pure and CPU-only (no torch).
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1 and §4."""
import json
import re
from collections.abc import Iterable
from pathlib import Path

REQUEST_KEYS = ("id", "kind", "messages", "facts", "expect", "fp")  # fp: build_sft.prompt_fingerprint() at build time
_HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECT_KEYS = ("must_mention", "must_not_invent", "must_admit_unknown", "must_say_ai", "max_sentences", "max_emoji")


def read_jsonl(path: Path) -> list[dict]:
    """One JSON value per line (utf-8); blank lines are skipped."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    """One JSON object per line (utf-8, non-ASCII kept, "\\n" endings), making the parent folder first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _int(v, lo: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def validate_request(row: dict) -> None:
    """ValueError unless row is a request: exactly REQUEST_KEYS; messages a system message, then user/assistant pairs,
    ending with a user message, all with str content; facts a list of str; expect exactly EXPECT_KEYS (list of str,
    bool, bool, bool, int >= 1, int >= 0); fp a sha256 hex digest (64 lowercase hex characters)."""
    if not isinstance(row, dict) or set(row) != set(REQUEST_KEYS) or len(row) != len(REQUEST_KEYS):
        raise ValueError(f"a request has exactly the keys {REQUEST_KEYS}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"{row['id']}: messages must be a system message, user/assistant pairs and a user message")
    for i, m in enumerate(messages):
        want = "system" if i == 0 else "user" if i % 2 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want or not isinstance(m.get("content"), str):
            raise ValueError(f"{row['id']}: message {i} must be a {want} message with str content")
    if not isinstance(row["facts"], list) or not all(isinstance(f, str) for f in row["facts"]):
        raise ValueError(f"{row['id']}: facts must be a list of str")
    e = row["expect"]
    if not isinstance(e, dict) or set(e) != set(EXPECT_KEYS) or len(e) != len(EXPECT_KEYS):
        raise ValueError(f"{row['id']}: expect has exactly the keys {EXPECT_KEYS}")
    if not isinstance(e["must_mention"], list) or not all(isinstance(f, str) for f in e["must_mention"]):
        raise ValueError(f"{row['id']}: must_mention must be a list of str")
    if not all(isinstance(e[k], bool) for k in ("must_not_invent", "must_admit_unknown", "must_say_ai")):
        raise ValueError(f"{row['id']}: must_not_invent, must_admit_unknown and must_say_ai must be bool")
    if not _int(e["max_sentences"], 1) or not _int(e["max_emoji"], 0):
        raise ValueError(f"{row['id']}: max_sentences must be an int >= 1 and max_emoji an int >= 0")
    if not isinstance(row["fp"], str) or not _HEX64.fullmatch(row["fp"]):
        raise ValueError(f"{row['id']}: fp must be a sha256 hex digest")
