"""evaluation/verdict.py: the checksummed verdict Colab writes. Pure files in tmp_path: no server, no model."""
import hashlib
import json

import pytest

from evaluation import champion, gate, verdict
from evaluation.verdict import VerdictError
from tests.gate_helpers import plan_transcript


def make(**over) -> dict:
    t = plan_transcript({"generic": [[1, 1, 1]], "unknown": [[1, 0, 1]]})
    kw = dict(mode="candidate", identity=gate.identity(),
              adapter={"name": "a.gguf", "size": 3, "sha256": "a" * 64},
              base_gguf={"name": "b.gguf", "size": 4, "sha256": "b" * 64}, llama_server="b1234", template_check="ok",
              base_transcript=t, cand_transcript=t[:3],
              decision={"adopt": False, "weak": ({"key": "invented"},), "reasons": ["Δ ≈ 0 — ok"]},
              timings={"base_s": 1.0, "candidate_s": 2.0, "total_s": 3.5}, failed={"base": 0, "candidate": 1})
    return verdict.make_verdict(**(kw | over))


def test_constants():
    assert (verdict.KIND, verdict.MODES) == ("riya-gate-verdict", ("candidate", "base-vs-base"))
    assert verdict.REQUIRED == ("kind", "mode", "gate_version", "probe_sha256", "prompt_fp", "scorer_sha256", "repeat",
                                "alpha", "B", "seed", "adapter", "base_gguf", "llama_server", "template_check",
                                "transcripts", "decision", "timings", "failed_turns", "created_at", "checksum")
    assert issubclass(VerdictError, ValueError)


def test_checksum_round_trip(tmp_path):
    v = make()
    assert v["checksum"] == verdict.checksum_of(v) and set(verdict.REQUIRED) <= set(v)
    ident = gate.identity()
    assert {k: v[k] for k in gate.IDENTITY_FIELDS} == ident
    assert (v["alpha"], v["B"], v["seed"]) == (gate.ALPHA, gate.B, gate.GATE_SEED)
    assert v["failed_turns"] == {"base": 0, "candidate": 1} and v["kind"] == verdict.KIND
    assert set(v["transcripts"]) == {"base", "candidate"} and len(v["transcripts"]["candidate"]) == 3
    assert all("pass" in r for r in v["transcripts"]["base"])
    path = tmp_path / "out" / "verdict.json"  # the parent folder is made
    verdict.write_verdict(v, path)
    assert verdict.load_verdict(path) == v  # tuples became lists before the checksum, so the round trip is equal
    assert verdict.canonical({"b": 1, "a": [2, "é"]}) == verdict.canonical({"a": [2, "é"], "b": 1}) \
        == '{"a":[2,"é"],"b":1}'.encode("utf-8")
    assert verdict.checksum_of(v | {"checksum": "x"}) == v["checksum"]  # every key but the checksum


def write_text(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _cut(v):
    return json.dumps(v, indent=2, ensure_ascii=False)[:len(json.dumps(v, indent=2, ensure_ascii=False)) // 2]


def _edited_reply(v):
    v["transcripts"]["base"][0]["reply"] = "an edited reply"
    return json.dumps(v)


MALFORMED = {
    "cut at half": _cut,
    "empty": lambda v: "",
    "a list": lambda v: "[]",
    "wrong kind": lambda v: '{"kind": "x"}',
    "another kind": lambda v: json.dumps(v | {"kind": "riya-gate-calibration"}),
    "unknown mode": lambda v: json.dumps(v | {"mode": "other"}),
    "transcripts a string": lambda v: json.dumps(v | {"transcripts": "x"}),
    "a transcript a string": lambda v: json.dumps(v | {"transcripts": {"base": "x", "candidate": []}}),
    "a transcript missing": lambda v: json.dumps(v | {"transcripts": {"base": []}}),
    "one key removed": lambda v: json.dumps({k: x for k, x in v.items() if k != "decision"}),
    "the checksum removed": lambda v: json.dumps({k: x for k, x in v.items() if k != "checksum"}),
    "an edited reply": _edited_reply,
    "an edited field": lambda v: json.dumps(v | {"repeat": 4}),
    "a checksum that isn't text": lambda v: json.dumps(v | {"checksum": 7}),
    "not utf-8": lambda v: None,
}


@pytest.mark.parametrize("name", MALFORMED)
def test_load_verdict_refuses_every_malformed_file(tmp_path, name):
    path = tmp_path / "v.json"
    text = MALFORMED[name](make())
    if text is None:
        path.write_bytes(b"\xff\xfe{ not utf-8")
    else:
        write_text(path, text)
    with pytest.raises(VerdictError) as e:
        verdict.load_verdict(path)
    assert str(e.value) and "\n" not in str(e.value)  # a one-line reason


def test_load_verdict_refuses_a_missing_file(tmp_path):
    with pytest.raises(VerdictError, match="read"):
        verdict.load_verdict(tmp_path / "nope.json")


def test_a_bank_verdict_has_no_adapter(tmp_path):
    v = make(mode="base-vs-base", adapter=None)
    verdict.write_verdict(v, tmp_path / "v.json")
    assert verdict.load_verdict(tmp_path / "v.json")["adapter"] is None


def test_file_identity(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"x" * ((1 << 20) + 5)  # more than one 1 MiB block
    p.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    assert verdict.sha256_file(p) == want
    assert verdict.file_identity(p) == {"name": "blob.bin", "size": len(data), "sha256": want}
    assert champion._sha256 is verdict.sha256_file
