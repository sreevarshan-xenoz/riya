"""evaluation/verdict.py: the checksummed verdict Colab writes. Pure files in tmp_path: no server, no model."""
import hashlib
import json
import os

import pytest

from evaluation import champion, gate, verdict
from evaluation.verdict import VerdictError
from tests.gate_helpers import plan_transcript


def make(**over) -> dict:
    t = plan_transcript({"generic": [[1] * gate.REPEAT], "unknown": [[1, 0] + [1] * (gate.REPEAT - 2)]})
    kw = dict(mode="candidate", identity=gate.identity(),
              adapter={"name": "a.gguf", "size": 3, "sha256": "a" * 64},
              base_gguf={"name": "b.gguf", "size": 4, "sha256": "b" * 64}, llama_server="b1234", template_check="ok",
              base_transcript=t, cand_transcript=t,
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
    assert set(v["transcripts"]) == {"base", "candidate"} and len(v["transcripts"]["candidate"]) == 2 * gate.REPEAT
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


def signed(**over) -> str:
    """A verdict with `over` applied and its checksum recomputed: only a structural check can refuse it."""
    body = make() | over
    body["checksum"] = verdict.checksum_of(body)
    return json.dumps(body)


def _cut(v):
    text = json.dumps(v, indent=2, ensure_ascii=False)
    return text[:len(text) // 2]


def _without(key):
    def drop(v):
        body = {k: x for k, x in v.items() if k != key}
        if key != "checksum":
            body["checksum"] = verdict.checksum_of(body)
        return json.dumps(body)
    return drop


def _edited_reply(v):
    v["transcripts"]["base"][0]["reply"] = "an edited reply"
    return json.dumps(v)


# name -> (the file's text from a good verdict, a fragment of the reason): the structural rows are re-checksummed, so
# each is refused by its own check and by no other
MALFORMED = {
    "cut at half": (_cut, "valid JSON"),
    "empty": (lambda v: "", "valid JSON"),
    "a list": (lambda v: "[]", "isn't a verdict"),
    "wrong kind": (lambda v: '{"kind": "x"}', "isn't a gate verdict"),
    "another kind, re-checksummed": (lambda v: signed(kind="riya-gate-calibration"), "isn't a gate verdict"),
    "unknown mode, re-checksummed": (lambda v: signed(mode="other"), "unknown mode"),
    "transcripts a string, re-checksummed": (lambda v: signed(transcripts="x"), "transcripts aren't lists"),
    "a transcript a string, re-checksummed": (lambda v: signed(transcripts={"base": "x", "candidate": []}),
                                              "transcripts aren't lists"),
    "a transcript missing, re-checksummed": (lambda v: signed(transcripts={"base": []}), "transcripts aren't lists"),
    "one key removed, re-checksummed": (_without("decision"), "missing decision"),
    "the checksum removed": (_without("checksum"), "missing checksum"),
    "a candidate without an adapter, re-checksummed": (lambda v: signed(adapter=None), "adapter has no sha256"),
    "an adapter without a sha256, re-checksummed": (lambda v: signed(adapter={"name": "a"}), "adapter has no sha256"),
    "an adapter sha256 that isn't hex, re-checksummed": (lambda v: signed(adapter={"name": "a", "sha256": "xyz"}),
                                                         "adapter has no sha256"),
    "a base without a sha256, re-checksummed": (lambda v: signed(base_gguf={"name": "b"}), "base_gguf has no sha256"),
    "an edited reply": (_edited_reply, "checksum"),
    "an edited field": (lambda v: json.dumps(v | {"repeat": gate.REPEAT + 1}), "checksum"),
    "a checksum that isn't text": (lambda v: json.dumps(v | {"checksum": 7}), "checksum"),
    "not utf-8": (lambda v: None, "valid JSON"),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_load_verdict_refuses_every_malformed_file(tmp_path, name):
    path = tmp_path / "v.json"
    build, reason = MALFORMED[name]
    text = build(make())
    if text is None:
        path.write_bytes(b"\xff\xfe{ not utf-8")
    else:
        write_text(path, text)
    with pytest.raises(VerdictError, match=reason) as e:
        verdict.load_verdict(path)
    assert "\n" not in str(e.value)  # a one-line reason


def test_a_bank_verdict_may_have_no_adapter(tmp_path):
    write_text(tmp_path / "v.json", signed(mode="base-vs-base", adapter=None))
    assert verdict.load_verdict(tmp_path / "v.json")["adapter"] is None
    write_text(tmp_path / "w.json", signed(mode="base-vs-base", adapter=None, base_gguf=None))
    with pytest.raises(VerdictError, match="base_gguf has no sha256"):  # every verdict names its base
        verdict.load_verdict(tmp_path / "w.json")


def _bumped(x):
    if isinstance(x, bool):
        return not x
    if isinstance(x, (int, float)):
        return x + 1
    if isinstance(x, str):
        return x + "x"
    if isinstance(x, dict):
        return x | {"zz": 1}
    return [*x, "zz"]


# every field the checksum must cover, edited after it was written (the structural fields refuse earlier, so are
# covered above)
COVERED = [k for k in verdict.REQUIRED if k not in ("checksum", "kind", "mode", "transcripts")]
NESTED = [("adapter", "sha256"), ("base_gguf", "sha256"), ("adapter", "size"), ("decision", "adopt"),
          ("decision", "reasons"), ("timings", "total_s"), ("failed_turns", "candidate")]


@pytest.mark.parametrize("key", COVERED)
def test_the_checksum_covers_every_field(tmp_path, key):
    v = make()
    v[key] = _bumped(v[key])
    write_text(tmp_path / "v.json", json.dumps(v))
    with pytest.raises(VerdictError, match="checksum"):
        verdict.load_verdict(tmp_path / "v.json")


@pytest.mark.parametrize("key,inner", NESTED)
def test_the_checksum_covers_the_nested_fields(tmp_path, key, inner):
    v = make()
    v[key][inner] = "0" * 64 if inner == "sha256" else _bumped(v[key].get(inner, 1))
    write_text(tmp_path / "v.json", json.dumps(v))
    with pytest.raises(VerdictError, match="checksum"):
        verdict.load_verdict(tmp_path / "v.json")


def test_the_covered_fields_are_all_the_non_structural_ones():
    assert set(COVERED) == {"gate_version", "probe_sha256", "prompt_fp", "scorer_sha256", "repeat", "alpha", "B",
                            "seed", "adapter", "base_gguf", "llama_server", "template_check", "decision", "timings",
                            "failed_turns", "created_at"}


def test_write_verdict_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "drive" / "verdict-abc.json"
    good = make()
    verdict.write_verdict(good, path)
    before = path.read_bytes()

    def dies(src, dst):
        raise OSError("the Drive went away")

    monkeypatch.setattr(os, "replace", dies)  # a failure after the temp file is written, before it is put in place
    with pytest.raises(OSError):
        verdict.write_verdict(make(llama_server="b9999"), path)
    assert path.read_bytes() == before and verdict.load_verdict(path) == good  # the earlier verdict is untouched
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # and no temp file is left behind

    fresh = tmp_path / "drive" / "verdict-new.json"
    with pytest.raises(OSError):
        verdict.write_verdict(good, fresh)
    assert not fresh.exists() and [p.name for p in path.parent.iterdir()] == [path.name]


def test_load_verdict_refuses_a_missing_file(tmp_path):
    with pytest.raises(VerdictError, match="read"):
        verdict.load_verdict(tmp_path / "nope.json")


def test_file_identity(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"x" * ((1 << 20) + 5)  # more than one 1 MiB block
    p.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    assert verdict.sha256_file(p) == want
    assert verdict.file_identity(p) == {"name": "blob.bin", "size": len(data), "sha256": want}
    assert champion._sha256 is verdict.sha256_file


# ---- final fix wave M4: the Python and numpy versions, for information only ----

def test_a_verdict_names_its_python_and_numpy_for_information_only(tmp_path):
    import platform

    import numpy as np
    v = make()
    assert v["env"] == verdict.env_info() == {"python": platform.python_version(), "numpy": np.__version__}
    assert "env" not in verdict.REQUIRED and "env" not in gate.IDENTITY_FIELDS  # never identity, never required
    other = v | {"env": {"python": "3.0.1", "numpy": "0.0.1"}}
    other["checksum"] = verdict.checksum_of(other)  # inside the checksum like everything else
    older = {k: x for k, x in v.items() if k != "env"}  # a verdict written before the field existed
    older["checksum"] = verdict.checksum_of(older)
    for n, body in enumerate((other, older)):
        verdict.write_verdict(body, tmp_path / f"{n}.json")
        assert verdict.load_verdict(tmp_path / f"{n}.json") == body  # loads: the versions are never checked
    assert verdict.env_note(v["env"], "The verdict") is None and verdict.env_note(None, "The verdict") is None
    note = verdict.env_note(other["env"], "The verdict")
    assert note.startswith(f"Note: The verdict ran on Python 3.0.1 with numpy 0.0.1; this machine has Python "
                           f"{platform.python_version()} with numpy {np.__version__}.")
    assert verdict.env_note({"numpy": np.__version__}, "x").startswith("Note: x ran on Python ? with numpy")
