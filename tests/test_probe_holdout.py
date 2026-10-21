"""v0.10 Task 3: the authoring hold-out check. No training row (the real round-2 files, or build() at seeds 0-2 under
the v0.9 probes) may overlap a new probe; a hit is fixed by rewriting the probe, never the row."""
import json

from evaluation.persona_eval import load_probes
from riya.train import holdout

TEXT = "what did i tell you about my landlord last week?"


def _row(user: str, rid: str) -> dict:
    return {"id": rid, "messages": [{"role": "system", "content": "You are Riya."}, {"role": "user", "content": user}]}


def _probes(path, texts: list[str]) -> None:
    convs = [{"id": f"g_x{i}", "facts": [], "turns": [{"id": "q", "kind": "unknown", "text": t}]}
             for i, t in enumerate(texts)]
    path.write_text(json.dumps({"conversations": convs}), encoding="utf-8")


def _jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_hits_and_short_probes():
    from evaluation.probe_holdout import hits, short_probes
    assert hits([_row(TEXT, "r1"), _row("hi there", "r2")], [TEXT], []) == [("r1", TEXT)]
    fact = "The user's cat is called Pebble"
    sys_row = {"id": "r3", "messages": [{"role": "system", "content": "Facts: the user's cat is called pebble."}]}
    assert hits([sys_row], [], [fact]) == [("r3", fact)]
    probes = {"conversations": [
        {"id": "g_x", "facts": [], "turns": [{"id": "q", "kind": "unknown", "text": "what's my cat's name, riya?"}]},
        {"id": "g_y", "facts": [], "turns": [{"id": "q", "kind": "unknown", "text": TEXT}]},
        {"id": "sister", "facts": [], "turns": [{"id": "q", "kind": "unknown", "text": "my sister?"}]},
    ]}
    assert short_probes(probes) == ["g_x"]  # 5 normalised words; v0.9's own short turns are not g_*


def test_the_real_probe_file_has_no_short_probes():
    from evaluation.probe_holdout import short_probes
    assert short_probes(load_probes()) == []


def test_main_reports_each_source(tmp_path, capsys):
    from evaluation.probe_holdout import main
    current, old = tmp_path / "probes.json", tmp_path / "old.json"
    _probes(current, [TEXT])
    _probes(old, ["what is the capital of a country i made up?"])
    clean, dirty = tmp_path / "clean.jsonl", tmp_path / "dirty.jsonl"
    _jsonl(clean, [_row("hey riya", "c1")])
    _jsonl(dirty, [_row("hey riya", "d1"), _row("so " + TEXT, "d2")])
    seen = []

    def build(seed):
        seen.append((seed, holdout.PROBES_PATH))
        if seed == 1 and holdout.PROBES_PATH == current:
            raise ValueError("probe overlap: ['x']")
        return [_row("good morning riya", f"s{seed}")]

    with holdout.probes_at(current):
        code = main(["--old-probes", str(old), "--files", str(clean), str(dirty), "--seeds", "0", "1"], build=build)
    out = capsys.readouterr().out
    assert code == 1
    assert f"HIT {dirty} d2 {TEXT}" in out
    assert any(line.startswith("HIT build:seed=1 ") and "probe overlap: ['x']" in line for line in out.splitlines())
    assert len([line for line in out.splitlines() if line.startswith("HIT")]) == 2
    assert out.strip().splitlines()[-1] == "VERDICT: 2 hits"
    assert seen == [(0, old), (1, old), (0, current), (1, current)]  # v0.9's draws first, then the new probes'

    with holdout.probes_at(current):
        code = main(["--old-probes", str(old), "--files", str(clean), str(tmp_path / "gone.jsonl"), "--seeds", "0"],
                    build=lambda seed: [])
    out = capsys.readouterr().out
    assert code == 2 and f"MISSING {tmp_path / 'gone.jsonl'}" in out and "VERDICT: 0 hits" not in out

    with holdout.probes_at(current):
        code = main(["--old-probes", str(old), "--files", str(clean), "--seeds", "0", "1", "2"],
                    build=lambda seed: [_row("good morning riya", f"s{seed}")])
    out = capsys.readouterr().out
    assert code == 0 and out.strip().splitlines()[-1] == "VERDICT: 0 hits" and "HIT" not in out


def test_main_counts_a_short_probe_as_a_hit(tmp_path, capsys):
    from evaluation.probe_holdout import main
    current, old = tmp_path / "probes.json", tmp_path / "old.json"
    _probes(current, ["what's my cat's name, riya?"])
    _probes(old, [])
    clean = tmp_path / "clean.jsonl"
    _jsonl(clean, [_row("hey riya", "c1")])
    with holdout.probes_at(current):
        code = main(["--old-probes", str(old), "--files", str(clean), "--seeds", "0"], build=lambda seed: [])
    out = capsys.readouterr().out
    assert code == 1 and "HIT short g_x0" in out and out.strip().splitlines()[-1] == "VERDICT: 1 hits"
