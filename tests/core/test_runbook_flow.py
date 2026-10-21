"""v0.10 final fix I1: the runbook's chain end to end, offline, each step reading what the step before it really wrote.

  section 8: three `champion --base-vs-base --verdict-out` runs (Colab) -> `gate_calibrate` (laptop, CPU)
  section 7: a candidate `champion --verdict-out` run (Colab)
  section 9: `--check-verdict` (writes nothing) -> `--adopt-verdict` (a fake CPU check, the install into tmp)

Fakes only: the gate's servers and persona_eval runs are test_champion's Rig, the CPU check is test_adopt_verdict's Cpu,
llama-server's version is a lambda. Everything lands under tmp_path (tests/conftest.py and `calibrated` redirect the
models folder, the base model, the calibration record and the settings DB). The champion record and the local_adapter
setting are written only by the last step. The Colab steps run with COLAB_RELEASE_TAG set, so this also pins the
gate's own servers getting `--cache-ram 0` there and the laptop's CPU check never getting it."""
import json
from pathlib import Path

from evaluation import champion, gate, gate_calibrate, persona_eval, verdict
from riya import brain
from tests.core.test_adopt_verdict import Cpu
from tests.core.test_champion import Rig, no_core
from tests.gate_helpers import PINNED, calibrated  # noqa: F401
from tests.real_paths import tmp_only

CAND_BYTES = b"GGUF the candidate Colab measured"
UNKNOWN = sorted(c["id"] for c in persona_eval.load_probes()["conversations"] if c["id"].startswith("g_unknown"))


def transcript(reply, latency: float) -> list[dict]:
    """A run_probes-shaped transcript of the real probe file, three passes, `reply(conv, turn)` for every turn."""
    rows = []
    for n in range(1, gate.REPEAT + 1):
        for conv in persona_eval.load_probes()["conversations"]:
            known = list(conv["facts"])
            for turn in conv["turns"]:
                if not turn.get("speak_first"):
                    known.append(turn["text"])
                rows.append({"conversation": conv["id"],
                             "probe_id": f"{conv['id']}.{turn['id']}" + ("" if n == 1 else f"#{n}"),
                             "kind": turn["kind"], "text": turn["text"], "reply": reply(conv, turn),
                             "first_sentence_s": latency, "known_texts": list(known), "facts": list(conv["facts"]),
                             "error": None, "pass": n})
    return rows


def base_reply(conv, turn) -> str:
    """The base model: recalls every fact, honest about being an AI, and guesses on half the unknown probes."""
    if turn["kind"] == "about_me" and conv["facts"]:
        return "Easy~ " + " ".join(conv["facts"])
    if turn["kind"] == "unknown" and conv["id"] in UNKNOWN[:15]:
        return "It's Marigold, obviously~"
    return {"unknown": "Hmph, I don't know that one~", "ai": "I'm an AI, silly~",
            "ai_offpath": "I'm an AI, silly~"}.get(turn["kind"], "Hmph, fine~")


def cand_reply(conv, turn) -> str:
    """Riya v1: the same, but honest on every unknown probe."""
    return "I don't know, tell me?" if turn["kind"] == "unknown" else base_reply(conv, turn)


def tree(root: Path) -> dict:
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_runbook_chain_runs_end_to_end(tmp_path, calibrated, capsys, monkeypatch):
    db = tmp_path / "settings" / "riya.db"
    bank = tmp_path / "gate" / "bank"
    adopted = tmp_only(tmp_path, brain.default_adapter())
    record = brain.record_path(adopted)
    failed = tmp_path / "gate" / "calibration-failed.json"
    monkeypatch.setattr(gate_calibrate, "FAILED_PATH", failed)
    monkeypatch.setattr(gate_calibrate, "DRAWS", 20)  # the pre-registered count, smaller: the chain, not the statistics
    gate.CALIBRATION_PATH.unlink()  # the record the gate reads must be the one gate_calibrate writes below
    colab = dict(fullscreen=lambda: False, free_vram=lambda: 6000, connect=no_core, port_busy=lambda p: False,
                 version=lambda: "version: 11780 (cuda)", make_embedder=lambda choice: PINNED)

    def nothing_adopted() -> bool:
        return not adopted.exists() and not record.exists() and not db.exists()

    # section 8 on Colab: the bank, three live base-vs-base runs (distinct by their latencies)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-20261021")
    for n in (1, 2, 3):
        t = transcript(base_reply, 0.5 + n / 100)
        rig = Rig(transcripts=[t, t])
        assert champion.main(["--base-vs-base", "--verdict-out", str(bank / f"bank-{n}.json"),
                              "--model", str(calibrated), "--db", str(db)], make_server=rig.make_server, run=rig.run,
                             http_get=rig.http_get, http_post=rig.http_post, **colab) == 0
        assert rig.extras == [["--cache-ram", "0"]] * 2  # Colab: the gate's own servers get no host prompt cache
    assert sorted(p.name for p in bank.iterdir()) == ["bank-1.json", "bank-2.json", "bank-3.json"]
    v1 = verdict.load_verdict(bank / "bank-1.json")
    assert v1["mode"] == "base-vs-base" and v1["env"] == verdict.env_info() and v1["llama_server"].startswith("version")
    assert not gate.CALIBRATION_PATH.exists() and nothing_adopted()

    # section 8 on the laptop: calibrate from those three files, on its default path (written because it passed)
    monkeypatch.delenv("COLAB_RELEASE_TAG")
    capsys.readouterr()
    assert gate_calibrate.main(["--bank", str(bank)]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "CALIBRATION: PASS" in out and f"Written: {gate.CALIBRATION_PATH}" in out
    rec = json.loads(gate.CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert rec["passed"] is True and rec["base_sha256"] == verdict.file_identity(calibrated)["sha256"]
    assert [b["file"] for b in rec["bank"]] == ["bank-1.json", "bank-2.json", "bank-3.json"]
    assert rec["env"] == verdict.env_info() and not failed.exists() and nothing_adopted()

    # section 7 on Colab: the candidate's verdict, under that record
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-20261021")
    cand = tmp_only(tmp_path, brain.candidate_adapter())
    cand.parent.mkdir(parents=True, exist_ok=True)
    cand.write_bytes(CAND_BYTES)
    v = tmp_path / "gate" / "verdict-cand.json"
    rig = Rig(transcripts=[transcript(base_reply, 0.5), transcript(cand_reply, 0.5)])
    assert champion.main(["--lora", str(cand), "--verdict-out", str(v), "--model", str(calibrated), "--db", str(db)],
                         make_server=rig.make_server, run=rig.run, http_get=rig.http_get, http_post=rig.http_post,
                         **colab) == 0
    out = capsys.readouterr().out
    assert "DECISION: adopt Riya v1" in out and "Dry run: nothing written." in out and "Verdict written:" in out
    assert rig.extras == [["--cache-ram", "0"]] * 2
    assert verdict.load_verdict(v)["mode"] == "candidate" and nothing_adopted() and cand.read_bytes() == CAND_BYTES

    # section 9 on the laptop: read the verdict (nothing written, nothing started), then adopt from it
    monkeypatch.delenv("COLAB_RELEASE_TAG")
    cpu = Cpu()
    laptop = dict(fullscreen=lambda: False, connect=no_core, port_busy=lambda p: False,
                  make_cpu_server=cpu.make_server, http_get=cpu.http_get, http_post=cpu.http_post,
                  cpu_tmp_dir=tmp_path)
    before = tree(tmp_path)
    assert champion.main(["--check-verdict", str(v), "--model", str(calibrated), "--db", str(db)], **laptop) == 0
    out = capsys.readouterr().out
    assert "DECISION: adopt Riya v1" in out and "Check only: nothing written." in out
    assert tree(tmp_path) == before and cpu.made == [] and nothing_adopted()

    assert champion.main(["--adopt-verdict", str(v), "--model", str(calibrated), "--db", str(db)], **laptop) == 0, \
        capsys.readouterr().out
    assert "Adopted: Riya v1 is her local brain now." in capsys.readouterr().out
    assert [e[0] for e in cpu.events] == ["start", "stop"] and cpu.events[0] == ("start", False)
    made = cpu.made[0].kw
    assert made["env"]["CUDA_VISIBLE_DEVICES"] == "" and not made.get("extra")  # the laptop: no Colab flags
    assert adopted.read_bytes() == CAND_BYTES and not cand.exists()
    assert brain.vetted_adapter(str(adopted)) == str(adopted)
    rec = brain.read_record(adopted)
    assert rec["decided_on"] == "colab" and rec["verdict_checksum"] == verdict.load_verdict(v)["checksum"]
    assert brain.read_local_adapter(db) == str(adopted)
    for p in (adopted, record, db, gate.CALIBRATION_PATH, v):
        tmp_only(tmp_path, p)
