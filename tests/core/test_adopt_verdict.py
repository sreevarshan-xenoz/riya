"""evaluation/champion.py's laptop side (v0.10, A10): --check-verdict and --adopt-verdict re-decide a Colab verdict
locally, check her own llama-server on the CPU (-ngl 0, an empty CUDA_VISIBLE_DEVICES) and adopt atomically. Fakes
only: no llama-server, no GPU, no real model, models folder or DB. The `calibrated` fixture's tiny base file is the
model every verdict here names, and tests/conftest.py points the adopted path into tmp_path."""
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import champion, gate, persona_eval, verdict
from evaluation.champion import ChampionError
from riya import brain
from riya.core import embed
from riya.core.settings import Settings, save_settings
from riya.core.store import KV, open_db
from riya.train.fmt import render_lfm2
from tests import gate_helpers
from tests.gate_helpers import calibrated, write_calibration  # noqa: F401
from tests.real_paths import tmp_only

CAND_BYTES = b"GGUF the candidate Colab measured"
URL = "http://127.0.0.1:8091"  # CPU_PORT


def phrases() -> list[str]:
    return persona_eval.style_phrases(persona_eval.load_probes())  # what the laptop re-scores with


def pair_transcripts() -> tuple[list[dict], list[dict]]:
    base, cand = gate_helpers.adopt_pair(gate.REPEAT)
    return list(base.transcript), list(cand.transcript)


def verdict_for(base_t: list[dict], cand_t: list[dict], adapter: Path, model: Path, /, **over) -> dict:
    """A checksummed candidate verdict on these transcripts, its decision from decide(), `over` applied before the
    checksum."""
    base, cand = gate.scored(base_t, phrases()), gate.scored(cand_t, phrases())
    d = gate.decide(base, cand)
    v = verdict.make_verdict(
        mode="candidate", identity=gate.identity(), adapter=verdict.file_identity(adapter),
        base_gguf=verdict.file_identity(model), llama_server="b1234", template_check="ok", base_transcript=base_t,
        cand_transcript=cand_t, decision=gate.decision_dict(d),
        timings={"base_s": 60.0, "candidate_s": 90.0, "total_s": 150.0},
        failed={"base": base.failed, "candidate": cand.failed}, embedder_id=embed.EMBEDDER_PIN)
    v.update(json.loads(json.dumps(over)))
    v["checksum"] = verdict.checksum_of(v)
    return v


def save(v: dict, path: Path) -> Path:
    path.write_text(json.dumps(v), encoding="utf-8")
    return path


class Cpu:
    """The injected CPU check: a fake llama-server recording how it was made, started and stopped, an http_get
    listing the adapter it serves, and an http_post answering /apply-template and /completion."""

    def __init__(self, *, template=None, listed=None, completion=None, post_raises=None):
        self.made, self.events, self.running = [], [], None
        self.template, self.listed, self.post_raises = template, listed, post_raises
        self.completion = {"content": "Hi there"} if completion is None else completion
        cpu = self

        class FakeCpuServer:
            def __init__(self, model, **kw):
                self.model, self.kw, self.lora = model, kw, kw.get("lora")
                cpu.made.append(self)

            def start(self, gpu=True):
                cpu.events.append(("start", gpu))
                cpu.running = self

            def stop(self):
                cpu.events.append(("stop",))
                cpu.running = None

        self.make_server = FakeCpuServer

    def http_get(self, url, timeout_s):
        assert url == f"{URL}/lora-adapters" and self.running
        if self.listed is not None:
            return self.listed.encode()
        return json.dumps([{"id": 0, "path": self.running.lora, "scale": 1.0}]).encode()

    def http_post(self, url, body, timeout):
        assert self.running
        if url == f"{URL}/apply-template":
            prompt = render_lfm2(body["messages"], add_generation_prompt=True)
            return {"prompt": prompt if self.template is None else self.template(prompt)}
        assert (url, body, timeout) == (f"{URL}/completion", champion.CPU_GEN, champion.CPU_TIMEOUT_S)
        if self.post_raises is not None:
            raise self.post_raises
        return self.completion


def no_core(url, **kw):
    raise ConnectionRefusedError("no core")


class Never:
    """A dependency the verdict modes must never touch (the GPU path: free VRAM, the gate's GPU server, a live run)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append(a)
        raise AssertionError("the verdict modes never use this")


CPU_TMP: list[Path] = []  # this test's directory for the CPU check's temp slot dir (never the real %TEMP%)


@pytest.fixture(autouse=True)
def cpu_tmp(tmp_path) -> Path:
    d = tmp_path / "cpu-tmp"
    d.mkdir()
    CPU_TMP[:] = [d]
    return d


def go(cpu: Cpu, argv, *, fullscreen=lambda: False, port_busy=lambda port: False, connect=no_core, never=None,
       **kw) -> int:
    never = never or Never()
    return champion.main(argv, fullscreen=fullscreen, free_vram=never, make_server=never, run=never,
                         http_get=cpu.http_get, http_post=cpu.http_post, connect=connect, port_busy=port_busy,
                         make_cpu_server=cpu.make_server, cpu_tmp_dir=CPU_TMP[0], make_embedder=never, **kw)


@pytest.fixture
def world(tmp_path, calibrated):
    """An adopted adapter with its record, a candidate next to it, a settings DB naming the adopted file and a passing
    verdict for the candidate on the calibrated base model."""
    adopted = tmp_only(tmp_path, brain.default_adapter())  # C1: fake adapters only ever go under tmp_path
    adopted.parent.mkdir(parents=True, exist_ok=True)
    adopted.write_bytes(b"GGUF the adapter adopted earlier")
    brain.write_record(adopted, {"decision": "adopt", "targets_base": 8, "targets_candidate": 9})
    candidate = tmp_only(tmp_path, brain.candidate_adapter())
    candidate.write_bytes(CAND_BYTES)
    db = tmp_path / "settings" / "riya.db"
    conn = open_db(db)
    save_settings(KV(conn), Settings(local_adapter=str(adopted)))
    conn.close()
    base_t, cand_t = pair_transcripts()
    v = save(verdict_for(base_t, cand_t, candidate, calibrated), tmp_path / "verdict.json")
    return {"adopted": adopted, "candidate": candidate, "db": db, "verdict": v, "model": calibrated,
            "transcripts": (base_t, cand_t)}


def argv(world, mode="--adopt-verdict", path=None, **extra) -> list[str]:
    out = [mode, str(path or world["verdict"]), "--model", str(world["model"]), "--db", str(world["db"])]
    for k, val in extra.items():
        out += [f"--{k}", str(val)]
    return out


def tree(root: Path) -> dict:
    """Every file under root with its bytes and mtime: what "nothing written" means."""
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


# ---- an adopt that passes ------------------------------------------------------------------------------------------

def test_a_passing_verdict_adopts_and_records(world, capsys, monkeypatch):
    adopted, cpu = world["adopted"], Cpu()
    v = verdict.load_verdict(world["verdict"])
    writes, real_write = [], brain.write_record
    monkeypatch.setattr(brain, "write_record", lambda *a, **kw: writes.append(kw) or real_write(*a, **kw))
    assert go(cpu, argv(world)) == 0
    assert [set(kw) for kw in writes] == [{"stat"}]  # keyed to the stat taken before the re-hash (M1)
    assert adopted.read_bytes() == CAND_BYTES and not world["candidate"].exists()
    rec = brain.read_record(adopted)
    assert rec is not None and rec["decision"] == "adopt" and rec["decided_on"] == "colab"
    assert rec["verdict_checksum"] == v["checksum"] and rec["adapter_sha256"] == v["adapter"]["sha256"]
    assert (rec["gate_version"], rec["probe_sha256"], rec["repeat"]) == (
        gate.GATE_VERSION, gate.probe_sha256(), gate.REPEAT)
    assert (rec["targets_base"], rec["targets_candidate"]) == (8, 9)
    assert rec["weak"] == {"generic_share": "kept", "invented": "same", "unknown_honesty": "kept",
                           "about_me_recall": "improved"}
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", rec["decided_at"])
    assert brain.read_local_adapter(world["db"]) == str(adopted)
    assert brain.vetted_adapter(str(adopted)) == str(adopted)
    out = capsys.readouterr().out
    assert "DECISION: adopt Riya v1" in out
    assert re.search(r"Adopted: Riya v1 is her local brain now\. "
                     r"\(saved; it applies on Riya's next start; took \d+ s\)", out)
    assert [e[0] for e in cpu.events] == ["start", "stop"]
    adopted.write_bytes(b"GGUF copied over the adopted file afterwards")  # vetted-only serving is unchanged
    assert brain.vetted_adapter(str(adopted)) is None


# ---- refusals: each one changes nothing on disk --------------------------------------------------------------------

def other_model(world) -> dict:
    other = world["model"].with_name("other-base.gguf")
    other.write_bytes(b"another base model")
    return {"model": other}


def missing_model(world) -> dict:
    return {"model": world["model"].with_name("no-such-base.gguf")}


def changed_candidate(world) -> dict:
    world["candidate"].write_bytes(b"GGUF retrained after Colab measured it")
    return {}


def resave(world, **over) -> dict:
    base_t, cand_t = world["transcripts"]
    save(verdict_for(base_t, cand_t, world["candidate"], world["model"], **over), world["verdict"])
    return {}


def keep_forced_adopt(world) -> dict:
    t = world["transcripts"][1]
    decision = verdict_for(t, t, world["candidate"], world["model"])["decision"]
    save(verdict_for(t, t, world["candidate"], world["model"], decision={**decision, "adopt": True}), world["verdict"])
    return {}


def bad_checksum(world) -> dict:
    v = json.loads(world["verdict"].read_text(encoding="utf-8"))
    v["checksum"] = "0" * 64
    save(v, world["verdict"])
    return {}


def no_calibration(world) -> dict:
    gate.CALIBRATION_PATH.unlink()
    return {}


REFUSALS = {
    "adapter sha": (changed_candidate, {}, "This adapter isn't the one the verdict measured."),
    "base sha": (other_model, {}, "The base model isn't the one the verdict measured."),
    "probe_sha256": (lambda w: resave(w, probe_sha256="f" * 64), {}, "The verdict is for probe_sha256 ffffffffffff"),
    "gate_version": (lambda w: resave(w, gate_version="0.9.0"), {}, "The verdict is for gate_version 0.9.0, this "
                                                                     f"code has {gate.GATE_VERSION}."),
    "repeat": (lambda w: resave(w, repeat=gate.REPEAT + 1), {},
               f"The verdict is for repeat {gate.REPEAT + 1}, this code has {gate.REPEAT}."),
    "prompt_fp": (lambda w: resave(w, prompt_fp="x" * 64), {}, "The verdict is for prompt_fp xxxxxxxxxxxx"),
    "scorer_sha256": (lambda w: resave(w, scorer_sha256="0" * 64), {}, "The verdict is for scorer_sha256 "
                                                                       "000000000000"),
    "no calibration": (no_calibration, {}, "There's no gate calibration record."),
    "failing calibration": (lambda w: write_calibration(gate.CALIBRATION_PATH, passed=False) and {}, {},
                            "The gate calibration didn't pass."),
    "calibrated on another base": (lambda w: write_calibration(gate.CALIBRATION_PATH, base_sha256="0" * 64) and {},
                                   {}, "The gate calibration is for base model 000000000000"),
    "missing model": (missing_model, {}, "Couldn't read "),
    "base-vs-base": (lambda w: resave(w, mode="base-vs-base", adapter=None), {},
                     "That's a base-vs-base verdict: it can't adopt anything."),
    "bad checksum": (bad_checksum, {}, "checksum doesn't match"),
    "re-decides keep": (keep_forced_adopt, {}, "The verdict says adopt, but this code decides keep"),
    "template differs": (lambda w: {}, {"template": lambda p: p.replace("<|im_end|>\n", "<|im_end|>")},
                         "chat template differs"),
    "adapter not listed": (lambda w: {}, {"listed": "[]"}, "the adapter didn't load"),
    "no content": (lambda w: {}, {"completion": {"tokens": []}}, "didn't generate on the CPU"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_adopt_refusals_change_nothing_on_disk(world, tmp_path, capsys, case):
    setup, cpu_kw, reason = REFUSALS[case]
    extra = setup(world) or {}
    adopted, candidate = world["adopted"], world["candidate"]
    before = tree(tmp_path)
    record_before = brain.record_path(adopted).read_text(encoding="utf-8")
    cpu = Cpu(**cpu_kw)
    assert go(cpu, argv(world, **extra)) == 2
    out = capsys.readouterr().out
    assert reason in out and out.rstrip().endswith("Nothing changed.") and "Traceback" not in out
    assert tree(tmp_path) == before  # not a byte, not an mtime
    assert brain.record_path(adopted).read_text(encoding="utf-8") == record_before
    assert brain.read_record(adopted) is not None and candidate.exists()
    assert brain.read_local_adapter(world["db"]) == str(adopted)
    assert cpu.running is None and cpu.events.count(("stop",)) == len(cpu.made)


def with_passes(rows: list[dict], keep: int) -> list[dict]:
    """The rows' passes 1..keep: fewer than REPEAT drops the last ones; more adds copies of the last pass."""
    last = [r for r in rows if r["pass"] == gate.REPEAT]
    out = [r for r in rows if r["pass"] <= keep]
    for n in range(gate.REPEAT + 1, keep + 1):
        out += [r | {"pass": n, "probe_id": gate.probe_key(r["probe_id"]) + f"#{n}"} for r in last]
    return out


@pytest.mark.parametrize("sides, held, said", [
    (("base",), gate.REPEAT - 1, "base transcript has {held} passes"),
    (("candidate",), gate.REPEAT - 1, "candidate transcript has {held} passes"),
    (("base", "candidate"), gate.REPEAT - 1, "base transcript has {held} passes"),  # both short, so they still agree
    (("base",), gate.REPEAT + 1, "base transcript has {held} passes"),
    (("candidate",), gate.REPEAT + 1, "candidate transcript has {held} passes"),
    (("base", "candidate"), gate.REPEAT + 1, "base transcript has {held} passes"),
])
@pytest.mark.parametrize("mode", ["--check-verdict", "--adopt-verdict"])
def test_a_verdict_must_hold_exactly_its_repeat_in_passes(world, tmp_path, capsys, mode, sides, held, said):
    """I3: a checksum-valid verdict labelled repeat R but holding another number of passes would be re-decided at that
    other k under a calibration for R. Both modes refuse it, with nothing written and no server started."""
    base_t, cand_t = world["transcripts"]
    changed = {"base": with_passes(base_t, held) if "base" in sides else base_t,
               "candidate": with_passes(cand_t, held) if "candidate" in sides else cand_t}
    v = verdict_for(base_t, cand_t, world["candidate"], world["model"], transcripts=changed)
    assert v["repeat"] == gate.REPEAT  # the label is right; the transcripts are not
    save(v, world["verdict"])
    before, cpu = tree(tmp_path), Cpu()
    assert go(cpu, argv(world, mode)) == 2
    out = capsys.readouterr().out
    assert said.format(held=held) in out and f"not the {gate.REPEAT} it claims." in out and "Traceback" not in out
    assert cpu.made == [] and tree(tmp_path) == before


def test_a_verdict_whose_decision_disagrees_is_refused_both_ways(world, capsys):
    base_t, cand_t = world["transcripts"]
    decision = json.loads(world["verdict"].read_text(encoding="utf-8"))["decision"]
    assert decision["adopt"] is True
    resave(world, decision={**decision, "adopt": False})  # stored keep, re-decided adopt
    for mode in ("--adopt-verdict", "--check-verdict"):
        cpu = Cpu()
        assert go(cpu, argv(world, mode)) == 2
        out = capsys.readouterr().out
        assert "The verdict says keep, but this code decides adopt: not adopting on a disagreement." in out
        assert cpu.made == []
    keep_forced_adopt(world)  # stored adopt, re-decided keep
    cpu = Cpu()
    assert go(cpu, argv(world)) == 2
    out = capsys.readouterr().out
    assert "The verdict says adopt, but this code decides keep: not adopting on a disagreement." in out
    ours = verdict.env_info()  # M4: both sides' versions are named, since numpy's stream can differ between them
    assert (f"The verdict ran on Python {ours['python']} with numpy {ours['numpy']}; this machine has Python "
            f"{ours['python']} with numpy {ours['numpy']}. Nothing changed.") in out
    assert cpu.made == [] and brain.read_local_adapter(world["db"]) == str(world["adopted"])


def test_a_stored_keep_that_re_decides_keep_is_not_adopted(world, capsys):
    t = world["transcripts"][1]
    save(verdict_for(t, t, world["candidate"], world["model"]), world["verdict"])
    cpu = Cpu()
    assert go(cpu, argv(world, "--check-verdict")) == 0  # a consistent verdict: the check passes
    assert "DECISION: keep the base model" in capsys.readouterr().out
    assert go(cpu, argv(world)) == 2
    assert "The verdict's decision is to keep the base model. Nothing changed." in capsys.readouterr().out
    assert cpu.made == [] and world["candidate"].exists()


# ---- the CPU check of her own llama-server -------------------------------------------------------------------------

def cpu_check(cpu: Cpu, lora="C:/m/cand.gguf", *, fullscreen=lambda: False, port_busy=lambda port: False):
    return champion.cpu_check("C:/m/base.gguf", lora, make_server=cpu.make_server, http_get=cpu.http_get,
                              http_post=cpu.http_post, fullscreen=fullscreen, port_busy=port_busy,
                              tmp_dir=CPU_TMP[0])


def test_cpu_env_hides_the_gpu():
    assert champion.cpu_env({"PATH": "x", "CUDA_VISIBLE_DEVICES": "0"}) == {"PATH": "x", "CUDA_VISIBLE_DEVICES": ""}
    assert champion.cpu_env()["CUDA_VISIBLE_DEVICES"] == ""
    assert (champion.CPU_PORT, champion.CPU_CTX, champion.CPU_THREADS, champion.CPU_TIMEOUT_S) == (8091, 2048, 4, 60.0)
    # M2: the slowest refusal (a load that never gets ready, then the template, the adapter list and the generation at
    # their timeouts) stays inside the adopt step's 2 minutes
    assert champion.CPU_READY_S + 10.0 + 1.5 + champion.CPU_TIMEOUT_S <= 120.0
    assert champion.CPU_GEN == {"prompt": "Hi", "n_predict": 4, "temperature": 0}


def test_cpu_check_runs_on_cpu_and_always_stops(monkeypatch, cpu_tmp):
    cpu = Cpu()
    assert cpu_check(cpu) is None
    (srv,) = cpu.made
    assert srv.model == "C:/m/base.gguf" and srv.lora == "C:/m/cand.gguf"
    kw = srv.kw
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and (kw["ctx"], kw["threads"], kw["port"]) == (2048, 4, 8091)
    assert kw["vetted_only"] is False and kw["fallback_to_base"] is False
    assert kw["ready_timeout_s"] == champion.CPU_READY_S  # M2
    assert Path(kw["slot_dir"]).parent == cpu_tmp and list(cpu_tmp.iterdir()) == []  # M4: made there, then removed
    assert cpu.events == [("start", False), ("stop",)]

    for bad in ({"template": lambda p: p + "x"}, {"listed": "[]"}, {"listed": "not json"},
                {"completion": {"content": 4}}, {"completion": ["Hi"]}, {"post_raises": TimeoutError("timed out")}):
        cpu = Cpu(**bad)
        with pytest.raises(ChampionError):
            cpu_check(cpu)
        assert cpu.events == [("start", False), ("stop",)], bad

    cpu = Cpu(post_raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        cpu_check(cpu)
    assert cpu.events == [("start", False), ("stop",)] and cpu.running is None

    cpu = Cpu()

    def never_ready(self, gpu=True):
        cpu.events.append(("start", gpu))
        raise RuntimeError("llama-server did not become ready")

    monkeypatch.setattr(cpu.make_server, "start", never_ready)
    with pytest.raises(RuntimeError, match="did not become ready"):
        cpu_check(cpu)
    assert cpu.events == [("start", False), ("stop",)]

    cpu = Cpu()

    def no_binary(self, gpu=True):  # M3: LLAMA_SERVER points nowhere, so popen raises
        cpu.events.append(("start", gpu))
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(cpu.make_server, "start", no_binary)
    with pytest.raises(ChampionError, match=r"Couldn't start her llama-server \(The system cannot find the file"):
        cpu_check(cpu)
    assert cpu.events == [("start", False), ("stop",)]

    cpu = Cpu()
    with pytest.raises(ChampionError, match="while you're gaming"):
        cpu_check(cpu, fullscreen=lambda: True)
    assert cpu.made == []
    with pytest.raises(ChampionError, match="port 8091"):
        cpu_check(cpu, port_busy=lambda port: port == 8091)
    with pytest.raises(ChampionError, match="port 8091"):  # both at once: the port first, as in v0.8's _check_ready
        cpu_check(cpu, fullscreen=lambda: True, port_busy=lambda port: True)
    assert cpu.made == []


def test_the_real_wrapper_gets_a_cpu_only_command_and_environment(tmp_path):
    """The default make_server (RiyaServer) with a fake popen: the keywords fit its signature, and llama-server would
    get -ngl 0, the small context, 4 threads, port 8091, the candidate as --lora and no visible GPU."""
    from riya.server import RiyaServer

    lora = tmp_path / "cand.gguf"
    lora.write_bytes(CAND_BYTES)  # unvetted on purpose: the CPU check serves it as it is
    launched, cpu = [], Cpu()

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            launched.append("terminate")

        def wait(self, timeout=None):
            return 0

    def popen(cmd, **kw):
        launched.append((cmd, kw))
        return Proc()

    def make_server(model, ready_timeout_s, **kw):
        assert ready_timeout_s == champion.CPU_READY_S
        cpu.running = RiyaServer(model, popen=popen, healthy=lambda port: True, ready_timeout_s=0, **kw)
        return cpu.running

    champion.cpu_check("C:/m/base.gguf", str(lora), make_server=make_server, http_get=cpu.http_get,
                       http_post=cpu.http_post, fullscreen=lambda: False, port_busy=lambda port: False,
                       tmp_dir=CPU_TMP[0])
    (cmd, kw), stopped = launched
    flag = {f: cmd[cmd.index(f) + 1] for f in ("-ngl", "-c", "-t", "--port", "--lora")}
    assert flag == {"-ngl": "0", "-c": "2048", "-t": "4", "--port": "8091", "--lora": str(lora)}
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "" and stopped == "terminate"


def test_the_adopt_refuses_while_gaming_before_any_server(world, tmp_path, capsys):
    before, cpu = tree(tmp_path), Cpu()
    assert go(cpu, argv(world), fullscreen=lambda: True) == 2
    out = capsys.readouterr().out
    assert "while you're gaming" in out and out.rstrip().endswith("Nothing changed.")
    assert cpu.made == [] and tree(tmp_path) == before


def test_the_adopt_never_touches_the_gpu_path(world, capsys):
    never = Never()
    assert go(Cpu(), argv(world), never=never) == 0
    assert never.calls == []  # no free-VRAM read, no GPU server, no live run


# ---- --check-verdict: read-only -------------------------------------------------------------------------------------

def test_check_verdict_writes_nothing(world, tmp_path, capsys):
    before, cpu, never, looked = tree(tmp_path), Cpu(), Never(), []
    assert go(cpu, argv(world, "--check-verdict"), never=never, fullscreen=lambda: looked.append(1)) == 0
    out = capsys.readouterr().out
    assert "Targets met: base 8, Riya v1 9" in out and "DECISION: adopt Riya v1" in out
    assert out.rstrip().endswith("Check only: nothing written.")
    assert cpu.made == [] and never.calls == [] and looked == []
    assert tree(tmp_path) == before
    assert brain.read_local_adapter(world["db"]) == str(world["adopted"])


def test_check_verdict_refuses_with_the_reason(world, capsys):
    changed_candidate(world)
    assert go(Cpu(), argv(world, "--check-verdict")) == 2
    assert "This adapter isn't the one the verdict measured. Nothing changed." in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["--check-verdict", "--adopt-verdict"])
def test_check_verdict_refuses_another_embedder_id(world, tmp_path, capsys, mode):
    """§4.7.5: a verdict whose runs used another embedder (the hash fallback, an unverified bge) is refused by name,
    with nothing on disk changed and nothing started."""
    resave(world, embedder_id="x")
    before, cpu, never = tree(tmp_path), Cpu(), Never()
    assert go(cpu, argv(world, mode), never=never) == 2
    out = capsys.readouterr().out
    assert ("The verdict's runs used embedder x, this code pins " + embed.EMBEDDER_PIN[:12] + ". Nothing changed."
            in out) and "Traceback" not in out
    assert tree(tmp_path) == before and cpu.made == [] and never.calls == []
    assert brain.read_local_adapter(world["db"]) == str(world["adopted"])


# ---- after the install: the installed bytes must be the measured ones ------------------------------------------------

def test_a_rehash_mismatch_after_install_writes_no_record(world, capsys, monkeypatch):
    def install(candidate, adopted):
        adopted.write_bytes(b"GGUF not the bytes the verdict measured")

    monkeypatch.setattr(champion, "install", install)
    assert go(Cpu(), argv(world)) == 2
    out = capsys.readouterr().out
    assert f"The candidate is installed as {world['adopted']}, but its champion record couldn't be written" in out
    assert "base model" in out and "Nothing changed." not in out
    assert brain.read_record(world["adopted"]) is None
    assert brain.vetted_adapter(str(world["adopted"])) is None  # she serves the base model
    assert brain.read_local_adapter(world["db"]) == str(world["adopted"])


def test_an_adopted_file_that_changes_while_it_is_hashed_gets_no_record(world, capsys, monkeypatch):
    """M1: the record is keyed to the stat taken before the re-hash, and a file that changed by the stat after it is
    refused: the record never names a size and mtime whose bytes weren't hashed."""
    real_identity = champion.file_identity

    def identity_then_replaced(path):
        got = real_identity(path)
        if Path(path) == world["adopted"]:  # the post-install re-hash: the right bytes, then replaced at once
            world["adopted"].write_bytes(b"GGUF a second adopt landing in between")
        return got

    monkeypatch.setattr(champion, "file_identity", identity_then_replaced)
    assert go(Cpu(), argv(world)) == 2
    out = capsys.readouterr().out
    assert "couldn't be written (it changed while it was hashed)" in out and "Nothing changed." not in out
    assert brain.read_record(world["adopted"]) is None
    assert brain.read_local_adapter(world["db"]) == str(world["adopted"])


def test_write_record_keys_to_a_given_stat(tmp_path):
    f = tmp_path / "a.gguf"
    f.write_bytes(b"GGUF")
    rec = json.loads(brain.write_record(f, {"decision": "adopt"}, stat=SimpleNamespace(st_size=7, st_mtime_ns=9))
                     .read_text(encoding="utf-8"))
    assert (rec["size"], rec["mtime_ns"]) == (7, 9) and brain.read_record(f) is None  # not this file as it is now
    brain.write_record(f, {"decision": "adopt"})  # v0.8's callers: the file as it is now
    st = os.stat(f)
    assert (brain.read_record(f)["size"], brain.read_record(f)["mtime_ns"]) == (st.st_size, st.st_mtime_ns)


def test_a_missing_llama_server_is_refused_not_a_traceback(world, capsys, monkeypatch):
    """M3: LLAMA_SERVER points nowhere."""
    cpu = Cpu()

    def no_binary(self, gpu=True):
        cpu.events.append(("start", gpu))
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(cpu.make_server, "start", no_binary)
    record_before = brain.record_path(world["adopted"]).read_text(encoding="utf-8")
    assert go(cpu, argv(world)) == 2
    out = capsys.readouterr().out
    assert "Couldn't start her llama-server" in out and out.rstrip().endswith("Nothing changed.")
    assert "Traceback" not in out and cpu.events == [("start", False), ("stop",)]
    assert brain.record_path(world["adopted"]).read_text(encoding="utf-8") == record_before
    assert world["candidate"].exists() and brain.read_local_adapter(world["db"]) == str(world["adopted"])


def test_a_failed_install_writes_nothing(world, capsys, monkeypatch):
    def locked(candidate, adopted):
        raise ChampionError(f"Couldn't install {candidate} as {adopted} (the file is in use).")

    monkeypatch.setattr(champion, "install", locked)
    record_before = brain.record_path(world["adopted"]).read_text(encoding="utf-8")
    assert go(Cpu(), argv(world)) == 2
    assert "Couldn't install" in capsys.readouterr().out
    assert brain.record_path(world["adopted"]).read_text(encoding="utf-8") == record_before
    assert world["candidate"].exists() and brain.read_local_adapter(world["db"]) == str(world["adopted"])


# ---- final fix wave M4: another Python or numpy is a note, never a refusal ----

def test_another_python_or_numpy_is_a_note_never_a_refusal(world, tmp_path, capsys):
    base_t, cand_t = world["transcripts"]
    v = save(verdict_for(base_t, cand_t, world["candidate"], world["model"],
                         env={"python": "3.0.1", "numpy": "0.0.1"}), tmp_path / "other-env.json")
    assert go(Cpu(), argv(world, "--check-verdict", v)) == 0
    out = capsys.readouterr().out
    assert "Note: The verdict ran on Python 3.0.1 with numpy 0.0.1; this machine has Python " in out
    assert "DECISION: adopt Riya v1" in out and "Check only: nothing written." in out
    assert go(Cpu(), argv(world, "--check-verdict")) == 0  # the same versions: no note
    assert "Note:" not in capsys.readouterr().out
