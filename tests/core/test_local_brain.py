"""The Health "Local brain" row and the Use base model / Use Riya v1 switch (Review Focus 5).

No llama-server, no network: /v1/models and /lora-adapters are answered by a fake http_get, the adapter is a tmp file
and its champion record is written with brain.write_record."""
import json
import os
from datetime import datetime

import pytest

from riya import brain
from riya.core import health
from riya.core.health import BROKEN, DETAIL_MAX, IN_USE_UNVETTED, STILL_V1, UNVETTED, check_local
from riya.core.protocol import HealthGet, HealthMsg, SettingsMsg, SetupRequest, SetupResult
from riya.core.setup import (NOT_VETTED, PLAIN, SUMMARIES, USING_BASE, USING_V1, Setup, SetupDeps)
from tests.core.test_health import ctx as health_ctx
from tests.core.test_setup import FakeHost, make_deps
from tests.fakes import make_core, wait_for

URL = "http://127.0.0.1:8090"
MODELS = json.dumps({"data": [{"id": "C:/models/x.gguf"}]}).encode()


def http(adapters=(), models=MODELS, calls=None):
    """Answers /v1/models and /lora-adapters by URL; `adapters` is a list of served paths (scale 1)."""
    def get(url, timeout):
        if calls is not None:
            calls.append((url, timeout))
        if url.endswith("/v1/models"):
            return models
        if url.endswith("/lora-adapters"):
            return json.dumps([{"id": i, "path": p, "scale": 1.0} for i, p in enumerate(adapters)]).encode()
        raise AssertionError(url)
    return get


def adapter(tmp_path, record=None):
    path = tmp_path / "riya-v1-lora.gguf"
    path.write_bytes(b"GGUF-lora")
    if record is not None:
        brain.write_record(path, record)
    return path


ADOPT = {"decision": "adopt", "targets_base": 2, "targets_candidate": 4}


def row(tmp_path, **over):
    return check_local(health_ctx(tmp_path, local_url=URL, **over))


def test_local_row_adapter_states(tmp_path):
    path = adapter(tmp_path, ADOPT)
    date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    on = health.Settings(local_adapter=str(path))
    calls = []

    item = row(tmp_path, settings=on, http_get=http([str(path)], calls=calls), adapter_path=path)
    assert (item.status, item.detail, item.fix) == ("ok", f"Riya v1 (adapter {date}, champion +2 targets)", "use_base")
    assert calls[0][0] == URL + "/v1/models" and calls[1] == (URL + "/lora-adapters", 1.5)

    for gone in (False, True):  # wanted but llama-server isn't serving it (still loading, or the file is deleted)
        if gone:
            path.unlink()
        item = row(tmp_path, settings=on, http_get=http([]), adapter_path=path)
        assert (item.status, item.detail, item.fix) == ("warn", BROKEN, "use_base")
    assert "still loading" in BROKEN and len(BROKEN) <= DETAIL_MAX

    path = adapter(tmp_path, ADOPT)
    item = row(tmp_path, http_get=http([str(path)]), adapter_path=path)  # served, not wanted
    assert (item.status, item.detail, item.fix) == ("warn", STILL_V1, None) and len(STILL_V1) <= DETAIL_MAX

    item = row(tmp_path, http_get=http([]), adapter_path=path)  # base, with an adopted adapter on disk
    assert (item.status, item.detail, item.fix) == ("ok", "Base model · x.gguf", "use_riya_v1")
    assert row(tmp_path, http_get=http([], models=b'{"data":[]}'), adapter_path=path).detail == "Base model"

    brain.write_record(path, {**ADOPT, "decision": "reject"})
    assert row(tmp_path, http_get=http([]), adapter_path=path).fix is None
    brain.record_path(path).unlink()
    assert row(tmp_path, http_get=http([]), adapter_path=path).fix is None  # no record
    brain.write_record(path, ADOPT)
    path.write_bytes(b"GGUF-retrained-and-longer")  # the record belongs to the old file
    assert row(tmp_path, http_get=http([]), adapter_path=path).fix is None
    path.unlink()
    assert row(tmp_path, http_get=http([]), adapter_path=path).fix is None
    assert row(tmp_path, http_get=http([])).fix is None  # no adapter_path at all


def test_local_row_details(tmp_path):
    path = adapter(tmp_path)  # no record
    on = health.Settings(local_adapter=str(path))
    date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    item = row(tmp_path, settings=on, http_get=http([str(path)]))  # served, but nothing vetted it (I2)
    assert (item.status, item.detail, item.fix) == ("warn", IN_USE_UNVETTED, "use_base")
    assert IN_USE_UNVETTED == "A Riya v1 adapter is in use without a passing champion check."
    brain.write_record(path, {**ADOPT, "targets_candidate": 1})
    item = row(tmp_path, settings=on, http_get=http([str(path)]))
    assert item.status == "ok" and item.detail == f"Riya v1 (adapter {date}, champion -1 targets)"
    brain.write_record(path, {**ADOPT, "decision": "reject"})  # a rejected file someone served anyway
    assert row(tmp_path, settings=on, http_get=http([str(path)])).detail == IN_USE_UNVETTED
    brain.write_record(path, ADOPT)
    path.write_bytes(b"GGUF replaced after adoption")  # served from the old launch; the record is stale now
    item = row(tmp_path, settings=on, http_get=http([str(path)]))
    assert (item.status, item.detail, item.fix) == ("warn", IN_USE_UNVETTED, "use_base")
    path.unlink()  # served by a llama-server that hasn't noticed yet: nothing vouches for it
    item = row(tmp_path, settings=on, http_get=http([str(path)]))
    assert (item.status, item.detail) == ("warn", IN_USE_UNVETTED)
    # the setting is compared as a path: another spelling of the same file counts
    path = adapter(tmp_path, ADOPT)
    other = str(path).replace("\\", "/").upper() if os.name == "nt" else str(path)
    item = row(tmp_path, settings=health.Settings(local_adapter=other), http_get=http([str(path)]))
    assert item.status == "ok" and "champion +2" in item.detail
    # a bad answer from /lora-adapters is "nothing served", never an error
    item = row(tmp_path, http_get=lambda u, t: MODELS if u.endswith("/v1/models") else b"not json")
    assert (item.status, item.detail) == ("ok", "Base model · x.gguf")


def test_unreachable_rows_are_unchanged(tmp_path):
    def boom(url, timeout):
        raise OSError("refused")
    item = row(tmp_path, http_get=boom)
    assert item.status == "warn" and URL in item.detail and item.fix is None
    assert check_local(health_ctx(tmp_path, local_url="")).detail == "No local server is configured."


def make_setup(tmp_path):
    host = FakeHost()
    deps = make_deps(tmp_path)
    deps.adapter_path = tmp_path / "riya-v1-lora.gguf"
    results = []
    return Setup(host, deps, results.append, spawn=lambda fn: fn()), host, deps, results


def test_use_base_and_use_riya_v1(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    assert "use_base" in PLAIN and "use_riya_v1" in PLAIN
    assert SUMMARIES["use_base"] == "Switched to the base model" and SUMMARIES["use_riya_v1"] == "Switched to Riya v1"

    assert s.request("use_riya_v1", {}) is None  # a plain click: no confirm
    assert not results[-1].ok and results[-1].detail.startswith("There's no Riya v1 adapter")
    assert host.patches == []

    path = adapter(tmp_path)  # a file, but nothing vetted it
    s.request("use_riya_v1", {})
    assert (results[-1].ok, results[-1].detail) == (False, NOT_VETTED) and host.patches == []

    brain.write_record(path, {**ADOPT, "decision": "reject"})
    s.request("use_riya_v1", {})
    assert results[-1].detail == NOT_VETTED and host.patches == []

    brain.write_record(path, ADOPT)
    assert s.request("use_riya_v1", {}) is None
    assert (results[-1].ok, results[-1].detail) == (True, USING_V1)
    assert host.patches == [{"local_adapter": str(path)}]

    assert s.request("use_base", {}) is None
    assert (results[-1].ok, results[-1].detail) == (True, USING_BASE)
    assert host.patches[-1] == {"local_adapter": None} and len(host.patches) == 2
    assert [r[2:] for r in host.rows if r[3] == "ran"] == [
        ("Switched to Riya v1", "ran"), ("Switched to the base model", "ran")]
    assert all(r[:2] == ("setup", 0) for r in host.rows) and [r[3] for r in host.rows] == ["failed"] * 3 + ["ran"] * 2


def test_missing_adapter_message_fits(tmp_path):
    s, _, deps, results = make_setup(tmp_path)
    deps.adapter_path = tmp_path / ("d" * 200) / "riya-v1-lora.gguf"
    s.request("use_riya_v1", {})
    assert len(results[-1].detail) <= DETAIL_MAX and results[-1].detail.endswith("riya-v1-lora.gguf")


def test_sandboxed_deps_carry_an_adapter_path(tmp_path):
    assert SetupDeps.sandboxed(tmp_path).adapter_path == tmp_path / "riya-v1-lora.gguf"
    assert SetupDeps.sandboxed(tmp_path).adapter_path != brain.default_adapter()


def test_engine_offers_the_switch(tmp_path):
    deps = SetupDeps.sandboxed(tmp_path)
    core, emitted, *_ = make_core(setup_deps=deps)
    core.local_url = URL
    core.health_http = http([])
    path = adapter(tmp_path, ADOPT)
    assert deps.adapter_path == path

    core.handle(HealthGet())
    h = wait_for(emitted, HealthMsg)
    (local,) = [i for i in h.items if i.key == "local"]
    assert local.fix == "use_riya_v1" and local.detail == "Base model · x.gguf"

    mark = len(emitted)
    core.handle(SetupRequest(action="use_riya_v1"))
    msg = wait_for(emitted, SettingsMsg, after=mark)
    assert msg.settings["local_adapter"] == str(path)
    assert core.settings.local_adapter == str(path)

    mark = len(emitted)
    core.handle(SetupRequest(action="use_base"))
    assert wait_for(emitted, SettingsMsg, after=mark).settings["local_adapter"] is None


def test_engine_withholds_the_switch_without_a_record(tmp_path):
    deps = SetupDeps.sandboxed(tmp_path)
    core, emitted, *_ = make_core(setup_deps=deps)
    core.local_url, core.health_http = URL, http([])
    adapter(tmp_path)
    core.handle(HealthGet())
    (local,) = [i for i in wait_for(emitted, HealthMsg).items if i.key == "local"]
    assert local.fix is None
    mark = len(emitted)
    core.handle(SetupRequest(action="use_riya_v1"))
    r = wait_for(emitted, SetupResult, after=mark)
    assert not r.ok and r.detail == NOT_VETTED
    assert not [m for m in emitted[mark:] if isinstance(m, SettingsMsg)]
    assert core.settings.local_adapter is None


def test_the_fixes_are_listed():
    assert health.FIXES[-2:] == ("use_base", "use_riya_v1")  # appended last, in this order (the v0.7 order holds)
    assert health.FIXES[:-2] == ("online_test", "voice_sample", "mic_check", "eyes_test", "hooks_connect",
                                 "hooks_disconnect", "autostart")
    assert len(set(health.FIXES)) == len(health.FIXES)


@pytest.mark.parametrize("action", ["use_base", "use_riya_v1"])
@pytest.mark.parametrize("args", [{"path": "C:/evil/other.gguf"}, {"local_adapter": "C:/evil/other.gguf"},
                                  {"adapter_path": "C:/evil/other.gguf"}, {"on": True}, {"local_adapter": None}])
def test_crafted_args_are_rejected(tmp_path, action, args):
    """The switch takes no arguments: a crafted one can never pick the adapter path (or anything else)."""
    from riya.core.setup import ARGS, SetupError

    assert action not in ARGS
    s, host, deps, results = make_setup(tmp_path)
    adapter(tmp_path, ADOPT)  # a vetted adapter is there, so only the args could make it fail
    with pytest.raises(SetupError, match=f"bad arguments for {action}"):
        s.request(action, args)
    assert host.patches == [] and results == []


def test_crafted_args_through_the_core_change_nothing(tmp_path):
    deps = SetupDeps.sandboxed(tmp_path)
    core, emitted, *_ = make_core(setup_deps=deps)
    adapter(tmp_path, ADOPT)
    mark = len(emitted)
    core.handle(SetupRequest(action="use_riya_v1", args={"path": "C:/evil/other.gguf"}))
    assert not [m for m in emitted[mark:] if isinstance(m, (SettingsMsg, SetupResult))]
    assert core.settings.local_adapter is None


def test_local_row_names_an_unvetted_wanted_adapter(tmp_path):
    """C1: the wrapper serves base for a wanted adapter without a matching adopt record; the row says why."""
    path = adapter(tmp_path)  # never measured
    on = health.Settings(local_adapter=str(path))
    item = row(tmp_path, settings=on, http_get=http([]), adapter_path=path)
    assert (item.status, item.detail, item.fix) == ("warn", UNVETTED, "use_base") and len(UNVETTED) <= DETAIL_MAX
    brain.write_record(path, {**ADOPT, "decision": "reject"})
    assert row(tmp_path, settings=on, http_get=http([]), adapter_path=path).detail == UNVETTED
    brain.write_record(path, ADOPT)
    path.write_bytes(b"GGUF copied over the adopted file")  # the record is stale
    assert row(tmp_path, settings=on, http_get=http([]), adapter_path=path).detail == UNVETTED
    brain.write_record(path, ADOPT)  # vetted, but not (yet) served: still loading, or it didn't load
    assert row(tmp_path, settings=on, http_get=http([]), adapter_path=path).detail == BROKEN
    path.unlink()
    assert row(tmp_path, settings=on, http_get=http([]), adapter_path=path).detail == BROKEN


def test_the_row_names_the_served_gguf_and_promises_nothing(tmp_path):
    other = json.dumps({"data": [{"id": r"C:\models\Some-Other-Model.Q8_0.gguf"}]}).encode()
    assert STILL_V1 == "llama-server is serving an adapter that Riya's settings don't name."
    assert "next check" not in STILL_V1 and "switch" not in STILL_V1
    item = row(tmp_path, http_get=http([], models=other))
    assert (item.status, item.detail) == ("ok", "Base model · Some-Other-Model.Q8_0.gguf")
    assert "LFM2.5" not in item.detail
