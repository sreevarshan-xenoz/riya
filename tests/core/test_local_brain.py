"""The Health "Local brain" row and the Use base model / Use Riya v1 switch (Review Focus 5).

No llama-server, no network: /v1/models and /lora-adapters are answered by a fake http_get, the adapter is a tmp file
and its champion record is written with brain.write_record."""
import json
import os
from datetime import datetime

import pytest

from riya import brain
from riya.core import health
from riya.core.health import BROKEN, DETAIL_MAX, STILL_V1, check_local
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
    assert (item.status, item.detail, item.fix) == ("ok", "LFM2.5 base · x.gguf", "use_riya_v1")
    assert row(tmp_path, http_get=http([], models=b'{"data":[]}'), adapter_path=path).detail == "LFM2.5 base"

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
    item = row(tmp_path, settings=on, http_get=http([str(path)]))
    assert item.detail == f"Riya v1 (adapter {date}, no champion record)" and item.fix == "use_base"
    brain.write_record(path, {**ADOPT, "targets_candidate": 1})
    assert row(tmp_path, settings=on, http_get=http([str(path)])).detail.endswith("champion -1 targets)")
    path.unlink()  # served by a llama-server that hasn't noticed yet
    item = row(tmp_path, settings=on, http_get=http([str(path)]))
    assert (item.status, item.detail) == ("ok", "Riya v1 (adapter file missing)")
    # the setting is compared as a path: another spelling of the same file counts
    path = adapter(tmp_path, ADOPT)
    other = str(path).replace("\\", "/").upper() if os.name == "nt" else str(path)
    item = row(tmp_path, settings=health.Settings(local_adapter=other), http_get=http([str(path)]))
    assert item.status == "ok" and "champion +2" in item.detail
    # a bad answer from /lora-adapters is "nothing served", never an error
    item = row(tmp_path, http_get=lambda u, t: MODELS if u.endswith("/v1/models") else b"not json")
    assert (item.status, item.detail) == ("ok", "LFM2.5 base · x.gguf")


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
    assert local.fix == "use_riya_v1" and local.detail == "LFM2.5 base · x.gguf"

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


@pytest.mark.parametrize("fix", ["use_base", "use_riya_v1"])
def test_the_fixes_are_listed(fix):
    assert fix in health.FIXES
