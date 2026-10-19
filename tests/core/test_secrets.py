import io
import json
import logging
import os
from pathlib import Path

import pytest

from riya.agents import install_hooks
from riya.core import secrets as keystore
from riya.core.models import ONLINE_TIMEOUT_S, VISION_TIMEOUT_S, online_pair
from riya.core.secrets import KEY_VAR, MASK, REDACTED, InvalidKey

CANARY = "AIzaCANARY-0123456789abcdefWXYZ"


def test_save_load_clear_round_trip(tmp_path):
    p = tmp_path / "Riya" / "secrets.env"
    assert keystore.save(CANARY, p) == CANARY
    assert p.read_text() == "RIYA_ONLINE_API_KEY=" + CANARY + "\n"
    assert keystore.load(p) == CANARY
    assert keystore.clear(p) is True
    assert keystore.clear(p) is False
    assert keystore.load(p) is None
    keystore.save(CANARY, p)
    assert [f.name for f in p.parent.iterdir()] == ["secrets.env"]  # no .tmp left behind


def test_a_failed_write_keeps_the_old_key(tmp_path, monkeypatch):
    p = tmp_path / "secrets.env"
    keystore.save("OLDKEY-11111111", p)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        keystore.save(CANARY, p)
    assert keystore.load(p) == "OLDKEY-11111111"
    assert [f.name for f in tmp_path.iterdir()] == ["secrets.env"]


def test_validate():
    for raw in ("", "   "):
        with pytest.raises(InvalidKey) as e:
            keystore.validate(raw)
        assert str(e.value) == "The key is empty."
    with pytest.raises(InvalidKey) as e:
        keystore.validate("a" * 201)
    assert str(e.value) == "The key is longer than 200 characters."
    assert keystore.validate("a" * 200) == "a" * 200
    for raw in ("ab cd", "ab\tcd", "ab\x00cd", "ab​cd"):
        with pytest.raises(InvalidKey) as e:
            keystore.validate(raw)
        assert str(e.value) == "The key can't contain spaces or control characters."
        assert raw not in str(e.value)
    assert keystore.validate("  AIzaKEY12345678  ") == "AIzaKEY12345678"
    for raw in ("a" * 201, "ab cd"):
        with pytest.raises(InvalidKey) as e:
            keystore.validate(raw)
        assert raw not in str(e.value)


def test_mask():
    assert keystore.mask("AIzaSyD-zzzz1a2b") == "••••1a2b"
    assert keystore.mask("short") == "••••"
    assert keystore.mask(None) is None
    assert MASK == "••••"


def test_key_precedence_and_clear(tmp_path):
    p = tmp_path / "secrets.env"
    k = keystore.OnlineKey.at_start({KEY_VAR: "ENVKEY-11111111"}, True, p)
    keystore.save(CANARY, p)
    assert k.current() == (CANARY, "app")

    p2 = tmp_path / "other" / "secrets.env"
    k = keystore.OnlineKey.at_start({KEY_VAR: "DOTENV-22222222"}, False, p2)
    assert k.current() == ("DOTENV-22222222", ".env")
    assert k.info().clearable is False
    assert k.set(CANARY) == CANARY
    assert k.info() == keystore.KeyInfo(keystore.mask(CANARY), "app")
    assert k.info().clearable is True
    assert k.clear() is True
    assert k.current() == ("DOTENV-22222222", ".env")
    assert k.clear() is False

    none = keystore.OnlineKey.at_start({}, False, tmp_path / "third" / "secrets.env")
    assert none.current() == (None, None)
    assert none.info() == keystore.KeyInfo(None, None)
    assert KEY_VAR not in none.env({"X": "1"})
    assert none.env({"X": "1"}) == {"X": "1"}
    none.set(CANARY)
    out = none.env({KEY_VAR: "OLD-33333333", "X": "1"})
    assert out == {KEY_VAR: CANARY, "X": "1"}
    env = {KEY_VAR: "OLD-33333333"}
    none.env(env)
    assert env == {KEY_VAR: "OLD-33333333"}  # a copy


def test_load_ignores_a_bad_file(tmp_path, caplog):
    p = tmp_path / "secrets.env"
    p.write_text("RIYA_ONLINE_API_KEY=has space inside\n")
    with caplog.at_level(logging.DEBUG):
        assert keystore.load(p) is None
    assert "has space" not in caplog.text
    assert keystore.load(tmp_path / "missing.env") is None


def test_load_takes_the_last_line_and_a_bom(tmp_path):
    p = tmp_path / "secrets.env"
    p.write_bytes("﻿OTHER=1\nRIYA_ONLINE_API_KEY=first-key-1234\nRIYA_ONLINE_API_KEY=last-key-5678\n".encode())
    assert keystore.load(p) == "last-key-5678"


def test_redaction_scrubs_every_logger():
    root = logging.getLogger()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    solo_buf = io.StringIO()
    solo = logging.getLogger("riya.test_solo")
    solo_handler = logging.StreamHandler(solo_buf)
    solo.addHandler(solo_handler)
    solo.propagate = False
    old_level = root.level
    root.setLevel(logging.DEBUG)
    solo.setLevel(logging.DEBUG)
    try:
        keystore.remember(CANARY)
        keystore.install_redaction()
        logging.getLogger("riya.models").warning("bad %s", CANARY)
        try:
            raise ValueError(f"HTTP 400 {CANARY}")
        except ValueError:
            logging.getLogger("riya.models").exception("request failed")
        solo.warning("solo %s", CANARY)
        logging.getLogger("riya.models").warning("stack", stack_info=True, extra={})
        keystore.remember("k")
        logging.getLogger("riya.models").warning("keep k")
    finally:
        keystore.uninstall_redaction()
        root.removeHandler(handler)
        solo.removeHandler(solo_handler)
        solo.propagate = True
        root.setLevel(old_level)
    out, solo_out = buf.getvalue(), solo_buf.getvalue()
    assert CANARY not in out and CANARY not in solo_out
    assert out.count(REDACTED) >= 2 and REDACTED in solo_out
    assert "HTTP 400 " + REDACTED in out
    assert "keep k" in out
    assert keystore.scrub("keep k") == "keep k"


def test_install_redaction_is_idempotent_and_uninstall_restores():
    before = logging.getLogRecordFactory()
    f1 = keystore.install_redaction()
    f2 = keystore.install_redaction()
    try:
        assert f1 is f2
        assert logging.getLogger().filters.count(f1) == 1
    finally:
        keystore.uninstall_redaction()
    assert logging.getLogRecordFactory() is before
    assert f1 not in logging.getLogger().filters


def test_redact_filter_always_passes_and_scrubs_exc_info():
    keystore.remember(CANARY)
    flt = keystore.RedactFilter()
    try:
        raise ValueError(f"boom {CANARY}")
    except ValueError:
        import sys
        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "msg %s", (CANARY,), sys.exc_info())
    assert flt.filter(rec) is True
    assert CANARY not in rec.msg and rec.args is None
    assert CANARY not in rec.exc_text and REDACTED in rec.exc_text


def test_online_pair(monkeypatch):
    assert online_pair({}) == (None, None)
    online, vision = online_pair({KEY_VAR: "ENVKEY-11111111"})
    assert online is not None and vision is not None and online is not vision
    assert online.first_token_timeout_s == 8 and vision.first_token_timeout_s == 20
    assert online.name == "online" and vision.name == "vision"
    assert ONLINE_TIMEOUT_S == 8.0 and VISION_TIMEOUT_S == 20.0
    calls = []

    def make(env, name, timeout_s):
        calls.append((env, name, timeout_s))
        return None
    env = {KEY_VAR: "ENVKEY-11111111"}
    assert online_pair(env, make) == (None, None)
    assert calls == [(env, "online", 8.0), (env, "vision", 20.0)]


def test_the_runner_still_exports_the_vision_timeout():
    from riya.core.__main__ import VISION_TIMEOUT_S as from_runner
    assert from_runner == 20.0


def test_default_paths_are_guarded(tmp_path):
    assert tmp_path in keystore.default_path().parents
    assert tmp_path in install_hooks.default_settings_path().parents
    assert tmp_path in install_hooks.default_token_path().parents
    assert os.environ["APPDATA"].startswith(str(tmp_path))


def test_remembered_keys_are_reset_between_tests():
    keystore.remember(CANARY)
    keystore.forget_all()
    assert keystore.scrub(CANARY) == CANARY


def test_gitignore_lists_secrets_env():
    lines = (Path(__file__).parents[2] / ".gitignore").read_text().splitlines()
    assert "secrets.env" in lines


def test_redaction_scrubs_extra_fields():
    root = logging.getLogger()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s | %(x)s"))
    root.addHandler(handler)
    try:
        keystore.remember(CANARY)
        keystore.install_redaction()
        logging.getLogger("riya.models").warning("hello", extra={"x": CANARY})
    finally:
        keystore.uninstall_redaction()
        root.removeHandler(handler)
    assert CANARY not in buf.getvalue() and "hello | " + REDACTED in buf.getvalue()


def test_scrub_covers_repr_and_json_escaped_forms():
    tricky = "abcd\\efgh\"ijkl\\mn"
    keystore.remember(tricky)
    for shown in (tricky, repr(tricky), repr([tricky]), json.dumps(tricky), json.dumps({"k": tricky})):
        out = keystore.scrub(shown)
        assert "efgh" not in out and "ijkl" not in out, shown
        assert REDACTED in out


def test_load_strips_matching_outer_quotes(tmp_path):
    p = tmp_path / "secrets.env"
    for line, want in (('RIYA_ONLINE_API_KEY="quoted-key-1234"', "quoted-key-1234"),
                       ("RIYA_ONLINE_API_KEY='single-key-5678'", "single-key-5678"),
                       ('RIYA_ONLINE_API_KEY="unbalanced-1234', '"unbalanced-1234')):  # only a matching pair goes
        p.write_text(line + "\n")
        assert keystore.load(p) == want


def test_clear_on_a_folder_raises_a_clean_oserror(tmp_path):
    folder = tmp_path / "secrets.env"
    folder.mkdir()
    with pytest.raises(keystore.KeyFileError) as e:
        keystore.clear(folder)
    assert isinstance(e.value, OSError) and str(e.value) == "The key file path is a folder."
    assert str(folder) not in str(e.value)


def test_mask_has_its_own_threshold():
    assert keystore.MASK_MIN == 8
    assert keystore.mask("1234567") == MASK and keystore.mask("12345678") == MASK + "5678"


def test_env_vars_are_guarded_too(tmp_path):
    assert os.environ["LOCALAPPDATA"].startswith(str(tmp_path))
    assert os.environ["USERPROFILE"].startswith(str(tmp_path))
    assert tmp_path in Path.home().parents
