"""Providers join the pool from providers.env (or the environment), live, with their keys kept out of logs."""
import pytest

from riya.core import models, pool, providers, secrets
from riya.core.models import ModelUnavailable
from riya.core.pool import Member, ModelPool

MSGS = [{"role": "user", "content": "hi"}]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Fake:
    def __init__(self, name, base_url, api_key, model, first_token_timeout_s=8.0):
        self.name, self.base_url, self.api_key, self.model = name, base_url, api_key, model
        self.first_token_timeout_s, self.fail, self.calls, self.finish_reason = first_token_timeout_s, None, 0, None

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        yield self.model


def test_load_keys_reads_known_names_only(tmp_path):
    f = tmp_path / "providers.env"
    f.write_text("# my keys\nGROQ_API_KEY=gsk_aaaaaaaa\nexport CEREBRAS_API_KEY='csk-bbbbbbbb'\nUNRELATED=x\n"
                 "OPENROUTER_API_KEY=\nGROQ_MODELS=a, b\n", encoding="utf-8")
    assert providers.load_keys(f) == {"GROQ_API_KEY": "gsk_aaaaaaaa", "CEREBRAS_API_KEY": "csk-bbbbbbbb",
                                      "GROQ_MODELS": "a, b"}
    assert providers.load_keys(tmp_path / "missing.env") == {}


def test_members_follow_the_keys_in_provider_order(tmp_path):
    keys = {"OPENROUTER_API_KEY": "sk-or-cccccccc", "GROQ_API_KEY": "gsk_aaaaaaaa", "CLOUDFLARE_API_TOKEN": "cf-dddd"}
    ms = providers.members(keys, "online", 12.0, make=Fake)
    assert [m.provider for m in ms][:3] == ["groq"] * 3 and {m.provider for m in ms} == {"groq", "openrouter"}
    assert all(m.model.api_key == "gsk_aaaaaaaa" for m in ms if m.provider == "groq")
    assert all(m.shared for m in ms if m.provider == "openrouter") and not any(m.shared for m in ms[:3])
    keys["CLOUDFLARE_ACCOUNT_ID"] = "acct123"  # Cloudflare needs its account id in the URL
    cf = [m for m in providers.members(keys, "online", 12.0, make=Fake) if m.provider == "cloudflare"]
    assert cf and cf[0].model.base_url.endswith("/accounts/acct123/ai/v1")
    vision = providers.members(keys, "vision", 20.0, vision_only=True, make=Fake)
    assert vision and all(m.vision for m in vision)
    keys["GROQ_MODELS"] = "x, y"
    assert [m.name for m in providers.members(keys, "online", 12.0, make=Fake) if m.provider == "groq"] == ["x", "y"]


def test_provider_keys_are_redacted_in_logs():
    providers.members({"GROQ_API_KEY": "gsk_SECRETSECRET123"}, "online", 12.0, make=Fake)
    assert "gsk_SECRETSECRET123" not in secrets.scrub("key was gsk_SECRETSECRET123")


def test_a_key_added_later_joins_live_and_benches_survive(tmp_path):
    f = tmp_path / "providers.env"
    clock = Clock()
    a = Fake("online", "u", None, "a-free")
    a.fail = ModelUnavailable("x", status=429, retry_after_s=3600)
    p = ModelPool("online", [Member(a, group="opencode", provider="opencode")], clock=clock,
                  extra=lambda: providers.members(providers.load_keys(f), "online", 12.0, make=Fake),
                  token=lambda: providers.keys_token(f))
    with pytest.raises(ModelUnavailable):
        list(p.stream(MSGS))
    f.write_text("GROQ_API_KEY=gsk_aaaaaaaa\n", encoding="utf-8")
    clock.t += pool.RELOAD_EVERY_S + 1
    assert "".join(p.stream(MSGS)) == "openai/gpt-oss-120b"
    assert not p.states()[0].live  # opencode's bench survived the reload
    f.unlink()
    clock.t += pool.RELOAD_EVERY_S + 1
    with pytest.raises(ModelUnavailable):
        list(p.stream(MSGS))  # the key is gone, and so is groq
    assert [m.provider for m in p.members] == ["opencode"]


def test_a_rejected_key_benches_the_provider_and_a_missing_model_itself():
    ms = providers.members({"GROQ_API_KEY": "gsk_aaaaaaaa", "CEREBRAS_API_KEY": "csk-bbbbbbbb"}, "online", 12.0,
                           make=Fake)
    p = ModelPool("online", ms, clock=Clock())
    ms[0].model.fail = ModelUnavailable("x: HTTP 404", status=404)  # cerebras gpt-oss-120b: no such model
    ms[1].model.fail = ModelUnavailable("x: HTTP 401", status=401)  # cerebras' key rejected
    assert "".join(p.stream(MSGS)) == "openai/gpt-oss-120b"  # groq takes it
    states = {(s.provider, s.name): s for s in p.states()}
    assert not states[("cerebras", "gpt-oss-120b")].live and not states[("cerebras", "qwen-3.8-27b")].live
    assert states[("cerebras", "qwen-3.8-27b")].benched_for_s == pool.GONE_BENCH_S
    assert all(s.live for k, s in states.items() if k[0] == "groq")


def test_an_account_wide_limit_benches_the_whole_provider():
    ms = providers.members({"OPENROUTER_API_KEY": "sk-or-cccccccc"}, "online", 12.0, make=Fake)
    ms[0].model.fail = ModelUnavailable("x: HTTP 429", status=429, retry_after_s=60)
    p = ModelPool("online", ms, clock=Clock())
    with pytest.raises(ModelUnavailable):
        list(p.stream(MSGS))
    assert sum(m.model.calls for m in ms) == 1  # one 429, then the rest are benched with it


def test_online_pair_picks_up_providers_env(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "keys_path", lambda: tmp_path / "providers.env")
    (tmp_path / "providers.env").write_text("GROQ_API_KEY=gsk_aaaaaaaa\n", encoding="utf-8")
    online, vision = models.online_pair({})
    assert "groq" in {m.provider for m in online.members}
    assert online.members[0].provider == "opencode"  # the configured model still leads
    assert all(m.vision for m in vision.members)


def test_configured_lists_ready_providers(tmp_path):
    f = tmp_path / "providers.env"
    f.write_text("GROQ_API_KEY=a1b2c3d4\nCLOUDFLARE_API_TOKEN=t\n", encoding="utf-8")
    assert providers.configured({"CEREBRAS_API_KEY": "zz"}, f) == ["cerebras", "groq"]
