"""weather, lookup and UrlHttp: all against FakeHttp or a monkeypatched urlopen. Never the network."""
import http.client
import io
import json
from urllib.error import HTTPError

import pytest

from riya.core.hands import net
from riya.core.hands.net import FakeHttp, UrlHttp, lookup, weather
from riya.core.hands.tools import REGISTRY, Call, ToolCtx, validate_call
from riya.core.settings import Settings

GEO = {"results": [{"name": "Pune", "latitude": 18.5196, "longitude": 73.8553, "country": "India"}]}
FORECAST = {"current": {"temperature_2m": 28.4, "weather_code": 2},
            "daily": {"temperature_2m_max": [31.2], "temperature_2m_min": [22.0]}}
RESP = {net.GEOCODE_URL: GEO, net.FORECAST_URL: FORECAST}


def test_weather_calls_and_text():
    http = FakeHttp(RESP)
    r = weather(http, "Pune")
    assert http.requests == [
        (net.GEOCODE_URL, {"name": "Pune", "count": "1"}),
        (net.FORECAST_URL, {"latitude": "18.5196", "longitude": "73.8553", "current": "temperature_2m,weather_code",
                            "daily": "temperature_2m_max,temperature_2m_min", "timezone": "auto",
                            "forecast_days": "1"})]
    assert r.ok and "Pune" in r.text and "28" in r.text and "partly cloudy" in r.text and "22" in r.text
    assert len(r.text) <= 200


def test_weather_failures_are_kind_results():
    for http in (FakeHttp({net.GEOCODE_URL: {}}), FakeHttp({net.GEOCODE_URL: {"results": []}})):
        r = weather(http, "Nowhereville")
        assert not r.ok and "Nowhereville" in r.text
    assert not weather(FakeHttp(), "Pune").ok  # unknown URL: OSError
    assert not weather(FakeHttp({net.GEOCODE_URL: TimeoutError()}), "Pune").ok
    assert not weather(None, "Pune").ok


@pytest.mark.parametrize("geo", [{"results": [{}]}, "oops", {"results": ["x"]}, {"results": [{"latitude": "x",
                                 "longitude": 1}]}, {"results": [{"latitude": 99, "longitude": 1}]}, [], None,
                                {"results": [{"latitude": float("nan"), "longitude": 1}]},
                                {"results": [{"latitude": True, "longitude": 1}]}])
def test_weather_malformed_geocoding_body(geo):
    r = weather(FakeHttp({net.GEOCODE_URL: geo}), "Pune")
    assert not r.ok and r.text


@pytest.mark.parametrize("fc", ["oops", {}, {"current": {}, "daily": {}}, {"current": {"temperature_2m": None,
                               "weather_code": 1}, "daily": FORECAST["daily"]}, {"current": FORECAST["current"],
                               "daily": {"temperature_2m_max": [], "temperature_2m_min": []}},
                               {"current": {"temperature_2m": 1e9, "weather_code": 1}, "daily": FORECAST["daily"]},
                               {"current": {"temperature_2m": 20, "weather_code": float("inf")},
                                "daily": FORECAST["daily"]}, [1, 2], None])
def test_weather_malformed_forecast_body(fc):
    r = weather(FakeHttp({net.GEOCODE_URL: GEO, net.FORECAST_URL: fc}), "Pune")
    assert not r.ok and r.text


def test_weather_unknown_code_and_hostile_place_name():
    http = FakeHttp({net.GEOCODE_URL: {"results": [{**GEO["results"][0], "name": "Pu\nne\x1b[31m api_key=sk-abc12345"}]},
                     net.FORECAST_URL: {**FORECAST, "current": {"temperature_2m": 3, "weather_code": 123}}})
    r = weather(http, "Pune")
    assert r.ok and "unsettled" in r.text and "\n" not in r.text and "sk-abc12345" not in r.text


def test_weather_tool_run_place_rules():
    s = Settings()
    http = FakeHttp(RESP)

    def go(**args):
        call = validate_call(Call("weather", args), s)
        return REGISTRY["weather"].run(call.args, ToolCtx(settings=s, http=http))

    r = go()
    assert not r.ok and "city" in r.text and http.requests == []
    r = go(place="Pune")
    assert r.ok and r.data["save_place"] == "Pune"
    s.weather_place = "Delhi"
    r = go(place="Pune")
    assert r.ok and "save_place" not in (r.data or {}) and http.requests[-2][1]["name"] == "Pune"
    r = go()
    assert r.ok and http.requests[-2][1]["name"] == "Delhi"
    assert "save_place" not in (go().data or {})


def test_weather_tool_does_not_save_a_failed_place():
    s = Settings()
    call = validate_call(Call("weather", {"place": "Nowhere"}), s)
    r = REGISTRY["weather"].run(call.args, ToolCtx(settings=s, http=FakeHttp({net.GEOCODE_URL: {}})))
    assert not r.ok and not r.data


# ---- lookup ----
SEARCH = ["Ada", ["Ada Lovelace"], [""], ["https://en.wikipedia.org/wiki/Ada_Lovelace"]]
SUMMARY = {"type": "standard", "title": "Ada Lovelace", "extract": "Augusta Ada King was an English mathematician."}


def test_lookup_calls_and_text():
    http = FakeHttp({net.WIKI_API_URL: SEARCH, net.WIKI_SUMMARY_URL + "Ada_Lovelace": SUMMARY})
    r = lookup(http, "Ada")
    assert http.requests == [(net.WIKI_API_URL, {"action": "opensearch", "search": "Ada", "limit": "1",
                                                 "namespace": "0", "format": "json"}),
                             (net.WIKI_SUMMARY_URL + "Ada_Lovelace", {})]
    assert r.ok and r.text == SUMMARY["extract"]


def test_lookup_title_is_quoted_and_text_is_capped():
    search = ["x", ["AC/DC & 100% (band)?"], [], []]
    url = net.WIKI_SUMMARY_URL + "AC%2FDC_%26_100%25_%28band%29%3F"
    http = FakeHttp({net.WIKI_API_URL: search, url: {"extract": "word " * 200}})
    r = lookup(http, "acdc")
    assert http.requests[1][0] == url and r.ok and len(r.text) <= 200


def test_lookup_failures():
    dis = {"type": "disambiguation", "title": "Mercury", "extract": "Mercury may refer to:"}
    r = lookup(FakeHttp({net.WIKI_API_URL: ["m", ["Mercury"], [], []], net.WIKI_SUMMARY_URL + "Mercury": dis}), "m")
    assert not r.ok and "specific" in r.text
    r = lookup(FakeHttp({net.WIKI_API_URL: ["zzz", [], [], []]}), "zzz")
    assert not r.ok and "zzz" in r.text
    assert not lookup(FakeHttp(), "x").ok and not lookup(None, "x").ok
    assert not lookup(FakeHttp({net.WIKI_API_URL: SEARCH}), "x").ok  # the summary request fails (OSError)


@pytest.mark.parametrize("search", [[], {}, "oops", None, ["a"], ["a", "b"], ["a", [7]], ["a", [""]], ["a", [None]]])
def test_lookup_malformed_search_body(search):
    r = lookup(FakeHttp({net.WIKI_API_URL: search}), "x")
    assert not r.ok and r.text


@pytest.mark.parametrize("page", ["oops", [], None, {}, {"extract": 5}, {"extract": None}, {"extract": ""},
                                  {"extract": "   "}])
def test_lookup_malformed_summary_body(page):
    http = FakeHttp({net.WIKI_API_URL: SEARCH, net.WIKI_SUMMARY_URL + "Ada_Lovelace": page})
    r = lookup(http, "Ada")
    assert not r.ok and r.text


def test_lookup_output_is_redacted_one_line():
    page = {"extract": "line one\nline two api_key=sk-abc12345678 \x1b[31m"}
    r = lookup(FakeHttp({net.WIKI_API_URL: SEARCH, net.WIKI_SUMMARY_URL + "Ada_Lovelace": page}), "Ada")
    assert r.ok and "\n" not in r.text and "sk-abc12345678" not in r.text and "\x1b" not in r.text


# ---- UrlHttp ----
class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(monkeypatch, body=b"{}", exc=None):
    seen = {}

    def fake(req, timeout=None):
        seen.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, timeout=timeout)
        if exc:
            raise exc
        return Resp(body)

    monkeypatch.setattr(net, "urlopen", fake)
    return seen


def test_urlhttp_sends_user_agent_timeout_and_params(monkeypatch):
    seen = patch_urlopen(monkeypatch, json.dumps({"a": 1}).encode())
    out = UrlHttp().get_json("https://x.test/api", {"q": "a b", "n": "1"}, net.NET_TIMEOUT_S)
    assert out == {"a": 1} and seen["timeout"] == 8.0 == net.NET_TIMEOUT_S
    assert seen["headers"]["user-agent"] == "Riya/0.5 (personal desktop companion)" == net.USER_AGENT
    assert seen["url"] == "https://x.test/api?q=a+b&n=1"
    UrlHttp().get_json("https://x.test/api", None, 3)
    assert seen["url"] == "https://x.test/api" and seen["timeout"] == 3


def test_urlhttp_limits_and_errors(monkeypatch):
    patch_urlopen(monkeypatch, b"[" + b"0," * 600_000 + b"0]")
    with pytest.raises(ValueError):
        UrlHttp().get_json("https://x.test/", None, 8.0)
    patch_urlopen(monkeypatch, b"not json")
    with pytest.raises(ValueError):
        UrlHttp().get_json("https://x.test/", None, 8.0)
    patch_urlopen(monkeypatch, b"[" * 400_000)
    with pytest.raises(ValueError):
        UrlHttp().get_json("https://x.test/", None, 8.0)
    patch_urlopen(monkeypatch, exc=HTTPError("https://x.test/", 404, "nf", {}, None))
    with pytest.raises(OSError):
        UrlHttp().get_json("https://x.test/", None, 8.0)
    patch_urlopen(monkeypatch, exc=http.client.IncompleteRead(b""))
    with pytest.raises(OSError):
        UrlHttp().get_json("https://x.test/", None, 8.0)


def test_urlhttp_failures_become_kind_results(monkeypatch):
    patch_urlopen(monkeypatch, b"<html>captive portal</html>")
    assert not weather(UrlHttp(), "Pune").ok and not lookup(UrlHttp(), "Ada").ok
