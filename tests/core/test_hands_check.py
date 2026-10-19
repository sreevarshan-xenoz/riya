"""The hands live check against canned http and recording effects: it prints a verdict and runs nothing."""
from evaluation import hands_check as hc
from riya.core.hands import net
from riya.core.hands.effects import FakeEffects
from riya.core.hands.net import FakeHttp

GEO = {"results": [{"name": "London", "latitude": 51.5, "longitude": -0.12}]}
FORECAST = {"current": {"temperature_2m": 12.4, "weather_code": 3},
            "daily": {"temperature_2m_max": [14.0], "temperature_2m_min": [8.0]}}
SUMMARY = {"type": "standard", "title": "Alan Turing", "extract": "Alan Turing was an English mathematician."}
CANNED = {net.GEOCODE_URL: GEO, net.FORECAST_URL: FORECAST, net.WIKI_API_URL: ["Alan Turing", ["Alan Turing"], [], []],
          net.WIKI_SUMMARY_URL + "Alan_Turing": SUMMARY}


def test_verdict_ok_and_nothing_is_run(capsys):
    effects, http = FakeEffects(), FakeHttp(CANNED)
    assert hc.main([], http=http, effects=effects) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "VERDICT: ok"
    assert effects.calls == []
    assert 'Run "backup"' in out and "robocopy" in out  # the dry-run confirmation question
    assert {u for u, _ in http.requests} <= set(CANNED)  # only the T0 network tools asked anything


def test_city_flag_reaches_the_weather_request(capsys):
    http = FakeHttp(CANNED)
    assert hc.main(["--city", "Pune"], http=http, effects=FakeEffects()) == 0
    assert (net.GEOCODE_URL, {"name": "Pune", "count": "1"}) in http.requests


def test_no_network_names_weather_and_lookup(capsys):
    effects = FakeEffects()
    assert hc.main([], http=FakeHttp({}), effects=effects) == 0
    verdict = capsys.readouterr().out.strip().splitlines()[-1]
    assert verdict.startswith("VERDICT: FAIL") and "weather" in verdict and "lookup" in verdict
    assert "time" not in verdict.split() and effects.calls == []
