"""The network tools (weather, lookup) behind a fake-able HTTP client. Nothing here raises: a failure is a kind result.

Weather sends only the place and lookup only the query, with the Riya User-Agent. Response bodies are untrusted.
"""
from __future__ import annotations

import http.client
import json
import math
from typing import Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from riya.agents.hub import summarize

from .types import ToolResult

__all__ = ["USER_AGENT", "NET_TIMEOUT_S", "HttpClient", "UrlHttp", "FakeHttp", "weather", "lookup"]

USER_AGENT = "Riya/0.5 (personal desktop companion)"
NET_TIMEOUT_S = 8.0
MAX_BODY = 1_000_000
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_BAD = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError)  # a bad body or link
_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rainy", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snowy", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
        96: "thunderstorms with hail", 99: "thunderstorms with hail"}


class HttpClient(Protocol):
    def get_json(self, url: str, params: dict | None, timeout_s: float) -> object: ...


class UrlHttp:
    """GET with urllib: the Riya User-Agent, a timeout, and at most 1 MB of JSON. Raises OSError or ValueError."""

    def get_json(self, url: str, params: dict | None = None, timeout_s: float = NET_TIMEOUT_S) -> object:
        full = f"{url}?{urlencode(params)}" if params else url
        req = Request(full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(req, timeout=timeout_s) as r:
                body = r.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise ValueError("response too large")
            return json.loads(body)
        except http.client.HTTPException as e:  # not an OSError, but the same kind of failure
            raise OSError(str(e)) from e
        except RecursionError as e:  # absurdly nested JSON
            raise ValueError("bad JSON") from e


class FakeHttp:
    """Canned answers by URL (a body, or an exception to raise); `.requests` records (url, params); an unknown URL
    raises OSError."""

    def __init__(self, responses: dict | None = None):
        self.responses, self.requests = dict(responses or {}), []

    def get_json(self, url: str, params: dict | None = None, timeout_s: float = NET_TIMEOUT_S) -> object:
        self.requests.append((url, dict(params) if params is not None else None))
        if url not in self.responses:
            raise OSError(f"no fake response for {url}")
        body = self.responses[url]
        if isinstance(body, BaseException):
            raise body
        return body


def _num(v, lo: float, hi: float) -> float:
    """v as a finite float within [lo, hi], else ValueError (bools and strings that aren't numbers are refused)."""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise TypeError
    f = float(v)
    if not math.isfinite(f) or not lo <= f <= hi:
        raise ValueError
    return f


def weather(http, place: str) -> ToolResult:
    """Open-Meteo: geocode the place, then today's forecast. Always a ToolResult."""
    shown = summarize(place, 40)
    try:
        try:
            hit = http.get_json(GEOCODE_URL, {"name": place, "count": "1"}, NET_TIMEOUT_S)["results"][0]
        except (KeyError, IndexError):
            return ToolResult(False, f"I couldn't find a place called {shown}.")
        lat, lon = _num(hit["latitude"], -90, 90), _num(hit["longitude"], -180, 180)
        name = summarize(hit.get("name") or place, 40) or shown
        body = http.get_json(FORECAST_URL, {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min", "timezone": "auto", "forecast_days": "1"}, NET_TIMEOUT_S)
        cur, day = body["current"], body["daily"]
        temp, code = _num(cur["temperature_2m"], -100, 70), int(_num(cur["weather_code"], 0, 1000))
        hi, lo = _num(day["temperature_2m_max"][0], -100, 70), _num(day["temperature_2m_min"][0], -100, 70)
    except OSError:
        return ToolResult(False, "I couldn't reach the weather service just now.")
    except _BAD:
        return ToolResult(False, "The weather service sent me something I couldn't read.")
    words = _WMO.get(code, "unsettled")
    return ToolResult(True, f"{name}: {round(temp)}°C and {words} right now; today {round(lo)} to {round(hi)}°C.",
                      {"place": name, "temp_c": temp, "code": code})


def lookup(http, query: str) -> ToolResult:
    """Wikipedia: opensearch for the best title, then its REST summary (the first sentences, at most 200 chars)."""
    shown = summarize(query, 40)
    try:
        found = http.get_json(WIKI_API_URL, {"action": "opensearch", "search": query, "limit": "1", "namespace": "0",
                                             "format": "json"}, NET_TIMEOUT_S)
        titles = found[1]
        if not isinstance(titles, list):
            raise TypeError
        if not titles:
            return ToolResult(False, f"I couldn't find anything on {shown}.")
        title = titles[0]
        if not isinstance(title, str) or not title:
            raise ValueError
        page = http.get_json(WIKI_SUMMARY_URL + quote(title.replace(" ", "_"), safe=""), {}, NET_TIMEOUT_S)
        name = summarize(title, 40)
        if page.get("type") == "disambiguation":
            return ToolResult(False, f"{name} could mean a lot of things. Could you be more specific?")
        extract = page["extract"]
        if not isinstance(extract, str):
            raise TypeError
    except OSError:
        return ToolResult(False, "I couldn't reach Wikipedia just now.")
    except _BAD:
        return ToolResult(False, "Wikipedia sent me something I couldn't read.")
    text = summarize(extract, 200)
    if not text:
        return ToolResult(False, f"I found {name}, but it has no summary.")
    return ToolResult(True, text, {"title": name})
