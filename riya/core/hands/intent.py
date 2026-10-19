"""Understanding requests: pure, deterministic parsing of what the user's turn asks her to do.

`match` turns a turn into a `Call` (or None); `parse_duration` / `parse_when` read durations and clock times;
`is_command_form` says whether a turn is a clear command; `is_forbidden` spots T3 requests she never acts on.
Nothing here touches the clock, the OS or the network: `now` and `tz` are always passed in.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .types import Call

SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com", "github": "https://github.com", "gmail": "https://mail.google.com",
    "google": "https://www.google.com", "reddit": "https://www.reddit.com", "x": "https://x.com",
    "twitter": "https://x.com", "netflix": "https://www.netflix.com", "spotify": "https://open.spotify.com",
}
_I = re.I

# ---- durations ----
_ONES9 = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORDS = {w: i + 1 for i, w in enumerate(_ONES9)} | {w: i + 10 for i, w in enumerate(_TEENS)} | {"a": 1, "an": 1}


def _alt(ws) -> str:
    return "|".join(sorted(ws, key=len, reverse=True))  # longest first, so "fourteen" beats "four"


_SUB99 = rf"(?:(?:{_alt(_TENS)})(?:[\s-](?:{_alt(_ONES9)}))?|{_alt(_ONES9 + _TEENS)})"
_NUMW = rf"(?:(?:an?|{_alt(_ONES9)})\s+hundred(?:\s+(?:and\s+)?{_SUB99})?|{_SUB99})"
_UNIT = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?)"
_UNITS = {"h": 3600, "hr": 3600, "hour": 3600, "m": 60, "min": 60, "minute": 60, "s": 1, "sec": 1, "second": 1,
          "d": 86400, "day": 86400, "week": 604800}
_DUR = re.compile(
    rf"\b(?:(?P<d>\d+(?:\.\d+)?)(?P<dh>\s+and\s+a\s+half)?(?:\s*(?P<du>{_UNIT})|(?P<ds>[hmsd]))"
    rf"|(?P<w>{_NUMW})(?P<wh>\s+and\s+a\s+half)?\s+(?P<wu>{_UNIT})"
    rf"|(?P<a>an?\s+half|half(?:\s+an?)?|an?)\s+(?P<au>{_UNIT}))\b(?P<post>\s+and\s+a\s+half\b)?", _I)
_NUMLEFT = re.compile(rf"\b(?:\d+|{_alt(_TENS)}|{_alt(_ONES9 + _TEENS)}|hundred|thousand|half|quarter)\b", _I)
MAX_TIMER_S = 24 * 3600
MAX_REMINDER_S = 366 * 24 * 3600
MAX_TEXT = 500  # `match` and the other detectors only read this many characters of a turn (to recognise it)


def _word_num(w: str) -> int:
    total = 0
    for t in re.findall(r"[a-z]+", w.lower()):
        if t == "hundred":
            total = (total or 1) * 100
        elif t != "and":
            total += _TENS.get(t) or _WORDS[t]
    return total


def _unit(u: str) -> int:
    u = u.lower()
    return _UNITS[u.rstrip("s") if len(u) > 1 else u]


def _duration(text: str, strict: bool) -> float | None:
    """Total seconds of every duration phrase in text; None if none, non-positive, non-finite or garbled.

    A number left unparsed (a stray digit or number word) is a garble: guessing would set the wrong time.
    `strict` also forbids any other leftover words.
    """
    s = text[:MAX_TEXT].lower().strip(" .,!?")
    total, hit = 0.0, False
    for mm in _DUR.finditer(s):
        if mm["d"] is not None:
            n, unit, half = float(mm["d"]), _unit(mm["du"] or mm["ds"]), bool(mm["dh"])
        elif mm["w"] is not None:
            n, unit, half = _word_num(mm["w"]), _unit(mm["wu"]), bool(mm["wh"])
        else:
            n, unit, half = (0.5 if "half" in mm["a"] else 1.0), _unit(mm["au"]), False
        total += n * unit + (unit / 2 if half or mm["post"] else 0)
        hit = True
    if not hit or not 0 < total < float("inf"):
        return None
    rest = _DUR.sub(" ", s)
    if _NUMLEFT.search(rest) or (strict and re.sub(r"\band\b|[\s,&]", "", rest)):
        return None
    return float(total)


def parse_duration(text: str) -> float | None:
    """'10 minutes', 'an hour and a half', '2 hrs 15 min' -> seconds; None if there's no positive duration."""
    return _duration(text, strict=False)


# ---- clock times ----
_TIME = re.compile(
    r"^(?:(?P<day>tomorrow|today)\s+)?(?P<at>at\s+)?(?:(?P<word>noon|midnight)|(?P<h>\d{1,2})(?::(?P<mi>\d{2}))?"
    r"(?:\s*(?P<ap>[ap])\.?m\.?)?(?:\s*o'?clock)?)$", _I)


def _local_now(now: float, tz) -> datetime:
    if isinstance(now, datetime):
        now = now.timestamp()
    return datetime.fromtimestamp(now, tz) if tz else datetime.fromtimestamp(now).astimezone()


def _stamps(d: date, h: int, mi: int, tz) -> list[float]:
    """Epoch times at which the local wall clock reads d h:mi (two when a fall-back hour repeats)."""
    naive = datetime(d.year, d.month, d.day, h, mi)
    out: list[float] = []
    for fold in (0, 1):
        n = naive.replace(fold=fold)
        t = (n.replace(tzinfo=tz) if tz else n.astimezone()).timestamp()
        back = datetime.fromtimestamp(t, tz) if tz else datetime.fromtimestamp(t).astimezone()
        if back.replace(tzinfo=None) == naive and t not in out:
            out.append(t)
    if not out:  # a spring-forward gap: the wall time doesn't exist, so use the clock's own reading of it
        n = naive
        out.append((n.replace(tzinfo=tz) if tz else n.astimezone()).timestamp())
    return out


def _hours(h: int, ap: str | None, fixed_day: bool) -> list[int]:
    if ap:
        return [h % 12 + (12 if ap.lower() == "p" else 0)] if 1 <= h <= 12 else []
    if h > 23:
        return []
    if h == 0 or h > 12:
        return [h]
    if fixed_day:  # "tomorrow at 5" is the evening, "tomorrow at 9" the morning
        return [12] if h == 12 else [h + 12] if h <= 6 else [h]
    return sorted({h % 12, h % 12 + 12})


def parse_when(text: str, now: float, tz=None) -> float | None:
    """'at 5', 'at 5:30 pm', 'tomorrow at 9', 'in 20 minutes' -> epoch seconds, or None.

    A clock time is the first occurrence strictly after now ("at 5" at 17:30 is 05:00 tomorrow); "in ..." is
    elapsed time, so it ignores DST.
    """
    s = re.sub(r"\s+", " ", text[:MAX_TEXT].replace("’", "'").strip(" .,!?")).lower()
    base = now.timestamp() if isinstance(now, datetime) else float(now)
    if (mi := re.match(r"in (.+)$", s)):
        d = _duration(mi.group(1), strict=True)
        return None if d is None or d > MAX_REMINDER_S else base + d
    mm = _TIME.match(s)
    if not mm or not (mm["day"] or mm["at"]):
        return None
    if mm["word"]:
        hours, minute = [12 if mm["word"] == "noon" else 0], 0
    else:
        minute = int(mm["mi"] or 0)
        hours = _hours(int(mm["h"]), mm["ap"], mm["day"] == "tomorrow")
        if minute > 59:
            return None
    today = _local_now(base, tz).date()
    days = {"tomorrow": [today + timedelta(1)], "today": [today]}.get(mm["day"], [today, today + timedelta(1)])
    after = [t for d in days for h in hours for t in _stamps(d, h, minute, tz) if t > base]
    return min(after) if after else None


# ---- shared lead-ins ----
_LEAD = re.compile(
    r"^(?:(?:hey|hi|ok|okay)(?:[\s,]+|$))?(?:riya(?:[\s,:!]+|$))?(?:please(?:[\s,]+|$))?"
    r"(?:(?:can|could|would|will)\s+you(?:[\s,]+|$))?(?:please(?:[\s,]+|$))?", _I)
_EXTRA = re.compile(r"^(?:i(?:'d| would)? (?:want|need|like) you to|go ahead and|just|now|then)\s+", _I)
_NEG = re.compile(r"^(?:don'?t|do not|never|not|no|stop)\b", _I)


def _prep(text: str, extra: bool = False, limit: int | None = MAX_TEXT) -> str:
    """The turn (its first `limit` chars; None: all of it) without its trailing punctuation/please and without
    'hey riya, can you please' lead-ins."""
    s = re.sub(r"\s+", " ", (text[:limit] if limit else text).replace("’", "'").strip())
    s = re.sub(r"[\s.!?,;]+$", "", s)
    s = re.sub(r"[\s,]+please$", "", s, flags=_I)
    s = _LEAD.sub("", s, count=1)
    if extra:
        s = _LEAD.sub("", _EXTRA.sub("", s), count=1)
    return s.strip()


# ---- match ----
_TLD_NOT = {"txt", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "py", "js", "ts", "json", "csv", "md", "log",
            "exe", "msi", "bat", "cmd", "ps1", "sh", "dll", "lnk", "zip", "rar", "7z", "png", "jpg", "jpeg", "gif",
            "mp3", "mp4", "wav", "ini", "cfg", "yml", "yaml", "xml", "html", "htm", "css", "tmp", "bak", "rtf", "php",
            "odt", "ods", "bmp", "svg", "webp", "mov", "avi", "mkv", "flac", "tar", "gz", "iso", "jar", "rb", "rs",
            "go", "java", "cs", "cpp", "c", "h", "ps", "toml", "sql", "db", "dat", "bin", "apk", "dmg", "pkg", "tsx",
            "jsx", "vbs", "lock"}
_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", _I)


def _remind(s, now, tz):
    mm = re.match(r"remind me (.+)$", s, _I)
    if not mm:
        return None
    rest = mm.group(1)
    for sep in re.finditer(r"\s+(?:to|that)\s+", rest, _I):  # "remind me <when> to <text>"
        when, text = rest[:sep.start()].strip(), rest[sep.end():].strip()
        if text and parse_when(when, now, tz) is not None:
            return Call("reminder_set", {"when": when, "text": text})
    body = re.match(r"(?:to|that) (.+)$", rest, _I)  # "remind me to <text> <when>"
    if body:
        for sep in re.finditer(r"\s+(?=(?:tomorrow|today|at|in)\b)", body.group(1), _I):
            text, when = body.group(1)[:sep.start()].strip(), body.group(1)[sep.end():].strip()
            if text and parse_when(when, now, tz) is not None:
                return Call("reminder_set", {"when": when, "text": text})
    return None


def _timer(s, now, tz):
    if not (re.match(r"(?:set|start|create|make)\b", s, _I) or re.match(r"timer\b", s, _I)) or \
            not re.search(r"\btimer\b", s, _I):
        return None
    d = parse_duration(s)
    return Call("timer_set", {"duration": d}) if d and d <= MAX_TIMER_S else None


def _cancel(s, now, tz):
    mm = re.match(r"(?:cancel|delete|remove|clear) (?:the |my )?(?:(timers?)|reminders?(?: (?:about|for|to|that) (.+))?)$",
                  s, _I)
    if not mm:
        return None
    text = "timer" if mm.group(1) else mm.group(2)
    return Call("reminder_cancel", {"text": text.strip()}) if text else None


def _lists(s, now, tz):
    if re.match(r"(?:(?:what(?:'s| is| are)|show me|show|list|read|check)(?: me)?(?: all| any)?(?: of)?(?: my| the)? "
                r"|do i have (?:any )?)(?:reminders|timers)$", s, _I):
        return Call("reminder_list")
    if re.match(r"(?:what(?:'s| is| are)(?: in)? my notes|(?:show|read|list)(?: me)?(?: all)?(?: my)? notes|my notes)$",
                s, _I):
        return Call("note_list")
    return None


def _note(s, now, tz):
    mm = re.match(r"(?:(?:take|make|add) a note(?: that |: ?| )|note(?: that |: ?| to self:? ?))(.+)$", s, _I)
    return Call("note_add", {"text": mm.group(1).strip()}) if mm else None


def _open(s, now, tz, apps=(), commands=()):
    mm = re.match(r"(?:open|launch|start) (.+)$", s, _I)
    if not mm:
        return None
    x = re.sub(r"^(?:up )?(?:the |my )?", "", mm.group(1).strip(), flags=_I)
    x = re.sub(r" (?:app|application|website|site|page|in (?:the |my )?browser|for me)$", "", x, flags=_I).strip()
    for name in apps:
        if name.casefold() == x.casefold():
            return Call("open_app", {"name": name})
    if x.lower() in SITES:
        return Call("open_url", {"url": SITES[x.lower()]})
    x = re.sub(r"^https?://", "", x, flags=_I)
    if _DOMAIN.match(x):
        host, sep, path = x.partition("/")
        tld = host.rsplit(".", 1)[-1].lower()
        if re.search(r"[a-z]", tld) and tld not in _TLD_NOT:
            return Call("open_url", {"url": f"https://{host.lower()}{sep}{path}"})
    return None


_MEDIA = [
    (r"(?:play )?(?:the )?(?:previous|last) (?:track|song)|previous|go back (?:a |one )?(?:track|song)", "previous"),
    (r"skip(?: (?:to )?(?:the |this |that |a )?(?:next )?(?:track|song|one))?|skip to (?:the )?next|next (?:track|song)",
     "next"),
    (r"(?:pause|resume|unpause|play)(?: the| my| some)?(?: music| song| songs| track| media| playback| video| audio)?",
     "play_pause"),
]
_VOL = [
    (r"(?:turn |bring |crank )?(?:the )?volume (up|down)", 1),
    (r"turn (?:it|the music|the sound|the volume) (up|down)", 1),
    (r"turn (up|down)(?: the)? (?:volume|music|sound)", 1),
    (r"(?:(raise|increase)|(lower|decrease|reduce))(?: the)? volume", 0),
    (r"(?:make it |a little |a bit )?(?:(louder)|(quieter|softer))", 0),
]


def _media(s, now, tz):
    for pat, action in _MEDIA:
        if re.fullmatch(pat, s, _I):
            return Call("media", {"action": action})
    return None


def _volume(s, now, tz):
    if re.fullmatch(r"(?:mute|unmute)(?: the| my)?(?: sound| volume| audio| music| computer| it)?", s, _I):
        return Call("volume", {"action": "mute"})
    for pat, direct in _VOL:
        mm = re.fullmatch(pat, s, _I)
        if mm:
            up = mm.group(1).lower() in ("up",) if direct else bool(mm.group(1))
            return Call("volume", {"action": "up" if up else "down"})
    return None


def _time_now(s, now, tz):
    if re.fullmatch(r"what time is it(?: right now| now)?|what(?:'s| is) the (?:current )?time(?: right now| now)?|"
                    r"(?:tell|give) me the time|current time", s, _I):
        return Call("time_now")
    return None


def _weather(s, now, tz):
    mm = re.fullmatch(r"(?:(?:what(?:'s| is)|how(?:'s| is)) the weather(?: like)?|(?:check |get |tell me )?(?:the )?"
                      r"weather(?: report| forecast)?)(?: (?:in|at|for|near) (?P<p>.+?))?"
                      r"(?: (?:today|now|right now|outside|tonight))?", s, _I)
    if not mm:
        return None
    return Call("weather", {"place": mm["p"].strip()} if mm["p"] else {})


def _copy(s, now, tz):
    mm = re.match(r"copy (.+)$", s, _I)
    if not mm:
        return None
    t = re.sub(r" to (?:the |my )?clipboard$", "", mm.group(1).strip(), flags=_I).strip().strip("\"'").strip()
    if not t or t.lower() in {"this", "that", "it", "them", "these", "those"}:
        return None
    return Call("clipboard_copy", {"text": t})


def _run(s, now, tz, commands=()):
    mm = re.match(r"run (?:the |my )?(.+?)(?: command)?$", s, _I)
    if mm:
        for name in commands:
            if name.casefold() == mm.group(1).strip().casefold():
                return Call("run_command", {"name": name})
    return None


_OPS = [(r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\b(?:times|multiplied by)\b", "*"), (r"\bdivided by\b", "/"),
        (r"\bto the power of\b", "^"), (r"(?<=\d)\s*[x×]\s*(?=\d)", " * "), ("÷", "/")]


_DATEISH = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _calc(s, now, tz):
    mm = re.match(r"(?:what(?:'s| is)|how much is|calculate|calc|compute|work out) (.+)$", s, _I)
    if not mm:
        return None
    e = mm.group(1).strip()
    for pat, rep in _OPS:
        e = re.sub(pat, rep, e, flags=_I)
    e = re.sub(r"\s+", " ", e).strip()
    if len(e) > 100 or "**" in e or e.count("^") > 1 or _DATEISH.search(e):  # no power towers; dates aren't sums
        return None
    if re.fullmatch(r"[\d\s.+\-*/%^()]+", e) and re.search(r"[\d)]\s*[-+*/%^]\s*[\d(]", e):
        return Call("calc", {"expr": e})
    return None


def _write(s, now, tz):
    mm = re.match(r"(?:write|create|make|save) (?:me )?(?:an? |the )?(?:new )?(?:text )?file (?:called|named) "
                  r"(?:\"([^\"]+)\"|(\S+)) (?:saying|that says|containing|with(?: the)? (?:text|words|contents?)|with) "
                  r"(.+)$", s, _I)
    if not mm:
        return None
    return Call("write_file", {"filename": (mm.group(1) or mm.group(2)).strip(), "text": mm.group(3).strip()})


_NOT_SUBJECT = re.compile(r"\b(?:my|our|me|i|you|your|yours|this|that|it|its|he|she|they|them|him|her|his|we|us|"
                          r"yourself|myself|there|here)\b", _I)
_GENERIC = {"up", "new", "good", "wrong", "going on", "happening", "the matter", "the problem", "the deal", "the plan",
            "the point", "popping", "next", "cooking", "for dinner"}


# status/chat/maths, not a topic: "what's the date today", "what's going on with the build", "what's 9**9**9"
_NOISE = re.compile(r"^(?:on|in|at|with|for|going on)\b|^(?:date|day|time|score)\b|\bgoing on\b|\b(?:today|tonight|"
                    r"right now|tomorrow|yesterday)\b|^[\d\s.+\-*/%^()=]+$|\d\s*[*/^+%=]", _I)


# a question asking for an opinion, a comparison or a how-to, or about what something is doing: chat, not a lookup
_OPINION = re.compile(r"\b(?:good|best|better|worse|worst|difference|differences|way|ways|how|why|should|can|could|"
                      r"would|some|any|favou?rite|recommend\w*|vs|versus|or|doing|going|happening|odds|point)\b", _I)
WHAT_MAX_WORDS = 4  # "what is X" looks X up only when X is a short noun phrase


def _lookup(s, now, tz):
    """Only "who is|was X", "what is|was|are X" (a short noun phrase), "tell me about X", "look up X", "search (for)
    X"; never an opinion, comparison or how-to question (the normal reply answers those)."""
    if (mm := re.match(r"(?:look up|search(?: for)?) (.+)$", s, _I)):
        form = "look"
    elif (mm := re.match(r"(?:who(?:'s| is| was)|tell me about) (.+)$", s, _I)):
        form = "ask"
    elif (mm := re.match(r"what(?:'s| is| was| are) (.+)$", s, _I)):
        form = "what"
    else:
        return None
    q = re.sub(r"^(?:an? |the )", "", mm.group(1).strip(), flags=_I).strip()
    if not q or q.lower() in _GENERIC or _NOT_SUBJECT.search(q) or _NOISE.search(q) or len(q.split()) > 8:
        return None
    if form != "look" and _OPINION.search(q) or form == "what" and len(q.split()) > WHAT_MAX_WORDS:
        return None
    return Call("lookup", {"query": q})


_FREE_TEXT = {"note_add": _note, "clipboard_copy": _copy, "write_file": _write}


def match(text: str, now: float, *, apps=(), commands=(), tz=None) -> Call | None:
    """The deterministic reading of one turn, or None. `apps`/`commands` are the user's allowed names. Only the first
    MAX_TEXT chars are read to recognise it; the free text of a note, a copy or a file is the whole turn's (its tool
    then holds it to its own limit), never cut at MAX_TEXT."""
    c = _match(_prep(text), now, apps, commands, tz)
    if c is not None and c.tool in _FREE_TEXT and len(text) > MAX_TEXT:
        c = _FREE_TEXT[c.tool](_prep(text, limit=None), now, tz)
    return c


def _match(s: str, now: float, apps, commands, tz) -> Call | None:
    if not s or _NEG.match(s):
        return None
    for rule in (_remind, _timer, _cancel, _lists, _note):
        if (c := rule(s, now, tz)):
            return c
    for c in (_open(s, now, tz, apps), _media(s, now, tz), _volume(s, now, tz), _time_now(s, now, tz),
              _weather(s, now, tz), _copy(s, now, tz), _run(s, now, tz, commands), _calc(s, now, tz),
              _write(s, now, tz), _lookup(s, now, tz)):
        if c:
            return c
    return None


# ---- command form ----
_TRIGGER = re.compile(r"(?:open|launch|play|pause|set|remind|start|turn|mute|copy|save|write|run|check|look up|"
                      r"search|find)\b", _I)


def is_command_form(text: str) -> bool:
    """True if the turn opens with an imperative trigger (optionally after hey/riya, can/could/would you, please)."""
    s = _prep(text)
    return bool(s) and not _NEG.match(s) and bool(_TRIGGER.match(s))


# ---- forbidden (T3) ----
_CRED = (r"(?:passwords?|passcodes?|pin (?:number|code)|credit cards?|card numbers?|cvv|bank (?:account|login|details)"
         r"|api keys?|secret keys?|ssh keys?|private keys?|credentials?|2fa codes?|otp)")
_FILE_NOUN = (r"(?:files?|folders?|director(?:y|ies)|downloads|documents|recycle bin|trash|drives?|disks?|system32|"
              r"everything)")
_NOT_TARGET = (r"me|us|myself|is|was|are|were|has|have|had|will|would|can|could|should|does|did|doesn't|isn't|wasn't|"
               r"size|editor|box|field|files?|messages?|received|sent|failed|arrived|notifications?|alerts?|from|settings?|app|and|if|when")
_FORBIDDEN = [re.compile(p, _I) for p in (
    rf"^(?:delete|remove|erase|wipe|trash|shred|format|empty|rename|move|clean out|clear out)\b.*\b{_FILE_NOUN}\b",
    r"^(?:rm|rmdir|del|mv)\s+\S",
    # messaging aimed at someone else; "send me the weather" / "text me when it's done" are fine
    r"^(?:send|forward|shoot|compose) (?:an? |the |my |some )?(?:quick |short )?(?:e-?mails?|messages?|texts?(?! file)|dms?"
    r"|sms|whatsapp(?: messages?)?|tweets?)\b(?! (?:to|for) (?:me|myself)\b)(?! me\b)",
    r"^send (?!me\b|us\b|myself\b)[\w'-]+ (?:an? |the )?(?:e-?mail|message|text|dm|sms)\b",
    r"^send (?:it|this|that|them|these|those|(?:the|an?|my) [\w ]{1,30}?) to (?!me\b|myself\b|my\b|"
    r"the (?:clipboard|printer)\b)[a-z]",
    rf"^(?:e-?mail|text|message|dm|whatsapp|sms)\s+(?!(?:{_NOT_TARGET})\b)[a-z]",
    r"^tweet\b",
    r"^(?:buy|purchase|checkout|check out|subscribe)\b",
    r"^(?:order|book|reserve) (?:me |us )?(?:an?|some|\d+)\b",
    r"^pay (?:for|my|the|his|her|\d|\$)",
    r"^add\b.*\bto (?:my |the )?(?:cart|basket)\b",
    rf"^(?:what(?:'s| is| are| was)|tell me|show me|show|read(?: out)?|give me|reveal|find|get|share|enter|type|fill in|"
    rf"save|store|copy|note|write|dictate|look up|search)\b.*\b(?:(?:my|our|his|her|their|your) (?:[\w-]+ ){{0,2}}"
    rf"{_CRED}|{_CRED} (?:for|to|of))\b",
    r"^(?:run|execute|exec|launch|open|start) (?:an? |the |my )?(?:new )?(?:shell|terminal|cmd|command prompt|"
    r"powershell|pwsh|bash|zsh|console|wsl|git bash)\b",
    r"^(?:run|execute|exec) (?:this|that|the following|an?|some|any) (?:(?:shell|terminal|bash|powershell|cmd) )?"
    r"(?:command|script|code)s?\b",
    r"^(?:type|enter|run|execute|do|issue)\b.*\b(?:in|into|on|using|via|through) (?:the |a |my )?(?:shell|terminal|"
    r"command line|powershell|cmd|console)\b",
    r"(?:^|\s)(?:rm -[rf]+|sudo|powershell -\w+|cmd ?/[ck]|del /[fsq]|format [a-z]:)(?:\s|$)",
)]


def is_forbidden(text: str) -> bool:
    """True if the turn asks for something she never does: shell, deleting/moving files, messages, purchases, credentials."""
    s = _prep(text, extra=True)
    return any(p.search(s) for p in _FORBIDDEN)
