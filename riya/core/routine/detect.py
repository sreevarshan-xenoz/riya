"""The claim guard and the routine, goodbye and dispute detectors (v0.12 §4.5.3, §4.5.4, §4.12.1).

claim_guard drops a model sentence about the user that states a clock time, a duration, or a habit tied to a time of day
or a day, when nothing the turn's prompt carried offered that value: "only describe what the state says", enforced in
code. The detectors are deterministic and read case- and apostrophe-blind. Pure.
"""
import re

TOPICS = ("stretch", "runs", "run_left", "gaming")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("‘", "'").casefold()).strip()


# --- what a sentence states --------------------------------------------------------------------------------------------

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORDS = r"(?:(?:" + "|".join(_TENS) + r")(?:[- ](?:" + "|".join(_ONES) + r"))?|" + "|".join({**_TEENS, **_ONES}) + ")"
_NUM = r"(\d+(?:\.\d+)?|" + _WORDS + r"|half an?|a couple(?: of)?|an?)"
_UNIT_S = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hr": 3600, "hour": 3600, "h": 3600, "m": 60,
           "day": 86400, "week": 7 * 86400}
_UNIT = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
_UNIT_NC = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
_DURATION = re.compile(r"\b" + _NUM + r"( and a half)?[ -](?:(?:whole|entire|solid) )?" + _UNIT + r"\b( and a half)?")
_COMPACT = re.compile(r"\b(\d+(?:\.\d+)?) ?(h|hrs?|m|mins?)\b")
_HOUR_WORD = {**_ONES, "ten": 10, "eleven": 11, "twelve": 12}
_HOUR = r"(\d{1,2}|" + "|".join(sorted(_HOUR_WORD, key=len, reverse=True)) + r")"
_AMPM = r"(?:am|pm|a\.m\.|p\.m\.)"
# a bare hour after a preposition counts only when what follows it reads like a time (not "after 3 tries")
_BARE_END = (r"(?=\s*(?:$|[,.!?~;:)]|o'?clock\b|sharp\b|today\b|tonight\b|again\b|this\b|every\b|on\b|in\b|and\b|or\b|"
             r"yesterday\b|tomorrow\b|last\b|then\b|so\b|already\b|now\b|ish\b|-ish\b|lol\b|haha\b|right\b|huh\b|daily\b|"
             r"for\b|to\b|with\b|when\b|because\b|while\b|(?:mon|tues|wednes|thurs|fri|satur|sun|week)days?\b|"
             r"weekends?\b|mornings?\b|afternoons?\b|evenings?\b|nights?\b))")
_CLOCK = re.compile(r"\b" + _HOUR + r"(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)(?![a-z])"
                    r"|\b(\d{1,2}):(\d{2})\b"
                    r"|\b(?:at|since|by|around|until|till|til|before|after|from|past) (?:about |around |like |maybe "
                    r"|almost |nearly )?" + r"(\d{1,2}|" + "|".join(sorted(_HOUR_WORD, key=len, reverse=True)) + r")"
                    r"(?:ish|-ish)?\b(?![:.]\d| ?" + _UNIT_NC + r"\b| ?" + _AMPM + r")" + _BARE_END +
                    r"|\b(noon|midnight)\b")
_TIME_IDIOMS = re.compile(r"\b(?:at one (?:with|point|time|another|go)|one another|one by one)\b")
_QUARTER = re.compile(r"\b(quarter|half|ten|twenty|five) (past|to) " + _HOUR + r"\b")
_FREQ = re.compile(r"\b(always|usually|normally|typically|never|often|every|mostly|most(?= (?:days|nights|mornings|"
                   r"evenings|afternoons|weekends|weekdays)\b))\b")
_WHEN_WORDS = re.compile(r"\b(mornings?|afternoons?|evenings?|nights?|nightly|tonight|noon|midnight|daily|every ?day|"
                         r"most days|weekdays?|weekends?|weekly|every week|mondays?|tuesdays?|wednesdays?|thursdays?|"
                         r"fridays?|saturdays?|sundays?)\b")
# words that mention the user without saying anything about their routine
_NOT_ABOUT = re.compile(r"\b(thank you|thanks to you|see you|see ya|love you|miss you|missed you|bless you|did you know|"
                        r"do you know|you know|got you|gotcha|i'll tell you|let you know|told you|hug you|"
                        r"you make me \w+|you made me \w+|good (?:morning|afternoon|evening|night))\b")
_YOU = re.compile(r"\b(you|your|yours|you're|you've|you'll|you'd|youre|youve|yourself|ur|u)\b")
_SUGGEST = re.compile(r"\byou (could|should|might|can|may)\b(?!'t|n't)|\bhow about\b|\bwhy not\b|\blet's\b|\blets\b"
                      r"|\bmaybe take\b|\btry\b")
# idioms that hold a unit but no duration: "wait a minute", "one day you'll", "quite a day"
_IDIOMS = re.compile(r"\b(?:an? (?:sec|second|minute|moment)\b|one day\b(?! (?:straight|in a row|ago|off))|"
                     r"(?:quite|what|such|of) an? (?:day|week)\b|(?:have|had|having) an? \w+ (?:day|week)\b|"
                     r"(?:the|this|that) (?:day|week)\b)")


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        pass
    if tok in ("a", "an"):
        return 1.0
    if tok.startswith("half"):
        return 0.5
    if "couple" in tok:
        return 2.0
    parts = re.split(r"[- ]", tok)
    return float(sum(_TENS.get(w, 0) + _TEENS.get(w, 0) + _ONES.get(w, 0) for w in parts))


def _durations(s: str) -> set[float]:
    """Every duration stated, in seconds."""
    s = _IDIOMS.sub(" ", s)
    out = set()
    for m in _DURATION.finditer(s):
        unit = m.group(3)[:-1] if m.group(3).endswith("s") else m.group(3)
        half = 0.5 if m.group(2) else 0.0
        n = _num(m.group(1))
        out.add(round((n + half) * _UNIT_S[unit] * (1.5 if m.group(4) and n == 1 else 1.0)))
    for m in _COMPACT.finditer(s):
        unit = m.group(2)[:-1] if m.group(2).endswith("s") else m.group(2)
        out.add(round(float(m.group(1)) * _UNIT_S[unit]))
    for m in re.finditer(r"\b(half an hour|an hour and a half)\b", s):
        out.add(1800 if m.group(1).startswith("half") else 5400)
    out.discard(0)
    return out


def _hour(tok: str) -> int:
    return int(tok) if tok.isdigit() else _HOUR_WORD[tok]


def _times(s: str) -> list[tuple[int, int | None, str | None]]:
    """(hour on a 12-hour dial, minute or None, 'am'/'pm' or None)."""
    out = []
    for m in _QUARTER.finditer(s):
        h = _hour(m.group(3)) % 12
        mins = {"quarter": 15, "half": 30, "ten": 10, "twenty": 20, "five": 5}[m.group(1)]
        if m.group(2) == "to":
            h, mins = (h - 1) % 12, 60 - mins
        out.append((h, mins, None))
    s = _TIME_IDIOMS.sub(" ", _QUARTER.sub(" ", s))
    for m in _CLOCK.finditer(s):
        if m.group(1):
            h, mi, ap = _hour(m.group(1)), m.group(2), m.group(3)[0] + "m"
        elif m.group(4):
            h, mi, ap = int(m.group(4)), m.group(5), None
        elif m.group(6):
            h, mi, ap = _hour(m.group(6)), None, None
        else:
            h, mi, ap = 12, "00", "pm" if m.group(7) == "noon" else "am"
        if 12 < h <= 23 and ap is None:
            h, ap = h - 12, "pm"
        if 0 <= h <= 12:
            out.append((h % 12, int(mi) if mi is not None else None, ap))
    return out


def _when_words(s: str) -> set[str]:
    words = set()
    for w in _WHEN_WORDS.findall(s):
        w = {"daily": "day", "everyday": "day", "every day": "day", "most days": "day", "nightly": "night",
             "weekly": "week", "every week": "week", "tonight": "night"}.get(w, w)
        words.add(w[:-1] if w.endswith("s") else w)
    return words


def _time_offered(t, offered: list) -> bool:
    h, mi, ap = t
    for oh, omi, oap in offered:
        if oh != h or (ap and oap and ap != oap):
            continue
        if mi is None or mi == (omi or 0):
            return True
    return False


def _clauses(s: str) -> list[str]:
    return [c for c in re.split(r"[,;]|:(?!\d)|\s(?:but|so)\s|\sand\s(?!a half)|\s[-–—]\s", s) if c.strip()]


def claim_guard(sentence: str, offered: str, user_text: str | None) -> bool:
    """True keeps the sentence. It drops only a sentence about the user that holds a clock time, a duration, or a
    frequency word with a time-of-day or day word, when that value matches nothing offered. A suggestion clause ("you
    could take a 5-minute break") is not a claim and is skipped."""
    s = _NOT_ABOUT.sub(" ", _norm(sentence))
    if not _YOU.search(s):
        return True
    claim = " , ".join(c for c in _clauses(s) if not _SUGGEST.search(c))
    if not claim.strip():
        return True
    src = _norm(offered) + " \n " + _norm(user_text or "")
    times, durs = _times(claim), _durations(claim)
    if times:
        off_times = _times(src)
        if not all(_time_offered(t, off_times) for t in times):
            return False
    if durs and not durs <= _durations(src):
        return False
    if _FREQ.search(claim):
        when = _when_words(claim)
        if when and not when <= _when_words(src):
            return False
    return True


# --- routine questions ---------------------------------------------------------------------------------------------------

_AT_IT = (r"(at it|on here|here|online|on (this|the|my) (computer|pc|laptop|screen|desk)|at (the|my) "
          r"(computer|pc|laptop|desk)|coding|programming|working|going|sitting here|staring at (this|the|my) screen|"
          r"on (this|the|my) (computer|pc|laptop))")
_AT_IT_END = r"(?=\s*(?:$|[?.!,~]|for\b|today\b|now\b|straight\b|already\b|lol\b|so far\b|riya\b|again\b|exactly\b))"
_STRETCH = re.compile(r"\b(how long (have|has) i been|how long i('ve| have)? been|how long ive been|since when have i "
                      r"been|how long (was|am) i|how many hours (have )?i('ve)? been) " + _AT_IT + _AT_IT_END)
_AGENT = (r"(?:agent|agents|claude|colab|riya|codex|run|runs)"
          r"(?: (?:agent|agents|run|runs|job|jobs|task|tasks|session|sessions))?")
_RUNS = re.compile(r"\bhow long (do|does|did) (my |a |an |the |your )?(\w+ )?" + _AGENT +
                   r"( usually| normally| typically| tend to)? (take|last|run)\b"
                   r"|\bhow long (are|were) my (\w+ )?(runs|jobs|sessions)\b"
                   r"|\bwhat'?s? (the |my )?(usual|average|typical) (\w+ )?(run|job|session)( time| length)?\b")
_RUN_LEFT = re.compile(r"\bwhen (will|is|'ll|does) (my |the |that |this )?(\w+ )?" + _AGENT +
                       r" (be )?(done|finish|finished|end|complete|over|ready)\b"
                       r"|\bhow long (until|till|til|before) (my |the |that |this )?(\w+ )?" + _AGENT + r"\b"
                       r"|\bhow long (is|will) (my |the |that |this )?(\w+ )?" + _AGENT + r" (going to |gonna )?take\b"
                       r"|\bis (my |the )?(\w+ )?" + _AGENT + r" (almost |nearly )?(done|finished|ready)\b")
_LONGER = re.compile(r"\bhow much longer\b(.*)$")
_LONGER_OK = set("for on till until is will it that this my the agent run claude colab job task gonna going to take be "
                 "left one riya codex session".split())
_GAMING = re.compile(r"\b(when|what time|what days?|which days?|what nights?)\b(.*\b)?(do|did) i (usually |normally |"
                     r"mostly |typically |tend to |often )?(game\b|gaming\b|play (?:video )?games\b|play(?=\s*[?!.~]*$))"
                     r"|\bwhen'?s? my (usual )?(game time|gaming time)\b")


def routine_question(text: str) -> str | None:
    """The sensed topic a question asks about, or None."""
    s = _norm(text)
    if _STRETCH.search(s):
        return "stretch"
    if _RUN_LEFT.search(s):
        return "run_left"
    m = _LONGER.search(s)
    if m and set(re.findall(r"[a-z']+", m.group(1))) <= _LONGER_OK:
        return "run_left"
    if _RUNS.search(s):
        return "runs"
    if _GAMING.search(s):
        return "gaming"
    return None


# --- goodbyes and disputes -------------------------------------------------------------------------------------------------

_LEAVING = re.compile(r"\b(bye+|bye-bye|byebye|goodbye|good bye|brb|gtg|g2g|gotta go|got to go|gotta run|"
                      r"i (?:have|need|got) to go|i've got to go|see (?:you|ya|u)(?!['a-z])|cya|heading out|head out|"
                      r"(?<!a )good ?night(?!'s| sleep)|gn|night night|nighty night|logging off|signing off|ttyl|"
                      r"talk (to you )?later)\b"
                      r"|^(?:(?:ok|okay|well|alright|anyway|right)[\s,!.~]*)?(?:i'm |im |i am )?off to\b")
_NOT_LEAVING = re.compile(r"\b(say|said|tell|told|wave|waved) (bye|goodbye|good night|see you)\b")
_ASKING = re.compile(r"^(do|does|did|can|could|should|would|will|what|how|why|when|is|are|am|where|who)\b")
_BARE_BYE = re.compile(r"\b(bye+|bye-bye|byebye|goodbye|brb|gtg|g2g|cya|ttyl)\b")


def goodbye(text: str) -> bool:
    """A short leaving turn: at most 8 words with a leaving phrase (a question only with an outright bye)."""
    s = _norm(text)
    words = re.findall(r"[a-z0-9']+", s)
    if not (0 < len(words) <= 8) or not _LEAVING.search(s) or _NOT_LEAVING.search(s):
        return False
    return not _ASKING.match(s) or bool(_BARE_BYE.search(s))


_LEAD = r"^(?:(?:no+|nope|nah|um+|uh+|hey|riya|wait|actually|hmm+|lol)[\s,.!~]*)*"
_END = r"(?=\s*(?:$|[,.!?~;]|lol\b|haha\b|riya\b|though\b|tho\b|at all\b|i was\b|i wasn't\b|i didn't\b|i was not\b))"
_I_NOT = r"i (?:wasn't|wasnt|was not|didn't|didnt|did not|haven't|havent|have not)\b"
_DISPUTE = re.compile(
    _LEAD + r"(?:(?:that's|thats|that is|that was|it's|its|it is|this is) (?:so |totally |completely |just )?"
            r"(?:wrong|not (?:true|right|correct|what happened)|incorrect|false|untrue)" + _END +
            r"|not true" + _END +
            r"|(?:you're|youre|you are|ur) (?:wrong|mistaken)" + _END +
            r"|wrong" + _END +
            r"|nope(?=\s*[.!~?]*$)"
            r"|" + _I_NOT + r"(?=\s*(?:$|[,.!?~]|coding|gaming|playing|working|gone|away|at it|on (?:here|it|the)|"
            r"there|doing (?:that|it)|that long|even|really|actually))"
            r")"
    r"|^(?:no+|nope|nah)[\s,.!~]+" + _I_NOT)


def disputes(text: str) -> bool:
    """The user saying her remark was wrong, as the turn's main intent."""
    return bool(_DISPUTE.search(_norm(text)))
