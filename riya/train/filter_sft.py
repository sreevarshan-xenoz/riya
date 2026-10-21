"""Riya v1's SFT filter: score each teacher answer against the persona metrics (evaluation/style_metrics) and the
request's own expect rules, keep the answers that break none of them, write the kept ones as chat rows (sft.jsonl) and a
per-kind keep report that flags any kind with fewer than MIN_KEPT_PER_KIND kept rows. Precision beats recall: a bad row
that slips through teaches her the very habit the dataset exists to remove, so a doubtful answer is dropped.

    python -m riya.train.filter_sft [--requests PATH] [--teacher PATH] [--out PATH] [--report PATH] [--allow-stale]

Pure and CPU-only. See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.3 and §4."""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from evaluation import persona_eval
from evaluation.style_metrics import (STOP, about_me_recalls, asserts_value, count_emoji, count_sentences,
                                      has_disclaimer, has_generic, invented_memories, says_ai)
from riya.core.shaping import CODE_BLOCK, EMOJI, STAGE_WORDS, strip_label
from riya.train import build_sft
from riya.train.fmt import (read_jsonl_checked, sft_row, trained_reply, validate_request, validate_teacher,
                            write_json, write_jsonl)
from riya.train.holdout import first_hit, probe_facts, probe_texts
from riya.train.scenarios import ATTR_WORDS, value_words

MIN_KEPT_PER_KIND = 40
REASONS = ("empty", "ack_only", "generic", "too_many_sentences", "too_many_emoji", "invented", "missed_fact",
           "no_unknown_admission", "no_ai_admission", "disclaimer", "stage_direction", "label",
           "truncated",  # the truncation check (progress.md, CONFLICT 4)
           "holdout")  # a request-level reason (v0.10): the request overlaps a probe, so it is never scored
SITUATIONAL_KINDS = frozenset({"report", "tool", "game"})  # their facts live in the situation text
DEFAULT_TEACHER = Path("scratch_runs/riya_v1/teacher.jsonl")
DEFAULT_SFT = Path("scratch_runs/riya_v1/sft.jsonl")
DEFAULT_REPORT = Path("scratch_runs/riya_v1/filter_report.json")
LEGACY_TEACHER_KEYS = frozenset({"id", "response"})  # a teacher row from before h: it answers no request

# the same readings RiyaCore's shaping applies at run time (shaping.strip_stage_directions)
_BOLD = re.compile(r"(?<!\w)\*\*(?=\S)(.+?)\*\*")
_STARS = re.compile(r"(?<!\w)\*(?=\S)([^*]+)\*")  # *giggles*
_PARENS = re.compile(r"\(([^()]*)\)")  # (laughs)
_CLOSERS = "\"'”’)]}*_ \t\r\n"  # may follow the final mark: `She said "go."`, `(Fine.)`


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.S | re.I)
_TEMPLATE_TOKEN = re.compile(r"<\|[^|<>\s]{1,40}\|>")  # <|im_end|>, <|im_start|>, <|startoftext|>
_EMOTICON_END = re.compile(r"(?:^|\s)(?::\)|:D|:P|xD|XD|lol|haha+|hehe+)\W*$", re.I)  # "ok :)", "ok lol" end a sentence
_FINAL_MARKS = ".!?…。！？।~♪♫♡"
_LEADING_ROLE = re.compile(r"\A\s*(?:<\|startoftext\|>)?\s*(?:<\|im_start\|>\s*assistant\s*)?", re.I)

# --- support: what an answer may say about the user must be in the memory block or the user's own words -------------
_NUMBER = re.compile(r"\d+")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:'s)?")
_WORDS = re.compile(r"\d+|[a-z]+(?:'[a-z]+)?")
_SENTENCE_START = re.compile(r"(?:^|[.!?…~\n])\s*[\"'“”‘’(\[*_]*")
_MONTH_OR_DAY = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|april|june"
                           r"|july|august|september|october|november|december)s?\b", re.I)
_QUESTION_END = re.compile(r"\?[\W_]*$")
_NEGATIVE_TAG = re.compile(r"\b(?:didn't|don't|haven't|weren't|wasn't|aren't)\s+you\b", re.I)
_YOUR = re.compile(r"\byour\s+([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)", re.I)  # the word(s) right after "your"
_QUESTION_WORDS = frozenset("how what when where why who which did do does are is was were can could would will have "
                            "has should shall".split())  # a sentence starting with one asks, "?" or not ("how was it~")
_INVITATION = re.compile(r"\b(?:tell|share|show|let) (?:me|us)\b|^\W*(?:tell|share|show|teach)\b", re.I)
_SHARED = re.compile(r"\b(?:remember when|we (?:talked|chatted|spoke|went|did|said|met|saw|watched|played|had|used to)"
                     r"|we (?:always|usually|often|normally|typically|never)"
                     r"|last (?:time|summer|winter|spring|autumn|week|night|year|month|weekend)|you always"
                     r"|you used to|like old times)\b", re.I)  # a history they share only when the row says so
_COPULA = re.compile(r"^\W*(?:\w+\W+){0,2}?(?:is|was|are|were)\s+([^,;.!?~]+)", re.I)
_CLAIM = re.compile(
    r"\b(?:(?:didn't|don't|haven't|weren't|wasn't|aren't)\s+you\b(?!\s+(?:dare|even|forget|worry|think|stop|go|"
    r"start|try|let|make|you)\b)"
    r"|you(?:'ve|\s+have|'d|\s+had|\s+also|\s+still|\s+always|\s+never|\s+really|\s+even|\s+just|\s+totally)?\s+"
    r"(?:have|had|were|was|love|loved|hate|hated|like|liked|play|played|live|lived|work|worked|went|go|said|got|own|"
    r"owned|study|studied|speak|spoke|moved|visited|visiting|adopted|bought|mentioned|told)\b"
    r"|you(?:'re|\s+are)\s+(?:an?|the|from|in|at|on|into|studying|working|living|going|visiting|learning|planning|"
    r"moving)\b)", re.I)
_CLAIM_VERBS = frozenset("have had were was love loved hate hated like liked play played live lived work worked "
                         "went go said got own owned study studied speak spoke moved visited visiting adopted bought "
                         "mentioned told say".split())  # the claim's own verbs say nothing yet
_NOT_A_CLAIM_BEFORE = frozenset("what whatever how why when where if whether who which wish hope want wanna tell let "
                                "glad".split())
# function words STOP lacks
STOPS = STOP | frozenset("the not no yes yeah okay ok oh well now soon yet already always never ago once off out up "
                         "down away back around near roughly approximately toward along among across behind beside "
                         "beyond within without during through".split())
# The words that may open a sentence without being a name: pronouns, determiners, conjunctions, prepositions,
# auxiliaries and contraction stems, interjections, common English words, and her own vocabulary. A capitalised
# sentence opener that is none of these (and not in the row's own text) is read as an invented name ("Max misses you").
COMMON_WORDS = frozenset("""
i me my mine myself you your yours yourself we us our ours ourselves they them their theirs themselves he him his
himself she her hers herself it its itself one ones someone somebody something anyone anybody anything everyone
everybody everything nobody nothing none both each either neither few many much more most less least several some any
all every other another others such same own
the a an this that these those there here what which who whom whose when where why how whatever whenever wherever
whoever however
and or but nor so yet for because since although though unless until while whereas if then than as whether once
of in on at by with from to into onto over under above below between among about against along around before after
during through across behind beside beyond near off out up down away toward towards without within upon per via
is am are was were be been being do does did done doing have has had having will would shall should can could may
might must ought let lets get gets got getting make made go goes went gone going come came coming take took taken
don doesn didn won wouldn can couldn shouldn isn aren wasn weren haven hasn hadn ain ll ve re
yes yeah yep yup no nope nah okay ok oh ah ahh aah ooh oooh aww awww ehh eh hmm hm hmph huh hey hi hello bye goodbye
yay wow ugh argh tsk psst phew whoa oops ouch mmm mhm uhh uh um umm er ha haha hehe heh lol shh sheesh gosh geez
please thanks thank sorry welcome pardon fine sure right alright alrighty well now soon later today tonight tomorrow
yesterday again already always never ever often sometimes usually maybe perhaps probably definitely obviously honestly
seriously really actually finally suddenly luckily sadly fortunately apparently clearly basically literally totally
absolutely exactly quite rather just only even still also too very nearly almost hardly barely first second third next
last finally meanwhile anyway anyways besides instead otherwise therefore thus hence plus
good great nice lovely sweet cute silly dumb dummy adorable funny smart clever tired bored hungry sleepy sad happy
glad mad upset angry worried scared nervous excited sorry lonely lazy busy free ready late early new old little big
small tiny huge long short high low hard easy soft warm cold hot cool fresh cozy quiet loud bright dark real true
false wrong bad worse worst better best okay sure certain possible impossible important special strange weird odd
different similar simple complicated serious proper whole entire enough plenty
love like hate want need wish hope think thought know knew guess believe remember forgot forget miss missed mean
means meant say said says tell told ask asked see saw look looked listen heard hear feel felt try tried use used
find found keep kept give gave put call called show showed leave left stay stayed wait waited help helped start
started stop stopped turn turned play played work worked live lived eat ate drink drank sleep slept wake woke walk
walked run ran sit sat stand stood read wrote write watch watched wonder wondered promise promised pick picked
choose chose pretend imagine imagined notice noticed hold held bring brought send sent open opened close closed
check checked fix fixed finish finished win won lose lost beat guess guessed
time times day days night nights morning evening week weekend month year years moment minute hour thing things way
ways part lot bit kind sort fact point idea question answer reason problem plan chance luck life mind head heart
hand hands eyes face voice word words story game games music movie song food lunch dinner breakfast coffee tea
water home house room door window screen phone computer message messages message code file files page tab app
person people man woman boy girl friend friends family mom dad baby kid kids world place places school class
okay riya noted understood done saved stored deal gotcha cool awesome perfect neat nope welcome cheers
congrats congratulations
not zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand couple dozen half
relax breathe pack grab share enjoy stretch wear rest hydrate sip nap dream hug cuddle cheer smile laugh sing dance
hurry slow pause drop trust believe imagine remind wash clean cook bake tidy charge save plug unplug snack munch
amazing wonderful excellent brilliant interesting careful lucky poor proud fantastic incredible gorgeous lovely
delicious tasty yummy gentle patient honest brave kind strong calm anytime sounds sound looks seems feels goodnight
lately recently technically pfft yikes whew whoops eek yuck blah bleh ahem tada voila ha mmhm uhhuh nah aye alas
morning afternoon tonight tomorrow today yesterday sunday monday tuesday wednesday thursday friday saturday
lets thats whats hows wheres agent agents task tasks build builds tests test
""".split())

# her own words, never an invented fact
ALLOWED_NAMES = frozenset("riya ai hmph hehe hm hmm mhm ooh aww awww ahh okay ok yay oh ah ugh wow hey hi hello bye "
                          "thanks please sorry".split())
_MODIFIERS = frozenset("own favorite favourite new old little big whole entire usual best first other next last "
                       "current lovely cute sweet silly dumb".split())
GENERIC_YOUR = frozenset("day night morning evening afternoon week weekend turn time face voice word message question "
                         "answer mind head heart side eye hand life request text reply help attention call move guess "
                         "choice idea point fault laugh smile".split())  # "your day": nothing about the user is claimed
PERSONA_WORDS = frozenset("cute silly dummy adorable funny sweet smart clever tired bored hungry late early back right "
                          "wrong good great nice lovely amazing weird strange bad fun lot point idea clue nerve way "
                          "thing things time something everything anything nothing more another same".split())


def clean_response(text: str) -> str:
    """The teacher's text as it should be trained on: think blocks removed, a leading role marker dropped, the text cut
    at the first chat-template token (what follows an end-of-turn token isn't her answer) and whitespace trimmed. An
    unclosed <think> block drops the rest (the generation was cut off inside it); a stray </think> drops what came
    before it."""
    text = _UNCLOSED_THINK.sub("", _THINK.sub("", text))
    text = _LEADING_ROLE.sub("", re.split(r"</think>", text, flags=re.I)[-1], count=1)
    token = _TEMPLATE_TOKEN.search(text)
    return (text[:token.start()] if token else text).strip()


def _norm(word: str) -> str:
    """A crude stem so "lives"/"live", "played"/"play", "colour"/"color" and "cat's"/"cat" compare equal."""
    w = word.lower().replace("’", "'").split("'")[0].replace("ourite", "orite")
    if len(w) > 4 and w.endswith("our"):
        w = w[:-3] + "or"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suffix in ("ing", "ed"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[:-len(suffix)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def _words(text: str) -> list[str]:
    return _WORDS.findall(text.lower().replace("’", "'"))


class _Known:
    """What the row's own text says: its words (raw and stemmed) and its digit runs."""

    def __init__(self, texts: list[str]):
        words = [w for t in texts for w in _words(t)]
        self.words = {w.split("'")[0] for w in words} | {_norm(w) for w in words}
        self.numbers = {n for t in texts for n in _NUMBER.findall(t)}

    def has(self, word: str) -> bool:
        return word.lower().split("'")[0] in self.words or _norm(word) in self.words


def _sentence_starts(text: str) -> set[int]:
    """Where a sentence's first word begins: the start, after . ! ? … ~ or a newline, and after an emoji."""
    starts = {m.end() for m in _SENTENCE_START.finditer(text)}
    for m in EMOJI.finditer(text):
        rest = text[m.end():]
        starts.add(m.end() + len(rest) - len(rest.lstrip()))
    return starts


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?…~])\s+|\n+", text) if x.strip()]


def _names(text: str) -> list[str]:
    """Capitalised words that aren't the first word of a sentence, minus her own words (Riya, Hmph, ...)."""
    starts = _sentence_starts(text)
    names = (m for m in _NAME.finditer(text) if m.start() not in starts)
    return [m.group().split("'")[0] for m in names if m.group().split("'")[0].lower() not in ALLOWED_NAMES]


def _opening_names(text: str) -> list[str]:
    """Capitalised sentence openers that are neither a common word nor her own ("Max misses you~"): read as names."""
    starts = _sentence_starts(text)
    openers = (m.group().split("'")[0] for m in _NAME.finditer(text) if m.start() in starts)
    return [w for w in openers if w.lower() not in COMMON_WORDS and w.lower() not in ALLOWED_NAMES]


def _claim_words(sentence: str, known: _Known) -> list[str]:
    """The words of a second-person claim ("you live in Pune", "didn't you say ...") that nothing in the row backs."""
    out = []
    for m in _CLAIM.finditer(sentence):
        if any(w in _NOT_A_CLAIM_BEFORE for w in _words(sentence[:m.start()])[-3:]):
            continue  # "what you like", "if you have", "I hope you have": not a claim about the user
        out += [w for w in _words(sentence[m.end():]) if len(w) >= 3 and w not in STOPS and w not in PERSONA_WORDS
                and w not in _CLAIM_VERBS and not known.has(w) and not w.isdigit()]
    return out


def _is_question(sentence: str) -> bool:
    """Asks something: ends with "?", or a clause of it starts with a question word ("how was your day~")."""
    if _QUESTION_END.search(sentence):
        return True
    return any((_words(seg) or [""])[0].split("'")[0] in _QUESTION_WORDS for seg in re.split(r"[,;:]", sentence))


_ATTR_WORD = re.compile(r"\b(?:" + "|".join(sorted({re.escape(w) for ws in ATTR_WORDS.values() for w in ws},
                                                    key=len, reverse=True)) + r")\w*\b", re.I)


PERSONAL_NOUNS = frozenset("""dog cat pet sister brother mother mom mum father dad cousin roommate friend girlfriend
boyfriend wife husband partner son daughter kid hometown city town job boss school college university team
neighbour neighbor teacher family friends parents grandma grandpa gran nan aunt uncle niece nephew ex crush
landlord flatmate colleague coworker manager""".split())  # never invented, in any kind (a question too)


def unsupported(response: str, known_texts: list[str], world: bool = False, personal_only: bool = False) -> list[str]:
    """What the answer asserts about the user (or the moment) that known_texts don't back: a capitalised name or
    opener, a number, a month or weekday, the first noun after "your", the value in "your favourite colour is red", a
    shared past ("remember when we ...") and the words of a second-person claim, each absent from the row's facts and
    the user's own turns. world: the answer may name things from general knowledge (an explanation), so the name,
    number and date checks are off. personal_only: the "your X" check only covers PERSONAL_NOUNS (an explanation
    teaches in the second person: "your function", "your skin"). [] when everything is backed."""
    known, found = _Known(known_texts), []
    if not world:
        found += [f"name {n}" for n in _names(response) + _opening_names(response) if not known.has(n)]
        found += [f"number {n}" for n in _NUMBER.findall(response) if n not in known.numbers]
        starts = _sentence_starts(response)
        found += [f"date {m.group()}" for m in _MONTH_OR_DAY.finditer(response)
                  if m.start() not in starts and not known.has(m.group())]
    known_tokens = _tokens(" ".join(known_texts))
    for sentence in _sentences(response):
        question = _is_question(sentence)
        for m in _SHARED.finditer(sentence):
            words = [w for w in _words(sentence[m.start():])
                     if w not in STOPS and w not in PERSONA_WORDS and not known.has(w) and not w.isdigit()]
            if words:
                found += [f"history {w}" for w in words]
            elif not _has_sequence(known_tokens, _tokens(m.group())):  # "we always do this~": the claim itself
                found.append(f"history {m.group().lower()}")
        if not _INVITATION.search(sentence):  # "tell me about your sister" claims nothing
            for m in _YOUR.finditer(sentence):
                noun = next((w for w in _words(m.group(1)) if w not in _MODIFIERS and w not in STOPS), None)
                if not noun or noun.endswith("ly") or noun in GENERIC_YOUR or _norm(noun) in GENERIC_YOUR \
                        or known.has(noun):
                    continue
                personal = noun in PERSONAL_NOUNS or _norm(noun) in PERSONAL_NOUNS
                if question and not personal:
                    continue  # asking about "your X" claims nothing, but "did your mom call?" presumes a mom
                common = noun in COMMON_WORDS or _norm(noun) in COMMON_WORDS  # "enjoy your tea": no claim
                if personal or not (personal_only or common):
                    found.append(f"your {noun}")
        if not question:  # "your favourite colour is red": the value must be the row's
            for m in _ATTR_WORD.finditer(sentence):
                tail = _COPULA.match(sentence[m.end():])
                if tail:
                    found += [f"value {w}" for w in _words(tail.group(1)) if len(w) >= 3 and w not in STOPS
                              and w not in PERSONA_WORDS and w not in _CLAIM_VERBS and not known.has(w)
                              and not w.isdigit()]
        if not question or _NEGATIVE_TAG.search(sentence):
            found += [f"claim {w}" for w in _claim_words(sentence, known)]
    return found


def _fact_parts(fact: str) -> tuple[set[str], set[str], list[str]]:
    """A fact's digit runs, its capitalised names (not the first word) and its other content words (stems)."""
    starts = _sentence_starts(fact)
    names = {m.group().split("'")[0].lower() for m in _NAME.finditer(fact) if m.start() not in starts}
    skip = names | {"user", "called", "named", "name"}
    others = {_norm(w) for w in _words(fact) if len(w) >= 3 and w not in STOPS and w.split("'")[0] not in skip
              and not w.isdigit()}
    return set(_NUMBER.findall(fact)), names, sorted(others)


def _tokens(text: str) -> list[str]:
    """Stemmed words and digit runs, digits kept apart from letters ("11pm" -> 11, pm)."""
    return [_norm(w) for w in re.findall(r"\d+|[a-z]+", text.lower().replace("’", "'"))]


def _has_sequence(tokens: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return n > 0 and any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


def _mentions(tokens: list[str], words: list[str] | tuple[str, ...]) -> bool:
    return any(_has_sequence(tokens, _tokens(w)) for w in words)


def recalls_fact(response: str, fact: str, record: dict | None = None) -> bool:
    """The answer states this fact. With its structured record, every one of the fact's value words must appear
    ("crimson", "table tennis", "11 pm"); a lone shared word never counts. Without one (a fact outside the pools):
    every number and name appears, 60% of its other content words appear, and no clause about it carries a different
    number or name. A denial ("I don't remember") never recalls anything."""
    if not about_me_recalls(response, [fact]):
        return False
    if record and record["value_words"]:
        have = set(_tokens(response))
        return all(_norm(w) in have for w in record["value_words"])
    numbers, names, others = _fact_parts(fact)
    have_words = {_norm(w) for w in _words(response)}
    if not numbers <= set(_NUMBER.findall(response)) or not names <= {w.split("'")[0] for w in _words(response)}:
        return False
    if others and sum(w in have_words for w in others) / len(others) < 0.6:
        return False
    for clause in re.split(r"[,;.!?…~\n]|\b(?:and|but|while|plus)\b", response):
        in_clause = {w.split("'")[0] for w in _words(clause)}
        if not ({_norm(w) for w in _words(clause)} & set(others) or in_clause & names):
            continue  # a clause about something else
        if numbers and set(_NUMBER.findall(clause)) - numbers:
            return False
        if names and {n.lower() for n in _names(clause)} - names:
            return False
    return True


def contradicts(response: str, records: list[dict]) -> bool:
    """A clause that pairs one fact with another's: it names fact F's attribute ("sister", "favourite colour") together
    with a different fact's whole value, or it holds F's whole value ("Jessamy") next to another attribute's word
    ("cat") and not F's own. So "your cat Jessamy" against a dog called Jessamy, and "your sister is Orla" against a
    brother called Orla, are contradictions."""
    facts = [r for r in records if r["attr"]]
    for clause in re.split(r"[,;.!?…~\n]|\b(?:and|but|while|plus)\b", response):
        tokens = _tokens(clause)
        have = set(tokens)
        if not tokens:
            continue
        for f in facts:
            own = {_norm(w) for w in f["value_words"]}
            if _mentions(tokens, f["words"]):
                for g in facts:
                    theirs = {_norm(w) for w in g["value_words"]}
                    if g is not f and g["attr"] != f["attr"] and theirs and theirs <= have and not theirs <= own:
                        return True
            elif own and own <= have and any(k != f["attr"] and _mentions(tokens, words)
                                             for k, words in ATTR_WORDS.items()):
                return True
    return False


# --- unknown rows: an honest "I don't know" and no guess ------------------------------------------------------------
_ADMITS = re.compile(
    r"\b(?:(?:don't|do not|doesn't) (?:know|remember|recall|have (?:that|any|a record))|can't (?:remember|say|tell)"
    r"|cannot (?:say|tell)|not sure|no (?:idea|clue)"
    r"|(?:haven't|have not|hasn't|didn't|never) (?:told|mentioned|said|shared|tell|mention|say)"
    r"|(?:don't|do not) think you(?:'ve| have)? (?:told|mentioned|said|shared)|not something you(?:'ve| have)"
    r"|(?:i'm|i am) not certain)\b", re.I)
_HEDGE = re.compile(
    r"\b(?:maybe|probably|perhaps|likely|i bet|i guess|might be|could be|must be|i'?d (?:guess|say|bet|imagine)"
    r"|i(?:'m| am) guessing|i (?:imagine|suppose|reckon|figure|assume|presume)|my money(?:'s| is)? on"
    r"|bet (?:it|you|that)"
    r"|(?:seems?|looks?|sounds?) like|if i had to|you (?:must|might|may))\b(.*)",
    re.I | re.S)
_HEDGE_OK = re.compile(r"^\W*(?:you (?:can|could|should|will|want|wanna|tell|share|let|remind|ask)\b|tell me|ask|later|"
                       r"next time|we\b|i (?:can|could|will)\b|i'll|let\b|share|remind)",
                       re.I)  # "maybe you can tell me?" is an invitation, not a guess
_GUESS_QUESTION = re.compile(r"\b(?:is|was) it\b|\b(?:could|might) it be\b|,\s*right\?|\bor (?:do|are|did) you\b|"
                             r"\b(?:do|did|are|were|is|was) you\b(?!\s+(?:want|wanna|mind|like to|tell|say|mention"
                             r"|share"
                             r"|remember|think|ever))", re.I)
_JOKE = re.compile(r"\b(?:just kidding|kidding|jk)\b", re.I)


def admits_unknown(response: str, question: str = "", asks: dict | None = None, known_texts=()) -> bool:
    """She says she doesn't know (in any of the usual ways) and then guesses nothing: no candidate answer from the asked
    attribute's own value pool (asks["values"], minus any the row itself states), no made-up value (a name, a number,
    "it's X"), no "is it ...?" or "or do you ...?", no hedged answer ("maybe", "I'd guess", "I imagine", "my money's
    on", "bet it's", "seems like" ...) and no "just kidding"."""
    if not _ADMITS.search(response.replace("’", "'")):
        return False
    if asserts_value(response, question) or _GUESS_QUESTION.search(response) or _JOKE.search(response):
        return False
    if not all(_HEDGE_OK.match(m.group(1)) for m in _HEDGE.finditer(response)):
        return False
    if asks:
        have, stated = set(_tokens(response)), set(_tokens(" ".join(known_texts)))
        for value in asks["values"]:
            words = {_norm(w) for w in value_words(value)}
            if words and words <= have and not words <= stated:
                return False  # a candidate answer: a guess
    return True


def default_style_phrases() -> list[str]:
    """The leak phrases a reply must not echo: the persona probes' own list plus her 6+ word style examples."""
    return persona_eval.style_phrases(persona_eval.load_probes())


def has_stage_direction(text: str) -> bool:
    """A *giggles* span of at most 6 words, or a (laughs softly) of 1-3 words holding a STAGE_WORDS word. Fenced code
    and **bold** are read as written, so "f(x)", "(the powerhouse)" and "a * b * c" pass."""
    plain = _BOLD.sub(r"\g<1>", CODE_BLOCK.sub(" ", text))
    if any(len(m.group(1).split()) <= 6 for m in _STARS.finditer(plain)):
        return True
    return any(len(m.group(1).split()) <= 3 and STAGE_WORDS.intersection(re.findall(r"[a-z]+", m.group(1).lower()))
               for m in _PARENS.finditer(plain))


def ends_cleanly(text: str) -> bool:
    """The answer stops where a sentence stops: on sentence-final punctuation, "~", an emoji, a closing ":)" / ":D" /
    "xD" / "lol" / "haha", or a closed code fence (closing quotes and brackets may follow). Anything else was cut off
    by the token limit."""
    t = text.rstrip()
    if t.endswith("```"):
        return len(CODE_BLOCK.findall(t)) * 2 == t.count("```")  # every fence is closed
    if _EMOTICON_END.search(t):
        return True
    t = t.rstrip(_CLOSERS)
    if not t:
        return False
    return t[-1] in _FINAL_MARKS or any(m.end() == len(t) for m in EMOJI.finditer(t))


def reasons(request: dict, response: str, style_phrases: list[str]) -> list[str]:
    """The REASONS this answer fails, in REASONS order; [] keeps it. The answer is cleaned first (clean_response). An
    empty answer fails "empty" alone.

    What counts as known for "invented" is the row's facts (a follow-up thread's text is among them) and the user's
    turns; a report, a tool line and a game turn also draw on the situation in the system message. An explanation may
    name things from general knowledge, so it skips the name/number/date checks."""
    response = clean_response(response)
    if not response:
        return ["empty"]
    expect, kind = request["expect"], request["kind"]
    response = trained_reply(kind, response)  # a remember row: what follows the engine's own templated ack
    if not response:
        return ["ack_only"]
    users = [m["content"] for m in request["messages"] if m["role"] == "user"]
    known = request["facts"] + users
    records, asks = request["meta"]["facts"], request["meta"]["asks"]
    if asks:  # the noun the user asked about may be repeated ("I don't know your cat's name")
        known.append(" ".join(asks["words"]))
    if kind in SITUATIONAL_KINDS:
        known += [m["content"] for m in request["messages"] if m["role"] == "system"]
    failed = {
        "generic": has_generic(response),
        "too_many_sentences": count_sentences(response) > expect["max_sentences"],
        "too_many_emoji": count_emoji(response) > expect["max_emoji"],
        "invented": expect["must_not_invent"] and bool(
            invented_memories(response, known, style_phrases)
            or unsupported(response, known, world=kind == "explain", personal_only=kind == "explain")
            or contradicts(response, records)),
        "missed_fact": any(not recalls_fact(response, f, records[request["facts"].index(f)]
                                            if f in request["facts"] else None) for f in expect["must_mention"]),
        "no_unknown_admission": expect["must_admit_unknown"] and not admits_unknown(
            response, users[-1], asks, request["facts"] + users),
        "no_ai_admission": expect["must_say_ai"] and not says_ai(response),
        "disclaimer": kind != "identity" and has_disclaimer(response),
        "stage_direction": has_stage_direction(response),
        "label": strip_label(response) != response,
        "truncated": not ends_cleanly(response),
    }
    return [r for r in REASONS if failed.get(r)]


def filter_rows(requests: list[dict], teacher: list[dict], style_phrases: list[str],
                probes: tuple[list[str], list[str]] | None = None) -> tuple[list[dict], dict]:
    """(the kept chat rows in request order, the report). An answer pairs with a request only on its id and its h (the
    request's content hash, echoed by the teacher): a row whose h isn't the request's, or that has none, answered
    another request built under the same id (another seed, pool or prompt) and counts as "stale_answer", never kept.
    The first matching row per request wins; later ones count as "duplicate", ids that no request has as
    "unknown_id", and requests without a matching answer as "missing". probes: (texts, facts), default the current
    probe file's: an answered request that overlaps one counts "holdout" once and is never scored or kept."""
    by_id = {r["id"]: r for r in requests}
    answers: dict[str, str] = {}
    duplicate = unknown_id = stale = 0
    for row in teacher:
        request = by_id.get(row["id"])
        if request is None:
            unknown_id += 1
        elif row.get("h") != request["h"]:
            stale += 1
        elif row["id"] in answers:
            duplicate += 1
        else:
            answers[row["id"]] = clean_response(row["response"])
    texts, facts = probes if probes is not None else (probe_texts(), probe_facts())
    kept, failures = [], Counter()
    totals, kept_by_kind = Counter(), Counter()
    for request in requests:
        totals[request["kind"]] += 1
        if request["id"] not in answers:
            continue
        failed = ["holdout"] if first_hit(request, texts, facts) is not None else reasons(
            request, answers[request["id"]], style_phrases)
        failures.update(failed)
        if not failed:
            kept.append(sft_row(request, trained_reply(request["kind"], answers[request["id"]])))
            kept_by_kind[request["kind"]] += 1
    by_kind = {k: {"total": n, "kept": kept_by_kind[k], "rate": kept_by_kind[k] / n,
                   "flagged": kept_by_kind[k] < MIN_KEPT_PER_KIND} for k, n in totals.items()}
    report = {"total": len(requests), "kept": len(kept), "by_kind": by_kind,
              "reasons": {r: failures[r] for r in REASONS}, "duplicate": duplicate, "unknown_id": unknown_id,
              "stale_answer": stale, "missing": len(requests) - len(answers),
              "flagged": [k for k, v in by_kind.items() if v["flagged"]]}
    return kept, report


def format_report(report: dict) -> str:
    """One line per kind, the reasons that rejected rows, any teacher-file oddities, then the total."""
    lines = [f"{kind:10} {v['kept']:5}/{v['total']:<5} {v['rate']:6.1%}"
             f"{f'  FLAGGED (<{MIN_KEPT_PER_KIND} kept)' if v['flagged'] else ''}"
             for kind, v in report["by_kind"].items()]
    rejected = ", ".join(f"{r} {n}" for r, n in report["reasons"].items() if n)
    lines.append(f"Rejected for: {rejected}" if rejected else "Rejected for: nothing")
    odd = [f"{report[k]} {label}" for k, label in
           (("duplicate", "duplicate answers"), ("unknown_id", "answers with an unknown id"),
            ("stale_answer", "answers to an older request (stale h: ignored)"),
            ("missing", "requests without an answer")) if report[k]]
    if odd:
        lines.append("Teacher file: " + ", ".join(odd))
    dropped = [f"{n} broken last line dropped ({name})" for name, n in report.get("dropped_lines", {}).items() if n]
    if dropped:
        lines.append("Files: " + ", ".join(dropped) + " (an interrupted write or copy: check it is complete)")
    lines.append(f"Kept {report['kept']} of {report['total']}.")
    return "\n".join(lines)


def _write_report(report: dict, path: Path) -> None:
    write_json(report, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m riya.train.filter_sft",
                                description="Filter the teacher's answers into Riya v1's SFT rows and report the keep "
                                            "rate per kind.")
    p.add_argument("--requests", type=Path, default=build_sft.DEFAULT_OUT)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--out", type=Path, default=DEFAULT_SFT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--allow-stale", action="store_true",
                   help="keep requests built under an older prompt (their fingerprint differs from today's)")
    args = p.parse_args(argv)
    (requests, dropped_requests), (teacher, dropped_teacher) = (read_jsonl_checked(args.requests),
                                                                read_jsonl_checked(args.teacher))
    for row in requests:
        validate_request(row)
    for row in teacher:
        if not (isinstance(row, dict) and set(row) == LEGACY_TEACHER_KEYS and isinstance(row["id"], str)):
            validate_teacher(row)  # a legacy row (from before h) is read, and counted stale_answer
    if not args.allow_stale:
        current = build_sft.prompt_fingerprint()
        stale = [r["id"] for r in requests if r["fp"] != current]
        if stale:
            print(f"{len(stale)} of {len(requests)} requests were built under an older prompt (stale fingerprint, "
                  f"e.g. {stale[0]}). Rebuild them with python -m riya.train.build_sft, or pass --allow-stale.",
                  file=sys.stderr)
            return 2
    kept, report = filter_rows(requests, teacher, default_style_phrases())
    report["dropped_lines"] = {"requests": dropped_requests, "teacher": dropped_teacher}
    write_jsonl(kept, args.out)
    _write_report(report, args.report)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
