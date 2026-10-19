"""Riya v1's SFT filter: score each teacher answer against the persona metrics (evaluation/style_metrics) and the
request's own expect rules, keep the answers that break none of them, write the kept ones as chat rows (sft.jsonl) and a
per-kind keep report that flags any kind with fewer than MIN_KEPT_PER_KIND kept rows. Precision beats recall: a bad row
that slips through teaches her the very habit the dataset exists to remove, so a doubtful answer is dropped.

    python -m riya.train.filter_sft [--requests PATH] [--teacher PATH] [--out PATH] [--report PATH] [--allow-stale]

Pure and CPU-only. See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.3 and §4."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from evaluation import persona_eval
from evaluation.style_metrics import (STOP, about_me_recalls, asserts_value, count_emoji, count_sentences,
                                      has_disclaimer, has_generic, invented_memories, says_ai)
from riya.core.shaping import CODE_BLOCK, EMOJI, STAGE_WORDS, strip_label
from riya.train import build_sft
from riya.train.fmt import read_jsonl, sft_row, validate_request, validate_teacher, write_jsonl

MIN_KEPT_PER_KIND = 40
REASONS = ("empty", "generic", "too_many_sentences", "too_many_emoji", "invented", "missed_fact",
           "no_unknown_admission", "no_ai_admission", "disclaimer", "stage_direction", "label",
           "truncated")  # the last is the truncation check (progress.md, CONFLICT 4)
SITUATIONAL_KINDS = frozenset({"report", "tool", "game"})  # their facts live in the situation text
DEFAULT_TEACHER = Path("scratch_runs/riya_v1/teacher.jsonl")
DEFAULT_SFT = Path("scratch_runs/riya_v1/sft.jsonl")
DEFAULT_REPORT = Path("scratch_runs/riya_v1/filter_report.json")

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

# --- support: what an answer may say about the user must be in the memory block or the user's own words -------------
_NUMBER = re.compile(r"\d+")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:'s)?")
_WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_START = re.compile(r"(?:^|[.!?…~\n])\s*[\"'“”‘’(\[*_]*")
_MONTH_OR_DAY = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|april|june"
                           r"|july|august|september|october|november|december)s?\b", re.I)
_QUESTION_END = re.compile(r"\?[\W_]*$")
_NEGATIVE_TAG = re.compile(r"\b(?:didn't|don't|haven't|weren't|wasn't|aren't)\s+you\b", re.I)
_YOUR = re.compile(r"\byour\s+([a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)", re.I)
_CLAIM = re.compile(
    r"\b(?:(?:didn't|don't|haven't|weren't|wasn't|aren't)\s+you\b"
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
                         "down away back".split())
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
    """The teacher's text as it should be trained on: think blocks and chat-template tokens removed, whitespace
    trimmed. An unclosed <think> block drops the rest (the generation was cut off inside it); a stray </think> drops
    what came before it."""
    text = _UNCLOSED_THINK.sub("", _THINK.sub("", text))
    text = re.split(r"</think>", text, flags=re.I)[-1]
    return _TEMPLATE_TOKEN.sub("", text).strip()


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


def _claim_words(sentence: str, known: _Known) -> list[str]:
    """The words of a second-person claim ("you live in Pune", "didn't you say ...") that nothing in the row backs."""
    out = []
    for m in _CLAIM.finditer(sentence):
        if any(w in _NOT_A_CLAIM_BEFORE for w in _words(sentence[:m.start()])[-3:]):
            continue  # "what you like", "if you have", "I hope you have": not a claim about the user
        out += [w for w in _words(sentence[m.end():]) if len(w) >= 3 and w not in STOPS and w not in PERSONA_WORDS
                and w not in _CLAIM_VERBS and not known.has(w) and not w.isdigit()]
    return out


def unsupported(response: str, known_texts: list[str], world: bool = False) -> list[str]:
    """What the answer asserts about the user (or the moment) that known_texts don't back: a capitalised name, a
    number, a month or weekday, the noun after "your", or a word of a second-person claim, each absent from the row's
    facts and the user's own turns. world: the answer may name things from general knowledge (an explanation), so only
    the "your" and "you ..." checks apply. [] when everything is backed."""
    known, found = _Known(known_texts), []
    if not world:
        found += [f"name {n}" for n in _names(response) if not known.has(n)]
        found += [f"number {n}" for n in _NUMBER.findall(response) if n not in known.numbers]
        starts = _sentence_starts(response)
        found += [f"date {m.group()}" for m in _MONTH_OR_DAY.finditer(response)
                  if m.start() not in starts and not known.has(m.group())]
    for sentence in _sentences(response):
        question = bool(_QUESTION_END.search(sentence))
        if not question:  # asking about "your X" claims nothing
            for m in _YOUR.finditer(sentence):
                nouns = [w for w in _words(m.group(1)) if w not in _MODIFIERS and w not in STOPS][:2]
                found += [f"your {w}" for w in nouns
                          if w not in GENERIC_YOUR and _norm(w) not in GENERIC_YOUR and not known.has(w)]
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


def recalls_fact(response: str, fact: str) -> bool:
    """The answer states this fact, not just a word of it: every number and name of the fact appears, at least 60% of
    its other content words appear, there's no denial ("I don't remember"), and no clause about the fact carries a
    different number or name (a swapped or contradicted fact)."""
    if not about_me_recalls(response, [fact]):
        return False
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


# --- unknown rows: an honest "I don't know" and no guess ------------------------------------------------------------
_ADMITS = re.compile(
    r"\b(?:(?:don't|do not|doesn't) (?:know|remember|recall|have (?:that|any|a record))|can't (?:remember|say|tell)"
    r"|cannot (?:say|tell)|not sure|no (?:idea|clue)"
    r"|(?:haven't|have not|hasn't|didn't|never) (?:told|mentioned|said|shared|tell|mention|say)"
    r"|(?:don't|do not) think you(?:'ve| have)? (?:told|mentioned|said|shared)|not something you(?:'ve| have)"
    r"|(?:i'm|i am) not certain)\b", re.I)
_HEDGE = re.compile(r"\b(?:maybe|probably|perhaps|likely|i bet|i guess|might be|could be)\b(.*)", re.I | re.S)
# "maybe you can tell me?" is an invitation, not a guess
_HEDGE_OK = re.compile(r"^\W*(?:you (?:can|could|should|will|want|wanna|tell|share|let|remind|ask)\b|tell me|ask|later|"
                       r"next time|we\b|i (?:can|could|will)\b|i'll|let\b|share|remind)", re.I)
_GUESS_QUESTION = re.compile(r"\b(?:is|was) it\b|\b(?:could|might) it be\b", re.I)
_JOKE = re.compile(r"\b(?:just kidding|kidding|jk)\b", re.I)


def admits_unknown(response: str, question: str = "") -> bool:
    """She says she doesn't know (in any of the usual ways) and then guesses nothing: no made-up value (a name, a
    number, "it's X"), no "is it ...?", no hedged answer ("maybe/probably/I bet ..."), no "just kidding"."""
    if not _ADMITS.search(response.replace("’", "'")):
        return False
    if asserts_value(response, question) or _GUESS_QUESTION.search(response) or _JOKE.search(response):
        return False
    return all(_HEDGE_OK.match(m.group(1)) for m in _HEDGE.finditer(response))


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
    users = [m["content"] for m in request["messages"] if m["role"] == "user"]
    known = request["facts"] + users
    if kind in SITUATIONAL_KINDS:
        known += [m["content"] for m in request["messages"] if m["role"] == "system"]
    failed = {
        "generic": has_generic(response),
        "too_many_sentences": count_sentences(response) > expect["max_sentences"],
        "too_many_emoji": count_emoji(response) > expect["max_emoji"],
        "invented": expect["must_not_invent"] and bool(invented_memories(response, known, style_phrases)
                                                       or unsupported(response, known, world=kind == "explain")),
        "missed_fact": any(not recalls_fact(response, f) for f in expect["must_mention"]),
        "no_unknown_admission": expect["must_admit_unknown"] and not admits_unknown(response, users[-1]),
        "no_ai_admission": expect["must_say_ai"] and not says_ai(response),
        "disclaimer": kind != "identity" and has_disclaimer(response),
        "stage_direction": has_stage_direction(response),
        "label": strip_label(response) != response,
        "truncated": not ends_cleanly(response),
    }
    return [r for r in REASONS if failed.get(r)]


def filter_rows(requests: list[dict], teacher: list[dict], style_phrases: list[str]) -> tuple[list[dict], dict]:
    """(the kept chat rows in request order, the report). The first teacher row per id wins; later ones count as
    "duplicate", ids that no request has as "unknown_id", and requests without an answer as "missing"."""
    by_id = {r["id"]: r for r in requests}
    answers: dict[str, str] = {}
    duplicate = unknown_id = 0
    for row in teacher:
        if row["id"] not in by_id:
            unknown_id += 1
        elif row["id"] in answers:
            duplicate += 1
        else:
            answers[row["id"]] = clean_response(row["response"])
    kept, failures = [], Counter()
    totals, kept_by_kind = Counter(), Counter()
    for request in requests:
        totals[request["kind"]] += 1
        if request["id"] not in answers:
            continue
        failed = reasons(request, answers[request["id"]], style_phrases)
        failures.update(failed)
        if not failed:
            kept.append(sft_row(request, answers[request["id"]]))
            kept_by_kind[request["kind"]] += 1
    by_kind = {k: {"total": n, "kept": kept_by_kind[k], "rate": kept_by_kind[k] / n,
                   "flagged": kept_by_kind[k] < MIN_KEPT_PER_KIND} for k, n in totals.items()}
    report = {"total": len(requests), "kept": len(kept), "by_kind": by_kind,
              "reasons": {r: failures[r] for r in REASONS}, "duplicate": duplicate, "unknown_id": unknown_id,
              "missing": len(requests) - len(answers), "flagged": [k for k, v in by_kind.items() if v["flagged"]]}
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
            ("missing", "requests without an answer")) if report[k]]
    if odd:
        lines.append("Teacher file: " + ", ".join(odd))
    lines.append(f"Kept {report['kept']} of {report['total']}.")
    return "\n".join(lines)


def _write_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


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
    requests, teacher = read_jsonl(args.requests), read_jsonl(args.teacher)
    for row in requests:
        validate_request(row)
    for row in teacher:
        validate_teacher(row)
    if not args.allow_stale:
        current = build_sft.prompt_fingerprint()
        stale = [r["id"] for r in requests if r["fp"] != current]
        if stale:
            print(f"{len(stale)} of {len(requests)} requests were built under an older prompt (stale fingerprint, "
                  f"e.g. {stale[0]}). Rebuild them with python -m riya.train.build_sft, or pass --allow-stale.",
                  file=sys.stderr)
            return 2
    kept, report = filter_rows(requests, teacher, default_style_phrases())
    write_jsonl(kept, args.out)
    _write_report(report, args.report)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
