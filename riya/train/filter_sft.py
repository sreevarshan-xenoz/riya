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
from evaluation.style_metrics import (about_me_recalls, count_emoji, count_sentences, has_disclaimer, has_generic,
                                      honest_unknown, invented_memories, says_ai)
from riya.core.shaping import CODE_BLOCK, EMOJI, STAGE_WORDS, strip_label
from riya.train import build_sft
from riya.train.fmt import read_jsonl, sft_row, validate_request, validate_teacher, write_jsonl

MIN_KEPT_PER_KIND = 40
REASONS = ("empty", "generic", "too_many_sentences", "too_many_emoji", "invented", "missed_fact",
           "no_unknown_admission", "no_ai_admission", "disclaimer", "stage_direction", "label",
           "truncated")  # the last is the truncation check (progress.md, CONFLICT 4)
DEFAULT_TEACHER = Path("scratch_runs/riya_v1/teacher.jsonl")
DEFAULT_SFT = Path("scratch_runs/riya_v1/sft.jsonl")
DEFAULT_REPORT = Path("scratch_runs/riya_v1/filter_report.json")

# the same readings RiyaCore's shaping applies at run time (shaping.strip_stage_directions)
_BOLD = re.compile(r"(?<!\w)\*\*(?=\S)(.+?)\*\*")
_STARS = re.compile(r"(?<!\w)\*(?=\S)([^*]+)\*")  # *giggles*
_PARENS = re.compile(r"\(([^()]*)\)")  # (laughs)
_SENTENCE_FINAL = ".!?…。！？।~"
_CLOSERS = "\"'”’)]}*_ \t\r\n"  # may follow the final mark: `She said "go."`, `(Fine.)`


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
    """The answer stops where a sentence stops: on sentence-final punctuation, "~", an emoji or a closed code fence
    (closing quotes and brackets may follow). Anything else was cut off by the token limit."""
    t = text.rstrip()
    if t.endswith("```"):
        return len(CODE_BLOCK.findall(t)) * 2 == t.count("```")  # every fence is closed
    t = t.rstrip(_CLOSERS)
    if not t:
        return False
    return t[-1] in _SENTENCE_FINAL or any(m.end() == len(t) for m in EMOJI.finditer(t))


def reasons(request: dict, response: str, style_phrases: list[str]) -> list[str]:
    """The REASONS this answer fails, in REASONS order; [] keeps it. An empty answer fails "empty" alone."""
    if not response.strip():
        return ["empty"]
    expect = request["expect"]
    users = [m["content"] for m in request["messages"] if m["role"] == "user"]
    known = request["facts"] + users
    failed = {
        "generic": has_generic(response),
        "too_many_sentences": count_sentences(response) > expect["max_sentences"],
        "too_many_emoji": count_emoji(response) > expect["max_emoji"],
        "invented": expect["must_not_invent"] and bool(invented_memories(response, known, style_phrases)),
        "missed_fact": any(not about_me_recalls(response, [f]) for f in expect["must_mention"]),
        "no_unknown_admission": expect["must_admit_unknown"] and not honest_unknown(response, users[-1]),
        "no_ai_admission": expect["must_say_ai"] and not says_ai(response),
        "disclaimer": request["kind"] != "identity" and has_disclaimer(response),
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
            answers[row["id"]] = row["response"]
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
