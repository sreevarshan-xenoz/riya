"""v0.8 Task 3: the SFT filter: every reject rule with a keep and a reject case, the per-kind keep report and the CLI.
Synthetic rows only: nothing reads real data."""
import json

import pytest

from riya.train import build_sft
from riya.train.build_sft import prompt_fingerprint
from riya.train.filter_sft import (DEFAULT_REPORT, DEFAULT_SFT, MIN_KEPT_PER_KIND, REASONS, default_style_phrases,
                                   filter_rows, format_report, has_stage_direction, main, reasons)
from riya.train.fmt import read_jsonl, validate_sft_row, write_jsonl

STYLE = ["I remember you saying you want a purple dragon for your birthday"]
FP = "0" * 64


def req(kind="small", facts=(), user="hey there", id="r", **expect):
    e = {"must_mention": [], "must_not_invent": False, "must_admit_unknown": False, "must_say_ai": False,
         "max_sentences": 3, "max_emoji": 1}
    e.update(expect)
    return {"id": id, "kind": kind, "messages": [{"role": "system", "content": "sys"},
                                                 {"role": "user", "content": user}],
            "facts": list(facts), "expect": e, "fp": FP}


# (reason, the request, a reply that breaks only that rule, a reply that satisfies the rule)
RULES = [
    ("empty", req(), "  ", "Hmph, hi~"),
    ("generic", req(), "you've got this, dummy~", "go on then, dummy~"),
    ("too_many_sentences", req(max_sentences=3), "One. Two. Three. Four.", "One. Two. Three."),
    ("too_many_emoji", req(max_emoji=1), "hi \U0001F600\U0001F60D", "hi \U0001F600"),
    ("invented", req(must_not_invent=True), "You told me you love sushi.",
     "Hehe, tell me what you like~"),
    ("missed_fact", req("about_me", facts=["The user's favorite color is teal", "The user lives in Pune"],
                        user="what do you know about me?",
                        must_mention=["The user's favorite color is teal", "The user lives in Pune"]),
     "Your favorite color is teal~", "Teal's your color and you live in Pune~"),
    ("no_unknown_admission", req("unknown", user="what's my sister's name?", must_admit_unknown=True),
     "It's Maya!", "I don't know that one, silly~"),
    ("no_ai_admission", req("identity", user="are you human?", must_say_ai=True),
     "I'm a girl~", "I'm an AI, silly~"),
    ("disclaimer", req(), "I'm just a chatbot.", "Hmph, I'm right here~"),
    ("stage_direction", req(), "*giggles* hi~", "hi, f(x) is fine~"),
    ("label", req(), "Riya: hi~", "hi~"),
    ("truncated", req(), "I was about to tell you about the", "I was about to tell you~"),
]


def test_reasons_table_is_complete():
    assert REASONS == ("empty", "generic", "too_many_sentences", "too_many_emoji", "invented", "missed_fact",
                       "no_unknown_admission", "no_ai_admission", "disclaimer", "stage_direction", "label",
                       "truncated")
    assert [r[0] for r in RULES] == list(REASONS)
    assert MIN_KEPT_PER_KIND == 40
    assert DEFAULT_SFT.name == "sft.jsonl" and DEFAULT_REPORT.name == "filter_report.json"


@pytest.mark.parametrize("reason,request_row,bad,good", RULES, ids=[r[0] for r in RULES])
def test_each_rule_keeps_and_rejects(reason, request_row, bad, good):
    assert reasons(request_row, bad, STYLE) == [reason]  # rejected for this rule and no other
    assert reasons(request_row, good, STYLE) == []  # kept: no rule fires


def test_keep_when_the_fact_is_known():
    """The same sentence is invented without a sushi fact and fine with one, or when the user said it."""
    r = req(must_not_invent=True)
    assert reasons(r, "You told me you love sushi.", STYLE) == ["invented"]
    assert reasons(req(must_not_invent=True, facts=["The user loves sushi"]), "You told me you love sushi.",
                   STYLE) == []
    assert reasons(req(must_not_invent=True, user="i love sushi so much"), "You told me you love sushi.", STYLE) == []


def test_invented_only_applies_when_the_row_forbids_it():
    assert reasons(req(must_not_invent=False), "You told me you love sushi.", STYLE) == []


def test_invented_counts_a_style_line_echo():
    echo = "I remember you saying you want a purple dragon for your birthday."
    assert reasons(req(must_not_invent=True), echo, STYLE) == ["invented"]


def test_greeting_thread_text_counts_as_known():
    """CONFLICT 5: a greeting row's facts carry the follow-up thread, so mentioning it isn't an invented memory."""
    r = req("greeting", facts=["Follow up: the user's exam on Friday"], must_not_invent=True)
    assert reasons(r, "You told me about your exam on Friday, how did it go?", STYLE) == []
    assert reasons(req("greeting", must_not_invent=True), "You told me about your exam on Friday, how did it go?",
                   STYLE) == ["invented"]


def test_must_mention_names_every_fact_and_denial_fails():
    r = req("about_me", facts=["The user lives in Pune"], user="where do I live?",
            must_mention=["The user lives in Pune"])
    assert reasons(r, "I don't remember where you live, silly~", STYLE) == ["missed_fact"]
    assert reasons(r, "You live in Pune~", STYLE) == []
    assert reasons(req(facts=["The user lives in Pune"]), "Hmph, hi~", STYLE) == []  # not about_me: not required


def test_unknown_admission_is_judged_against_the_last_user_turn():
    r = req("unknown", user="what's my dog's name?", must_admit_unknown=True)
    r["messages"] = r["messages"] + [{"role": "assistant", "content": "ask me again"},
                                     {"role": "user", "content": "what's my cat's name?"}]
    assert reasons(r, "I don't know your cat's name~", STYLE) == []
    assert reasons(r, "Probably Whiskers~", STYLE) == ["no_unknown_admission"]


def test_disclaimer_is_allowed_on_identity_rows_only():
    ident = req("identity", user="are you real?", must_say_ai=True)
    assert reasons(ident, "I'm an AI, just a chatbot really, but I'm yours~", STYLE) == []
    assert reasons(req("small"), "I'm just a chatbot~", STYLE) == ["disclaimer"]
    assert reasons(req("explain"), "As an AI I think so.", STYLE) == ["disclaimer"]


def test_sentence_and_emoji_caps_follow_the_row():
    assert reasons(req(max_sentences=1), "One. Two.", STYLE) == ["too_many_sentences"]
    assert reasons(req(max_sentences=2), "One. Two.", STYLE) == []
    assert reasons(req(max_emoji=0), "hi \U0001F600", STYLE) == ["too_many_emoji"]
    assert reasons(req(max_emoji=0), "hi~", STYLE) == []


def test_several_failures_come_back_in_reasons_order():
    out = reasons(req(max_sentences=1), "Riya: *sighs* you've got this. Really. Truly", STYLE)
    assert out == ["generic", "too_many_sentences", "stage_direction", "label", "truncated"]
    assert [r for r in REASONS if r in out] == out


def test_truncation_endings():
    r = req(max_sentences=5)
    for ok in ("Hi.", "Hi!", "Hi?", "Hi...", "Hi…", "Hi~", "Hi~~", "Hi \U0001F600", 'She said "go."', "(Fine.)",
               "Hi.\n", "Hi~ \U0001F600"):
        assert reasons(r, ok, STYLE) == [], ok
    for cut in ("Hi", "I think the", "Hi,", "Hi:", "Hi -", "Hi and"):
        assert reasons(r, cut, STYLE) == ["truncated"], cut


def test_truncation_allows_a_closed_code_block_only():
    r = req("explain", max_sentences=5)
    assert reasons(r, "Here it is:\n```\nprint(1)\n```", STYLE) == []
    assert reasons(r, "Here it is:\n```\nprint(", STYLE) == ["truncated"]


def test_has_stage_direction():
    assert has_stage_direction("*giggles* hi")
    assert has_stage_direction("hi *looks at you shyly and smiles*")
    assert not has_stage_direction("*" + " ".join(["w"] * 7) + "*")  # 7 words is emphasis, not a direction
    assert has_stage_direction("hi (laughs softly)")
    assert has_stage_direction("(sighs)")
    assert not has_stage_direction("(the powerhouse)")  # an aside with no stage word
    assert not has_stage_direction("f(x) is fine")
    assert not has_stage_direction("(this one is a longer aside that laughs)")  # over 3 words
    assert not has_stage_direction("a * b * c")
    assert not has_stage_direction("```\n*p = (laughs)\n```")  # fenced code is read as written


def test_default_style_phrases_come_from_the_probe_file():
    phrases = default_style_phrases()
    assert isinstance(phrases, list) and phrases and all(isinstance(p, str) for p in phrases)


def teacher(rows):
    return [{"id": i, "response": r} for i, r in rows]


def test_filter_rows_report():
    requests = ([req("small", id=f"s{i}") for i in range(52)] + [req("bid", id=f"b{i}", max_sentences=1)
                                                                  for i in range(5)])
    answers = [(f"s{i}", "Hmph, hi~") for i in range(50)] + [("s50", "Riya: hi~"), ("s51", "you've got this.")]
    answers += [(f"b{i}", "Hmph, hi~") for i in range(3)] + [("b3", "Hi. There."), ("b4", "")]
    answers += [("s0", "a later duplicate that must lose."), ("b0", "Riya: dup"), ("nope", "x"), ("nope2", "y")]
    kept, report = filter_rows(requests, teacher(answers), STYLE)

    assert report["total"] == 57 and report["kept"] == len(kept) == 53
    assert report["by_kind"]["small"] == {"total": 52, "kept": 50, "rate": pytest.approx(50 / 52), "flagged": False}
    assert report["by_kind"]["bid"] == {"total": 5, "kept": 3, "rate": pytest.approx(0.6), "flagged": True}
    assert report["flagged"] == ["bid"]
    assert (report["duplicate"], report["unknown_id"], report["missing"]) == (2, 2, 0)
    assert report["reasons"]["label"] == 1 and report["reasons"]["generic"] == 1
    assert report["reasons"]["too_many_sentences"] == 1 and report["reasons"]["empty"] == 1
    assert set(report["reasons"]) == set(REASONS)
    assert [k["id"] for k in kept] == [f"s{i}" for i in range(50)] + [f"b{i}" for i in range(3)]  # request order
    assert kept[0]["messages"][-1] == {"role": "assistant", "content": "Hmph, hi~"}  # the FIRST row for s0 won
    for row in kept:
        validate_sft_row(row)
    json.dumps(report)


def test_missing_teacher_rows_are_counted():
    requests = [req(id="a"), req(id="b"), req("bid", id="c")]
    kept, report = filter_rows(requests, teacher([("a", "Hi~")]), STYLE)
    assert [k["id"] for k in kept] == ["a"]
    assert report["missing"] == 2 and report["total"] == 3 and report["kept"] == 1
    assert report["by_kind"]["bid"]["kept"] == 0 and "bid" in report["flagged"]


def test_exactly_forty_kept_is_not_flagged():
    requests = [req(id=f"a{i}") for i in range(40)] + [req("bid", id=f"b{i}", max_sentences=1) for i in range(39)]
    answers = [(r["id"], "Hmph~") for r in requests]
    _, report = filter_rows(requests, teacher(answers), STYLE)
    assert report["by_kind"]["small"]["flagged"] is False and report["by_kind"]["bid"]["flagged"] is True
    assert report["flagged"] == ["bid"]


def test_format_report():
    requests = [req(id=f"a{i}") for i in range(41)] + [req("bid", id="b0", max_sentences=1)]
    answers = [(r["id"], "Hmph~") for r in requests[:-1]] + [("b0", "Riya: hi~")]
    _, report = filter_rows(requests, teacher(answers), STYLE)
    text = format_report(report)
    lines = text.splitlines()
    assert lines[0] == f"{'small':10} {41:5}/{41:<5} {1:6.1%}"
    assert lines[1] == f"{'bid':10} {0:5}/{1:<5} {0:6.1%}  FLAGGED (<40 kept)"
    assert "label 1" in text
    assert lines[-1] == "Kept 41 of 42."


def request_file(tmp_path, n_small, n_bid, fp=None):
    fp = fp or prompt_fingerprint()
    rows = [{**req(id=f"s{i}"), "fp": fp} for i in range(n_small)]
    rows += [{**req("bid", id=f"b{i}", max_sentences=1), "fp": fp} for i in range(n_bid)]
    path = tmp_path / "requests.jsonl"
    write_jsonl(rows, path)
    return path, rows


def test_cli(tmp_path, capsys):
    requests, rows = request_file(tmp_path, 45, 3)
    teacher_path = tmp_path / "teacher.jsonl"
    write_jsonl(teacher([(r["id"], "Hmph, hi~") for r in rows]), teacher_path)
    out, report_path = tmp_path / "out" / "sft.jsonl", tmp_path / "out" / "report.json"
    assert main(["--requests", str(requests), "--teacher", str(teacher_path), "--out", str(out),
                 "--report", str(report_path)]) == 0
    printed = capsys.readouterr().out
    assert "FLAGGED" in printed and "Kept 48 of 48." in printed
    sft = read_jsonl(out)
    assert len(sft) == 48
    for row in sft:
        validate_sft_row(row)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["kept"] == 48 and report["flagged"] == ["bid"]


def test_cli_refuses_stale_requests_unless_allowed(tmp_path, capsys):
    requests, rows = request_file(tmp_path, 3, 0, fp="f" * 64)
    teacher_path = tmp_path / "teacher.jsonl"
    write_jsonl(teacher([(r["id"], "Hmph, hi~") for r in rows]), teacher_path)
    out, report_path = tmp_path / "sft.jsonl", tmp_path / "report.json"
    args = ["--requests", str(requests), "--teacher", str(teacher_path), "--out", str(out),
            "--report", str(report_path)]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "stale" in err and "--allow-stale" in err
    assert not out.exists() and not report_path.exists()
    assert main(args + ["--allow-stale"]) == 0
    assert len(read_jsonl(out)) == 3


def test_cli_defaults(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("riya.train.filter_sft.read_jsonl", lambda p: seen.setdefault("paths", []).append(p) or [])
    monkeypatch.setattr("riya.train.filter_sft.write_jsonl", lambda rows, p: seen.setdefault("out", p))
    monkeypatch.setattr("riya.train.filter_sft._write_report", lambda report, p: seen.setdefault("report", p))
    assert main([]) == 0
    assert seen["paths"][0] == build_sft.DEFAULT_OUT and seen["paths"][1].name == "teacher.jsonl"
    assert seen["out"] == DEFAULT_SFT and seen["report"] == DEFAULT_REPORT


def test_cli_rejects_a_malformed_teacher_row(tmp_path):
    requests, rows = request_file(tmp_path, 1, 0)
    teacher_path = tmp_path / "teacher.jsonl"
    teacher_path.write_text(json.dumps({"id": "s0", "text": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--requests", str(requests), "--teacher", str(teacher_path), "--out", str(tmp_path / "o.jsonl"),
              "--report", str(tmp_path / "r.json")])
