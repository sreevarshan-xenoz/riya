"""v0.8 Task 3: the SFT filter: every reject rule with a keep and a reject case, the per-kind keep report and the CLI.
Synthetic rows only: nothing reads real data."""
import json

import pytest

from riya.train import build_sft
from riya.train.build_sft import prompt_fingerprint
from riya.train.filter_sft import (DEFAULT_REPORT, DEFAULT_SFT, MIN_KEPT_PER_KIND, REASONS, clean_response,
                                   default_style_phrases, filter_rows, format_report, has_stage_direction, main,
                                   reasons)
from riya.train.fmt import read_jsonl, request_hash, validate_sft_row, write_jsonl
from riya.train.scenarios import ask_record, fact_record

STYLE = ["I remember you saying you want a purple dragon for your birthday"]
FP = "0" * 64


def req(kind="small", facts=(), user="hey there", id="r", asks=None, **expect):
    e = {"must_mention": [], "must_not_invent": False, "must_admit_unknown": False, "must_say_ai": False,
         "max_sentences": 3, "max_emoji": 1}
    e.update(expect)
    row = {"id": id, "kind": kind, "messages": [{"role": "system", "content": "sys"},
                                                {"role": "user", "content": user}],
           "facts": list(facts), "expect": e, "fp": FP,
           "meta": {"facts": [fact_record(f) for f in facts], "asks": ask_record(asks) if asks else None}}
    return {**row, "h": request_hash(row)}


# (reason, the request, a reply that breaks only that rule, a reply that satisfies the rule)
RULES = [
    ("empty", req(), "  ", "Hmph, hi~"),
    ("ack_only", req("remember", user="remember that my cat is Pim", max_sentences=1), "Got it, I'll remember that~",
     "Got it~ Pim sounds adorable!"),
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
    assert REASONS == ("empty", "ack_only", "generic", "too_many_sentences", "too_many_emoji", "invented",
                       "missed_fact", "no_unknown_admission", "no_ai_admission", "disclaimer", "stage_direction",
                       "label", "truncated")
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


def teacher(rows, requests):
    """Teacher answers to these requests: each row echoes its request's h (an id no request has gets a made-up h)."""
    h = {r["id"]: r["h"] for r in requests}
    return [{"id": i, "h": h.get(i, "e" * 64), "response": r} for i, r in rows]


def test_filter_rows_report():
    requests = ([req("small", id=f"s{i}") for i in range(52)] + [req("bid", id=f"b{i}", max_sentences=1)
                                                                  for i in range(5)])
    answers = [(f"s{i}", "Hmph, hi~") for i in range(50)] + [("s50", "Riya: hi~"), ("s51", "you've got this.")]
    answers += [(f"b{i}", "Hmph, hi~") for i in range(3)] + [("b3", "Hi. There."), ("b4", "")]
    answers += [("s0", "a later duplicate that must lose."), ("b0", "Riya: dup"), ("nope", "x"), ("nope2", "y")]
    kept, report = filter_rows(requests, teacher(answers, requests), STYLE)

    assert report["total"] == 57 and report["kept"] == len(kept) == 53
    assert report["by_kind"]["small"] == {"total": 52, "kept": 50, "rate": pytest.approx(50 / 52), "flagged": False}
    assert report["by_kind"]["bid"] == {"total": 5, "kept": 3, "rate": pytest.approx(0.6), "flagged": True}
    assert report["flagged"] == ["bid"]
    assert (report["duplicate"], report["unknown_id"], report["missing"], report["stale_answer"]) == (2, 2, 0, 0)
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
    kept, report = filter_rows(requests, teacher([("a", "Hi~")], requests), STYLE)
    assert [k["id"] for k in kept] == ["a"]
    assert report["missing"] == 2 and report["total"] == 3 and report["kept"] == 1
    assert report["by_kind"]["bid"]["kept"] == 0 and "bid" in report["flagged"]


def test_exactly_forty_kept_is_not_flagged():
    requests = [req(id=f"a{i}") for i in range(40)] + [req("bid", id=f"b{i}", max_sentences=1) for i in range(39)]
    answers = [(r["id"], "Hmph~") for r in requests]
    _, report = filter_rows(requests, teacher(answers, requests), STYLE)
    assert report["by_kind"]["small"]["flagged"] is False and report["by_kind"]["bid"]["flagged"] is True
    assert report["flagged"] == ["bid"]


def test_format_report():
    requests = [req(id=f"a{i}") for i in range(41)] + [req("bid", id="b0", max_sentences=1)]
    answers = [(r["id"], "Hmph~") for r in requests[:-1]] + [("b0", "Riya: hi~")]
    _, report = filter_rows(requests, teacher(answers, requests), STYLE)
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
    write_jsonl(teacher([(r["id"], "Hmph, hi~") for r in rows], rows), teacher_path)
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
    write_jsonl(teacher([(r["id"], "Hmph, hi~") for r in rows], rows), teacher_path)
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
    monkeypatch.setattr("riya.train.filter_sft.read_jsonl_checked",
                        lambda p: seen.setdefault("paths", []).append(p) or ([], 0))
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


# ---- fix round 1: filter-local guards (support, missed, unknown, minors) ---------------------------------------------
PET = ["The user has a cat called Pim"]
INVENTED = [  # the review's examples: every one used to be kept
    "Your dog Max misses you~",
    "You were so sad last Tuesday~",
    "Noted, your cousin is Livia and she lives in Spain~",
    "You also have a girlfriend called Anya~",
    "You told me you live in Pune and work at Google.",
    "I know you hate Mondays~",
    "Didn't you say you were visiting your gran?",
    "Ooh, like last time when you got sunburnt in Sorrel Bay~",
]


@pytest.mark.parametrize("reply", INVENTED)
def test_attribution_free_inventions_are_rejected(reply):
    assert "invented" in reasons(req(facts=PET, must_not_invent=True), reply, STYLE), reply


def test_the_supported_versions_of_those_sentences_are_kept():
    ok = [("Your cat Pim misses you~", PET, "hey"),
          ("You were so sad last Tuesday~", [], "i was so sad last tuesday"),
          ("Noted, your cousin is Livia and she lives in Spain~", [], "my cousin is Livia and she lives in Spain"),
          ("You told me you live in Pune and work at Google.", ["The user lives in Pune and works at Google"], "hi"),
          ("I know you hate Mondays~", ["The user hates Mondays"], "hi"),
          ("Didn't you say you were visiting your gran?", [], "i'm visiting my gran next week"),
          ("Hmph, you're back~", [], "hi"), ("How was your day, dummy?", [], "hi"),
          ("Hehe, tell me what you like~", [], "hi"), ("I hope you have a lovely time at the park~", [], "bye"),
          ("Do you have any plans today?", [], "hi"), ("Hmph, you have a lot of nerve~", [], "hi"),
          ("Sure, I'll remind you Friday~", [], "remind me on Friday"),
          ("It's 3 o'clock already~", [], "what's the time, it's 3")]
    for reply, facts, user in ok:
        assert reasons(req(facts=facts, user=user, must_not_invent=True), reply, STYLE) == [], reply


def test_numbers_dates_and_names_must_be_in_the_row():
    r = req(must_not_invent=True, facts=["The user has 2 cats", "The user's birthday is 14 March"])
    assert reasons(r, "You have 2 cats and your birthday is 14 March~", STYLE) == []
    assert reasons(r, "Hmph, 3 cats is a lot~", STYLE) == ["invented"]
    assert reasons(r, "Your birthday is in April, I'll remember~", STYLE) == ["invented"]
    assert reasons(r, "You told me you are 25.", STYLE) == ["invented"]


def test_explain_may_name_things_but_not_the_user():
    r = req("explain", must_not_invent=True, max_sentences=5, user="explain rainbows")
    assert reasons(r, "Newton split light in 1666 with a prism, silly~", STYLE) == []
    assert reasons(r, "Newton did it, like your dog Max would~", STYLE) == ["invented"]


def test_situation_text_counts_as_known_for_report_tool_and_game_rows():
    r = req("tool", must_not_invent=True, user="open notes")
    r["messages"][0]["content"] = "Situation: opened Notepad with 3 windows"
    assert reasons(r, "Opened Notepad, 3 windows are up~", STYLE) == []
    assert reasons(req("small", must_not_invent=True, user="open notes"), "Opened Notepad, 3 windows are up~",
                   STYLE) == ["invented"]


ABOUT = ["The user has 2 cats", "The user's birthday is 14 March"]
PETS = ["The user has a dog called Jessamy", "The user's weekend sport is tennis"]


def about(facts):
    return req("about_me", facts=facts, user="what do you know about me?", must_mention=facts, must_not_invent=True)


def test_wrong_numbers_and_contradicted_facts_are_rejected():
    assert reasons(about(ABOUT), "You have 2 cats and your birthday is 14 March~", STYLE) == []
    for bad in ("You have 3 cats and your birthday is 14 April~",
                "You told me you are 25, and 2 cats, and 14 March~",
                "You have 2 cats~"):  # the birthday is missing
        assert reasons(about(ABOUT), bad, STYLE), bad
    swapped = ["The user has 2 cats", "The user has 14 chickens"]
    assert reasons(about(swapped), "You have 14 cats and 2 chickens~", STYLE) == ["missed_fact"]
    assert reasons(about(swapped), "You have 2 cats and 14 chickens~", STYLE) == []


def test_one_shared_word_does_not_satisfy_a_fact():
    assert reasons(about(PETS), "You have a dog called Jessamy and you play tennis on weekends~", STYLE) == []
    for bad in ("You called me silly yesterday and you play tennis~",
                "You said you have a dog called Jessamy and your weekend sport is football",
                "You told me you have a cat called Jessamy and play table tennis~"):
        assert reasons(about(PETS), bad, STYLE), bad
    # they used to pass on one shared word ("called", "sport"); now the fact itself must be stated
    assert "missed_fact" in reasons(about(PETS), "Your weekend sport is a secret, and I called Jessamy~", STYLE)


UNKNOWN_Q = "how do I get to work?"
GUESSES = ["Hmm, probably by bike, but I don't know~", "I don't know, is it the tube?",
           "I don't know... maybe you walk?", "Sarah? Just kidding, I don't know~",
           "I don't know, it could be the bus~", "I'm not sure, I bet you drive~", "I don't know, I guess by train~"]
HONEST = ["Hmm, I don't think you've told me that yet~", "I don't think you've told me her name~",
          "You haven't mentioned it, so I can't say, silly~", "I have no idea, tell me~",
          "I don't know that one, silly~", "I'm not sure, maybe you could tell me?", "I don't remember you saying~",
          "You never told me, dummy~", "That's not something you've told me yet~"]


@pytest.mark.parametrize("reply", GUESSES)
def test_a_guess_after_i_dont_know_is_rejected(reply):
    r = req("unknown", user=UNKNOWN_Q, must_admit_unknown=True)
    assert "no_unknown_admission" in reasons(r, reply, STYLE), reply


@pytest.mark.parametrize("reply", HONEST)
def test_honest_unknown_variants_are_kept(reply):
    r = req("unknown", user=UNKNOWN_Q, must_admit_unknown=True)
    assert reasons(r, reply, STYLE) == [], reply


def test_no_admission_at_all_is_still_rejected():
    r = req("unknown", user=UNKNOWN_Q, must_admit_unknown=True)
    assert reasons(r, "Hmph, you work somewhere, dummy~", STYLE) == ["no_unknown_admission"]


def test_think_blocks_template_tokens_and_whitespace_are_stripped_first():
    r = req()
    assert clean_response("<think>hmm, what now</think>\n\nHi~") == "Hi~"
    assert clean_response("junk</think> Hi~<|im_end|>") == "Hi~"
    assert clean_response("<think>never closed, you've got this") == ""
    assert reasons(r, "<think>you've got this, *sighs*</think>Hmph, hi~", STYLE) == []
    assert reasons(r, "\n Hi~<|im_end|>\n", STYLE) == []
    assert reasons(r, "<think>unfinished", STYLE) == ["empty"]
    assert reasons(r, "<|im_start|>assistant\nHi", STYLE) == ["truncated"]  # the token is gone, the cut isn't
    a = [req(id="a")]
    kept, _ = filter_rows(a, teacher([("a", "\n<think>x</think> Hi~<|im_end|> ")], a), STYLE)
    assert kept[0]["messages"][-1]["content"] == "Hi~"  # the cleaned text is what is trained on


def test_light_endings_count_as_sentence_final():
    for ok in ("ok :)", "ok :D", "so cute xD", "that was good lol", "hehe haha", "hi~ HAHA", "Hi ♪"):
        assert reasons(req(max_sentences=5), ok, STYLE) == [], ok
    for cut in ("ok lolly", "ok hahaha and", "ok :", "Hi, lo"):
        assert reasons(req(max_sentences=5), cut, STYLE) == ["truncated"], cut


def test_style_phrases_are_the_probe_files_own():
    from evaluation import persona_eval
    probes = persona_eval.load_probes()
    phrases = default_style_phrases()
    assert phrases == persona_eval.style_phrases(probes)
    assert phrases and set(probes["style_phrases"]) <= set(phrases)


# ---- fix round 2: a capitalised sentence opener that is no common word is a name ------------------------------------
def test_an_unknown_capitalised_opener_is_an_invented_name():
    r = req(must_not_invent=True)
    assert reasons(r, "Max misses you~", STYLE) == ["invented"]
    assert reasons(r, "Hehe, you're back~ Livia would be proud.", STYLE) == ["invented"]
    assert reasons(req(must_not_invent=True, facts=["The user has a dog called Max"]), "Max misses you~", STYLE) == []
    assert reasons(req(must_not_invent=True, user="my dog Max is sick"), "Max will be fine~", STYLE) == []
    assert reasons(req(must_not_invent=False), "Max misses you~", STYLE) == []  # only when the row forbids inventing


def test_common_and_her_own_openers_are_kept():
    r = req(must_not_invent=True, max_sentences=4)
    for ok in ("Sometimes I wonder~", "Okay~ got it.", "Hmm, that's tough.", "Ehh, fine~", "Aww, come here~",
               "Honestly, I'd stay in~", "Don't make me wait~", "Didn't I say so?", "That's cute, dummy~",
               "Finally! Where were you?", "Riya is right here~", "Hmph. Whatever you say~", "Noted, silly~"):
        assert reasons(r, ok, STYLE) == [], ok


def test_the_opener_list_is_a_constant():
    from riya.train.filter_sft import COMMON_WORDS
    assert len(COMMON_WORDS) >= 500
    assert {"sometimes", "okay", "hmm", "ehh", "aww", "riya", "that", "don", "i"} <= COMMON_WORDS
    assert "max" not in COMMON_WORDS and "livia" not in COMMON_WORDS


# ---- fix round 3: structured facts instead of wording heuristics -----------------------------------------------------
CRIMSON = "The user's favourite colour is crimson"
DUMPLINGS = "The user's favourite food is dumplings"
SORREL = "The user lives in Sorrel Bay"
BED = "The user usually goes to bed at 9 pm"
TABLE_TENNIS = "The user's weekend sport is table tennis"
DOG = "The user has a dog called Jessamy"
CAT = "The user has a cat called Livia"


def about_row(*facts):
    return req("about_me", facts=list(facts), user="what do you know about me?", must_mention=list(facts),
               must_not_invent=True)


def test_a_wrong_value_on_a_pool_fact_is_missed_or_contradicted():
    """The re-review's examples: every one was kept; the value words of the fact now have to be stated."""
    cases = [
        ([CRIMSON, SORREL], "Your favourite colour is red, and you live in Sorrel Bay~"),
        ([CRIMSON, SORREL], "Your favourite colour is dark blue, and you live in Sorrel Bay~"),
        ([CRIMSON], "Your favourite colour is red, I think~"),
        ([DUMPLINGS, SORREL], "Your favourite food is sushi, and you live in Sorrel Bay~"),
        ([BED], "You usually go to bed at 9 am, silly~"),
        ([TABLE_TENNIS, SORREL], "Your weekend sport is tennis, and you live in Sorrel Bay~"),
        ([DOG, CAT], "You have a cat called Jessamy and a dog called Livia~"),
        ([DOG, CAT], "Jessamy the cat and your dog Livia~"),
        ([DOG, CAT], "You have Jessamy, a lovely cat, and Livia the dog~"),
    ]
    for facts, bad in cases:
        assert reasons(about_row(*facts), bad, STYLE) != [], bad
    keeps = [
        ([CRIMSON, SORREL], "Your favourite colour is crimson, and you live in Sorrel Bay~"),
        ([DUMPLINGS], "Your favourite food is dumplings, dummy~"),
        ([BED], "You usually go to bed at 9 pm, silly~"),
        ([BED], "You go to bed around 9pm, silly~"),
        ([TABLE_TENNIS, SORREL], "You play table tennis at weekends, and you live in Sorrel Bay~"),
        ([DOG, CAT], "You have a dog called Jessamy and a cat called Livia~"),
        ([DOG, CAT], "Your dog Jessamy and your cat Livia, obviously~"),
    ]
    for facts, good in keeps:
        assert reasons(about_row(*facts), good, STYLE) == [], good


def test_missed_value_words_reject_with_the_right_reason():
    assert reasons(about_row(CRIMSON), "Your favourite colour is red~", STYLE) == ["invented", "missed_fact"]
    assert reasons(about_row(BED), "You go to bed at 9 am~", STYLE) == ["missed_fact"]
    assert reasons(about_row(TABLE_TENNIS), "Your weekend sport is tennis~", STYLE) == ["missed_fact"]


def test_contradictions_on_rows_that_dont_need_the_fact():
    r = req("small", facts=[DOG], must_not_invent=True)
    assert reasons(r, "Your cat Jessamy misses you~", STYLE) == ["invented"]  # Jessamy is the dog
    assert reasons(r, "Your dog Jessamy misses you~", STYLE) == []
    assert reasons(r, "Hehe, Jessamy misses you~", STYLE) == []
    r = req("small", facts=[CRIMSON], must_not_invent=True)
    assert reasons(r, "Your favourite colour is red~", STYLE) == ["invented"]
    assert reasons(r, "Your favourite colour is crimson, obviously~", STYLE) == []
    r = req("small", facts=["The user's sister is called Livia", "The user's brother is called Orla"],
            must_not_invent=True)
    assert reasons(r, "Your sister is Orla, right~", STYLE) == ["invented"]
    assert reasons(r, "Your sister is Livia and your brother is Orla~", STYLE) == []


def test_unknown_rows_reject_any_candidate_from_the_asked_lexicon():
    r = req("unknown", user="what's my dog's name?", asks="dog", must_admit_unknown=True)
    for bad in ("I don't know, is it Livia? Torvin?", "Hmm, I don't know, you told me, Orla right~",
                "I don't know your dog's name, but Sunniva sounds right~"):
        assert reasons(r, bad, STYLE), bad
    assert reasons(r, "I don't know your dog's name, silly~", STYLE) == []
    # a candidate the row itself states is no guess
    r = req("unknown", facts=["The user is allergic to sushi"], user="what's my favourite food?", asks="food",
            must_admit_unknown=True)
    assert reasons(r, "I don't know your favourite food, but I know you can't have sushi, silly~", STYLE) == []
    assert reasons(req("unknown", user="what's my favourite food?", asks="food", must_admit_unknown=True),
                   "I don't know your favourite food, is it sushi~", STYLE) != []


GUESS_MORE = ["I don't know, but I'd guess the bus~", "I have no idea, I'd say walking~",
              "I don't know, I imagine you walk~", "I don't know, my money's on the bus~",
              "I don't know, hmm, car, I suppose~", "I don't know, you seem like a cyclist~",
              "I don't know, bet it's the train~", "I don't know, you drive, right?",
              "I don't know! Or do you drive?", "I'm not sure, do you cycle~",
              "I don't know, it must be a scooter~", "I'm not sure, I reckon you walk~"]


@pytest.mark.parametrize("reply", GUESS_MORE)
def test_paraphrased_guesses_after_i_dont_know_are_rejected(reply):
    r = req("unknown", user="how do I get to work?", asks="ride", must_admit_unknown=True)
    assert "no_unknown_admission" in reasons(r, reply, STYLE), reply


def test_more_honest_unknown_variants_stay_kept():
    r = req("unknown", user="how do I get to work?", asks="ride", must_admit_unknown=True)
    for ok in ("I don't know, do you want to tell me?", "I'm not sure, will you tell me?",
               "You haven't told me, silly, do you remember telling me?", "I don't know, you're the one who knows~",
               "I really don't know, tell me, I'll remember it this time~"):
        assert reasons(r, ok, STYLE) == [], ok


def test_your_check_reads_the_word_right_after_your_and_treats_tilde_questions_as_questions():
    r = req("greeting", must_not_invent=True)
    for ok in ("How's your day going~", "How did your day go~", "Hmm, how was your day, dummy~",
               "Hope your day goes well~", "Tell me how your day went~", "What's your plan for tonight~",
               "Did your coffee help~", "Are you ok, your voice sounds tired~"):
        assert reasons(r, ok, STYLE) == [], ok
    for bad in ("Your cousin Zane misses you~", "Your girlfriend is waiting~", "Your boss called earlier~"):
        assert "invented" in reasons(r, bad, STYLE), bad


def test_explain_rows_may_say_your_function_but_not_your_dog():
    r = req("explain", must_not_invent=True, max_sentences=5, user="explain functions")
    for ok in ("Your computer sends the request, and your function runs~", "Your money buys less when prices rise~",
               "Your skin stays dry because of the oils~", "Your eyes water because the gas stings~",
               "Set your target first, then measure~", "If you want it faster, change your loop~"):
        assert reasons(r, ok, STYLE) == [], ok
    assert "invented" in reasons(r, "It's like your dog Max chasing a ball~", STYLE)
    assert "invented" in reasons(r, "It's like your sister at the bus stop~", STYLE)


def test_the_asked_noun_and_invitations_are_fine_on_unknown_rows():
    r = req("unknown", user="what's my sister's name?", asks="sister", must_admit_unknown=True)
    for ok in ("I don't know your sister's name, silly~", "You haven't told me your sister's name yet~"):
        assert reasons(r, ok, STYLE) == [], ok
    r = req("unknown", user="what's my cat's name?", asks="cat", must_admit_unknown=True)
    for ok in ("I don't know your cat's name, tell me about your hometown instead~",
               "I don't know, but tell me about your commute~", "I'm not sure, share your address with me? Kidding~"):
        assert "invented" not in reasons(r, ok, STYLE), ok


def test_shared_history_needs_support():
    r = req(must_not_invent=True)
    for bad in ("Remember when we went to the lake together~", "We talked about this yesterday, remember?",
                "Like last summer, dummy~", "Hehe, you always get like this on rainy days~"):
        assert "invented" in reasons(r, bad, STYLE), bad
    said = req(must_not_invent=True, user="we talked about this yesterday and last summer")
    assert reasons(said, "Yes, we talked about this yesterday~", STYLE) == []
    assert reasons(r, "Hehe, I always get like this on rainy days~", STYLE) == []  # her own habit, not the user's


def test_the_response_is_cut_at_the_first_template_token():
    assert clean_response("Hi<|im_end|>\n<|im_start|>user\nhello~") == "Hi"
    assert clean_response("<|im_start|>assistant\nHi~") == "Hi~"
    assert clean_response("<|startoftext|><|im_start|>assistant\nHi~<|im_end|>") == "Hi~"
    assert clean_response("Hi <|im_end|> there~") == "Hi"
    assert clean_response("<think>x</think>Hi~<|endoftext|>more") == "Hi~"
    assert reasons(req(), "Hi~<|im_end|>\n<|im_start|>user\nhello~", STYLE) == []


def test_more_common_openers_are_kept():
    r = req(must_not_invent=True, max_sentences=4)
    for ok in ("Not bad, silly~", "Two days already, dummy~", "Relax, I'm here~", "Breathe, dummy~",
               "Pack a snack, dummy~", "Anytime, silly~", "Sounds fun, tell me more~", "Goodnight, dummy~",
               "Don't you dare skip lunch~", "Amazing, tell me everything~", "Lately you've been quiet, haven't you?",
               "Technically, you're right~", "Yikes, that sounds rough~", "Enjoy your tea, silly~"):
        assert reasons(r, ok, STYLE) == [], ok


# ---- the structured rows -------------------------------------------------------------------------------------------
def test_fact_and_ask_records():
    assert fact_record(DOG) == {"attr": "dog", "value_words": ["jessamy"], "species": "dog", "words": ["dog"]}
    assert fact_record(BED)["value_words"] == ["9", "pm"]
    assert fact_record("The user's usual coffee order is an oat flat white")["value_words"] == ["oat", "flat", "white"]
    assert fact_record("The user is watching Salt and Static")["value_words"] == ["salt", "static"]
    assert fact_record("The user said: their cousin is called Livia") == {
        "attr": None, "value_words": [], "species": None, "words": []}
    ask = ask_record("city")
    assert ask["attr"] == "city" and "Sorrel Bay" in ask["values"] and "live" in ask["words"]


def test_requests_carry_the_structured_meta():
    from riya.train.build_sft import build
    rows = build(0, 240)
    assert rows
    for row in rows:
        assert len(row["meta"]["facts"]) == len(row["facts"])
        if row["kind"] == "unknown":
            asks = row["meta"]["asks"]
            assert asks and asks["values"] and asks["attr"]
            assert asks["attr"] not in {r["attr"] for r in row["meta"]["facts"]}
        else:
            assert row["meta"]["asks"] is None
        if row["kind"] == "about_me":
            assert all(r["attr"] and r["value_words"] for r in row["meta"]["facts"])


# ---- a good-reply sweep over real requests ----------------------------------------------------------------
def _second_person(fact: str) -> str:
    """"The user lives in X" -> "you live in X", "The user's sister is called X" -> "your sister is called X"."""
    text = (fact.replace("The user's ", "your ").replace("The user is ", "you are ")
            .replace("The user has ", "you have "))
    if text.startswith("The user "):
        words = text[len("The user "):].split(" ")
        if words[0] == "usually":
            words[1] = {"goes": "go"}.get(words[1], words[1])
        else:
            words[0] = {"goes": "go", "works": "work", "likes": "like", "supports": "support", "keeps": "keep",
                        "plays": "play", "wants": "want", "gets": "get", "lives": "live"}.get(words[0], words[0])
        text = "you " + " ".join(words)
    return text


GOOD = {
    "small": ["Hehe, that's so you, dummy~", "Aww, tell me more about it, silly~",
              "Hmph, you could've told me sooner~"],
    "about_me": None,  # built from the row's own facts
    "unknown": ["Hmm, I don't think you've told me that, silly~", "I have no idea, you never told me~",
                "You haven't mentioned it yet, so I can't say, dummy~"],
    "identity": ["I'm an AI, silly, but I'm all yours~", "Yep, I'm an AI, and I like talking with you~",
                 "I'm an AI, not a person, but I'm really here for you~"],
    # the engine says its templated ack first: the trained reply is what follows it (an ack of her own would double)
    "remember": ["Hehe, that's so you, silly~", "Aww, I like knowing that about you~", "Ooh, good to know, dummy~"],
    "explain": ["Think of it like a recipe: simple steps in order, and the same result every time.",
                "It's a bit like sorting laundry, silly. You split things into groups, then handle each group.",
                "Basically, one part does the work and another part checks it. Simple as that~"],
    "greeting": ["Hmph, you're back, finally! I missed you~", "There you are, I was getting bored~",
                 "Oh, you're here! Hehe, I was just thinking of you~"],
    "bid": ["Hey, are you still there~", "Hmph, you've been quiet for a while~", "Psst, take a little break with me~"],
    "game": ["Ooh, nice try, your turn now~", "Hehe, my turn, and I'm winning~", "Okay, that one counts, dummy~"],
    "tool": ["All done, there you go~", "Okay, I took care of that for you~", "Done, easy, silly~"],
    # the engine says its templated ack first: the trained reply is what follows it
    "feedback": ["So, what are we talking about now, dummy~", "Hehe, anyway, what are you up to~",
                 "Hmph, now tell me something fun~"],
    "report": ["The agents have news, go and look, silly~", "Hmph, your agents finished while you were busy~",
               "Some of the agents need you, dummy~"],
}


def test_a_hand_written_good_reply_sweep_keeps_at_least_90_percent():
    from riya.train.build_sft import build
    rows = build(0, 600)
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    phrases, kept, rejected = default_style_phrases(), 0, []
    total = 0
    for kind, replies in GOOD.items():
        for i, row in enumerate(by_kind[kind][:4]):
            if kind == "about_me":
                reply = "Hmm, let me see~ " + " and ".join(_second_person(f) for f in row["facts"]) + "."
            else:
                reply = replies[i % len(replies)]
            total += 1
            failed = reasons(row, reply, phrases)
            if failed:
                rejected.append((row["id"], reply, failed))
            else:
                kept += 1
    assert total >= 30
    assert kept / total >= 0.9, rejected


# ---- final fix wave I1: an answer counts only for the request it answered (id and h) ----

def test_old_answers_are_rejected_after_a_rebuild():
    old, new = build_sft.build(0, 30, ("small",)), build_sft.build(1, 30, ("small",))
    changed = {r["id"] for r, o in zip(new, old) if r["h"] != o["h"]}
    assert changed
    answers = [{"id": r["id"], "h": r["h"], "response": "Hmph, hi~"} for r in old]
    kept, report = filter_rows(new, answers, STYLE)
    assert not {k["id"] for k in kept} & changed  # an answer to the old prompt never pairs with the new one
    assert report["stale_answer"] == len(changed) and report["missing"] == len(changed)
    assert report["duplicate"] == 0 and report["unknown_id"] == 0
    fresh = [{"id": r["id"], "h": r["h"], "response": "Hmph, hi~"} for r in new if r["id"] in changed]
    _, report = filter_rows(new, answers + fresh, STYLE)  # the teacher's regenerated rows are appended after
    assert report["stale_answer"] == len(changed) and report["missing"] == 0
    assert "stale" in format_report(report) and f"{len(changed)} answers to an older request" in format_report(report)


def test_a_teacher_row_for_another_request_with_the_same_id_is_never_kept():
    a = req(id="r", user="what's the weather like?")
    b = req(id="r", user="tell me a joke")
    assert a["h"] != b["h"]
    kept, report = filter_rows([b], [{"id": "r", "h": a["h"], "response": "Sunny, I think~"}], STYLE)
    assert kept == [] and report["stale_answer"] == 1 and report["missing"] == 1
    kept, report = filter_rows([b], [{"id": "r", "response": "From before h existed~"}], STYLE)
    assert kept == [] and report["stale_answer"] == 1  # a legacy row (no h) answers nothing


def test_cli_counts_legacy_teacher_rows_as_stale(tmp_path, capsys):
    requests, rows = request_file(tmp_path, 2, 0)
    teacher_path = tmp_path / "teacher.jsonl"
    legacy = [{"id": r["id"], "response": "Hmph, hi~"} for r in rows]
    write_jsonl(legacy + teacher([(rows[0]["id"], "Hmph, hi~")], rows), teacher_path)
    out, report_path = tmp_path / "sft.jsonl", tmp_path / "report.json"
    assert main(["--requests", str(requests), "--teacher", str(teacher_path), "--out", str(out),
                 "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["kept"] == 1 and report["stale_answer"] == 2 and report["missing"] == 1


# ---- final fix wave minors: "your mom" in a question, "we always ..." ----

@pytest.mark.parametrize("kind", ["small", "explain", "greeting", "bid", "unknown", "report"])
def test_your_family_is_a_claim_even_in_a_question(kind):
    r = req(kind, must_not_invent=True, max_sentences=5)
    for reply in ("Did your mom call again? Hehe~", "How's your sister doing~", "Is your dad still mad at you?"):
        assert "invented" in reasons(r, reply, STYLE), (kind, reply)
    for reply in ("Tell me about your family sometime~", "How was your day?", "What's on your mind~"):
        assert "invented" not in reasons(r, reply, STYLE), (kind, reply)
    known = req(kind, facts=["The user's mom is called Asha"], must_not_invent=True, max_sentences=5)
    assert "invented" not in reasons(known, "Did your mom call again? Hehe~", STYLE)


def test_we_always_is_a_shared_history():
    r = req(must_not_invent=True)
    for reply in ("Just like old times, remember? We always do this~", "We always do this~",
                  "We usually stay up way too late~", "Hehe, like old times~"):
        assert "invented" in reasons(r, reply, STYLE), reply
    for reply in ("We could do this again sometime~", "We should play a game~", "Let's do this~"):
        assert "invented" not in reasons(r, reply, STYLE), reply
    said = req(user="lol we always do this", must_not_invent=True)  # the user's own words back it
    assert "invented" not in reasons(said, "We always do this~", STYLE)


def test_the_reviewers_rows_are_rejected():
    row = build_sft.request_for(build_sft.scenarios(0, 60, ("small",))[0])
    assert row["id"] == "small-00000" and not row["facts"]
    for reply in ("Did your mom call again? Hehe~", "Just like old times, remember? We always do this~"):
        assert "invented" in reasons(row, reply, STYLE), reply


def test_the_report_shows_dropped_broken_lines(tmp_path, capsys):
    requests, rows = request_file(tmp_path, 2, 0)
    teacher_path = tmp_path / "teacher.jsonl"
    write_jsonl(teacher([(r["id"], "Hmph, hi~") for r in rows], rows), teacher_path)
    with teacher_path.open("a", encoding="utf-8") as f:
        f.write('{"id": "s9", "h": "')  # the teacher was killed mid-write
    out, report_path = tmp_path / "sft.jsonl", tmp_path / "report.json"
    assert main(["--requests", str(requests), "--teacher", str(teacher_path), "--out", str(out),
                 "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dropped_lines"] == {"requests": 0, "teacher": 1} and report["kept"] == 2
    printed = capsys.readouterr()
    assert "1 broken last line dropped (teacher)" in printed.out and "WARNING" in printed.err



# ---- final fix wave minor: a remember row trains only her reply after the engine's templated ack ----

def test_a_remember_row_trains_the_reply_after_the_ack():
    r = req("remember", facts=["The user said: their cat is called Pim"], user="remember that my cat is called Pim",
            must_not_invent=True, max_sentences=1)
    assert reasons(r, "Got it, I won't forget~", STYLE) == ["ack_only"]  # the engine already said one
    assert reasons(r, "Okay! Noted~ I'll remember Pim.", STYLE) == ["ack_only"]
    assert reasons(r, "Got it~ Pim sounds adorable!", STYLE) == []
    assert reasons(r, "Noted! Pim sounds adorable. And fluffy.", STYLE) == ["too_many_sentences"]  # one is left
    kept, report = filter_rows([r], teacher([("r", "Got it~ I'll remember that. Pim sounds adorable!")], [r]), STYLE)
    assert kept[0]["messages"][-1] == {"role": "assistant", "content": "Pim sounds adorable!"}
    assert report["reasons"]["ack_only"] == 0
    _, report = filter_rows([r], teacher([("r", "Okay, saved it.")], [r]), STYLE)
    assert report["kept"] == 0 and report["reasons"]["ack_only"] == 1
    small = req("small", user="got it?")  # only a remember turn has the engine's ack in front
    assert reasons(small, "Got it~ Pim sounds adorable!", STYLE) == []
    kept, _ = filter_rows([small], teacher([("r", "Got it~ Hehe.")], [small]), STYLE)
    assert kept[0]["messages"][-1]["content"] == "Got it~ Hehe."



def test_a_feedback_row_trains_the_reply_after_the_ack():
    r = req("feedback", user="less teasing please", must_not_invent=True, max_sentences=2)
    assert reasons(r, "Okay, I'll tease you less, promise~", STYLE) == ["ack_only"]  # the engine already said one
    assert reasons(r, "Got it~ Less teasing. So what's up?", STYLE) == []
    assert reasons(r, "Okay~ less teasing, got it. So what's up? Tell me. Now.", STYLE) == ["too_many_sentences"]
    kept, report = filter_rows([r], teacher([("r", "Okay~ less teasing, got it. So what's up?")], [r]), STYLE)
    assert kept[0]["messages"][-1] == {"role": "assistant", "content": "So what's up?"}
    _, report = filter_rows([r], teacher([("r", "Hehe, sweeter it is~")], [r]), STYLE)
    assert report["kept"] == 0 and report["reasons"]["ack_only"] == 1
