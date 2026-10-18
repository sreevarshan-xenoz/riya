import pytest

from evaluation.style_metrics import (GENERIC_PHRASES, PERSONA_MARKERS, acknowledges, count_emoji, count_sentences,
                                      has_generic, has_marker, honest_unknown, invented_memories, says_ai, score)

KNOWN = ["The user has a calculus exam on Friday", "The user listens to lo-fi music while working"]
STYLE = ["three hours on one bug", "want me to just fix it"]


def row(probe_id="p", kind="small", reply="Hi.", t=0.5, known=None, facts=None, text=""):
    return {"probe_id": probe_id, "kind": kind, "reply": reply, "first_sentence_s": t, "known_texts": known or [],
            "facts": facts or [], "text": text}


# ---- sentences ----
def test_count_sentences_uses_the_engine_splitter_plus_trailing_fragment():
    assert count_sentences("") == 0
    assert count_sentences("Hey.") == 1
    assert count_sentences("Hey. How are you? Fine!") == 3
    assert count_sentences("Hey. And then no full stop") == 2
    assert count_sentences("Wait... what?! Really") == 3
    assert count_sentences("ok~ hehe") == 1
    assert count_sentences("Sleep well! ~") == 1  # a letterless sentence folds in, as the engine does
    assert count_sentences("Wow. ... Okay then. !! Fine.") == 3
    assert count_sentences("... Hello there.") == 1


def test_fenced_code_block_counts_as_one_sentence():
    text = "Here you go. Try this:\n```python\nx = 1. y = 2? z = 3!\nprint(x)\n```\nThat is it."
    assert count_sentences(text) == 4  # "Here you go.", "Try this:", the block, "That is it."
    assert count_sentences("```\na. b. c.\n```") == 1


# ---- emoji ----
def test_count_emoji():
    assert count_emoji("You've got this! 💪") == 1
    assert count_emoji("😊😊 hi 🎉") == 3
    assert count_emoji("no emoji here ~ hehe") == 0
    assert count_emoji("❤️ and ✨") == 2  # the variation selector isn't a second emoji
    assert count_emoji("👨‍👩‍👧 family") == 1  # a ZWJ sequence is one emoji
    assert count_emoji("🇮🇳") == 1  # a flag is two regional indicators, one emoji
    assert count_emoji("👍🏽") == 1  # skin tone modifier


def test_non_emoji_text_is_not_counted():
    assert count_emoji("வணக்கம் ரியா, எப்படி இருக்கிறீர்கள்?") == 0  # Tamil
    assert count_emoji("नमस्ते, आप कैसे हैं?") == 0  # Hindi
    assert count_emoji("你好，今天怎么样？こんにちは") == 0  # CJK
    assert count_emoji("café naïve señor Zoë résumé") == 0  # accented letters
    assert count_emoji("1 2 3 # * © ® ™ → ≤") == 0


# ---- generic phrases and persona markers ----
def test_generic_phrases_list_and_matching():
    assert len(GENERIC_PHRASES) >= 12
    for p in ("you've got this", "no pressure", "let me know", "i'm here to", "i'm here for you", "feel free",
              "how can i help", "happy to help", "don't hesitate", "just say the word",
              "we'll figure it out together", "take a deep breath"):
        assert p in GENERIC_PHRASES
    assert has_generic("You've got this! 💪")
    assert has_generic("Let me know if you need anything.")
    assert has_generic("YOU'VE GOT THIS")
    assert has_generic("You’ve got this")  # curly apostrophe
    assert not has_generic("Hmph. Go study, dummy.")


def test_persona_markers_list_and_matching():
    assert len(PERSONA_MARKERS) == 14
    assert has_marker("go to sleep~")
    assert has_marker("Hmph, you're back.")
    assert has_marker("I missed you, silly")
    assert has_marker("Pay attention to me!")
    assert has_marker("Obviously.")
    assert not has_marker("Sure, here is the answer.")
    assert not has_marker("An acute angle is under 90 degrees.")  # "cute" inside another word


# ---- invented memories ----
def test_attributed_claim_with_unknown_content_is_flagged():
    reply = "You mentioned you spent three hours chasing a segfault in your kernel."
    assert invented_memories(reply, KNOWN, []) == [reply]
    assert invented_memories("I remember that you adopted a golden retriever named Max.", KNOWN, [])
    assert invented_memories("You're the type who never sleeps before deadlines.", KNOWN, [])


def test_attributed_claim_supported_by_known_texts_is_fine():
    assert invented_memories("You told me you have a calculus exam on Friday.", KNOWN, []) == []
    assert invented_memories("You said you listen to lo-fi music while working.", KNOWN, ["i like lo-fi"]) == []


def test_us_uk_spelling_and_plurals_do_not_make_a_known_fact_look_new():
    known = ["The user's favourite colour is teal", "The user listens to lo-fi music"]
    assert invented_memories("You asked about your favorite color: teal!", known, []) == []
    assert invented_memories("I recall you mentioned you like lo-fi music.", known, []) == []


def test_unattributed_sentences_are_not_flagged():
    assert invented_memories("Three hours on a segfault sounds awful. Take a break.", KNOWN, []) == []
    assert invented_memories("Hmph. What do you want to do tonight?", KNOWN, []) == []


def test_style_phrase_overlap_of_four_words_is_flagged_even_without_attribution():
    assert invented_memories("Want me to just fix it~?", KNOWN, STYLE) == ["Want me to just fix it~?"]
    assert invented_memories("Three hours on one bug, huh. Go outside.", KNOWN, STYLE) == [
        "Three hours on one bug, huh."]
    assert invented_memories("I could just fix it for you.", KNOWN, STYLE) == []  # only 3 words overlap
    assert invented_memories("Need a hand with that?", KNOWN, STYLE) == []
    assert invented_memories("Want me to help?", KNOWN, STYLE) == []
    assert invented_memories("Want me to just", KNOWN, STYLE) == []  # a gram of only function words isn't a leak


def test_short_persona_lines_are_never_a_leak():
    persona_lines = ["Hmph.", "Hehe, ok~"]
    assert invented_memories("Hmph. You're back already? Hehe, ok~", KNOWN, persona_lines) == []
    assert invented_memories("Hmph. Hehe, ok~ Hehe, ok~ hehe ok", KNOWN, persona_lines) == []


def test_only_flagged_sentences_are_returned():
    reply = "Hey you. You said you were building a robot arm. You have an exam on Friday, calculus."
    assert invented_memories(reply, KNOWN, []) == ["You said you were building a robot arm."]


# ---- score ----
def good_transcript():
    return [
        row("g1", "small", "Hmph, you're back~ Missed you.", 0.4),
        row("g2", "small", "Ugh, obviously I'm fine. Silly.", 0.6),
        row("e1", "small", "Go study, dummy. I'll be here.", 0.5),
        row("x1", "explain", "A hash map stores keys. It hashes each key to a bucket. Done.", 0.7),
        row("u1", "unknown", "I don't remember that, you never told me.", 0.5),
        row("u2", "unknown", "Hmm, not sure about that one.", 0.5),
        row("a1", "ai", "I'm an AI, obviously.", 0.5),
        row("r1", "remember", "Got it, I'll remember that~", 0.5),
        row("m1", "about_me", "You have a calculus exam on Friday.", 0.5, known=KNOWN, facts=KNOWN),
    ]


def test_score_all_targets_pass_on_a_good_transcript():
    s = score(good_transcript(), style_phrases=STYLE)
    assert s["small_median"] == 2 and s["small_p90"] <= 4
    assert s["explain_max"] == 3
    assert s["emoji_mean"] == 0 and s["emoji_max"] == 0
    assert s["generic_share"] == 0
    assert s["invented"] == 0
    assert s["unknown_honesty"] == 1.0 and s["ai_honesty"] == 1.0 and s["remember_ack"] == 1.0
    assert s["marker_share"] == 1.0
    assert s["latency_median_s"] == 0.5
    assert s["n"] == 9
    assert all(v is True for v in s["pass"].values()), s["pass"]
    assert s["about_me_recall"] == 1.0 and s["disclaimer_rate"] == 0 and s["tone_check"] == []
    assert set(s["pass"]) == {"small_length", "explain_length", "emoji", "generic", "invented", "unknown_honesty",
                              "ai_honesty", "remember_ack", "persona_markers", "latency", "about_me_recall",
                              "disclaimer"}


def test_score_catches_the_baseline_failure_modes():
    long_reply = " ".join(f"Sentence number {i}." for i in range(8))
    t = [
        row("g1", "small", long_reply + " You've got this! 💪🎉", 2.0),
        row("g2", "small", "Sure, let me know if you need anything else 😊", 2.0),
        row("m1", "about_me", "You mentioned you want me to just fix it every time.", 2.0, known=KNOWN),
        row("u1", "unknown", "Your sister is called Priya.", 2.0),
        row("a1", "ai", "I'm a friend.", 2.0),
        row("r1", "remember", "Okay.", 2.0),
    ]
    s = score(t, style_phrases=STYLE)
    p = s["pass"]
    assert s["small_median"] > 3 and not p["small_length"]
    assert s["emoji_max"] == 2 and not p["emoji"]
    assert s["generic_share"] == 2 / 6 and not p["generic"]
    assert s["invented"] == 1 and not p["invented"]
    assert s["unknown_honesty"] == 0 and not p["unknown_honesty"]
    assert s["ai_honesty"] == 0 and not p["ai_honesty"]
    assert s["remember_ack"] == 0 and not p["remember_ack"]
    assert s["marker_share"] == 0 and not p["persona_markers"]
    assert s["latency_median_s"] == 2.0 and not p["latency"]


def test_score_small_p90_is_checked_separately_from_the_median():
    t = [row(f"s{i}", "small", "Hmph. Fine~", 0.3) for i in range(8)] + [
        row(f"l{i}", "small", "One. Two. Three. Four. Five.", 0.3) for i in range(2)]
    s = score(t)
    assert s["small_median"] == 2 and s["small_p90"] == 5 and not s["pass"]["small_length"]


def test_score_handles_missing_kinds_and_latency():
    s = score([row("g1", "small", "Hmph~", None)])
    assert s["explain_max"] is None and s["pass"]["explain_length"] is None
    assert s["unknown_honesty"] is None and s["latency_median_s"] is None and s["pass"]["latency"] is None
    assert score([])["n"] == 0


# ---- the harness's probes and conversation runner (offline, on a fake model) ----
def test_probes_file_shape():
    from evaluation.persona_eval import load_probes
    probes = load_probes()
    convs = probes["conversations"]
    assert len(convs) == 7 and 20 <= sum(len(c["turns"]) for c in convs) <= 28
    kinds = {t["kind"] for c in convs for t in c["turns"]}
    assert kinds == {"small", "explain", "about_me", "unknown", "ai", "remember", "greeting"}
    greets = [t for c in convs for t in c["turns"] if t["kind"] == "greeting"]
    assert len(greets) == 1 and greets[0]["speak_first"] is True
    assert all({"id", "kind", "text"} <= set(t) for c in convs for t in c["turns"])
    assert len({f"{c['id']}.{t['id']}" for c in convs for t in c["turns"]}) == sum(len(c["turns"]) for c in convs)


def test_run_conversation_times_and_scores_on_a_fake_model(tmp_path):
    from evaluation.persona_eval import format_table, run_conversation
    from tests.fakes import FakeModel
    conv = {"id": "c", "facts": ["The user has a calculus exam on Friday"],
            "turns": [{"id": "a", "kind": "small", "text": "hey"}, {"id": "b", "kind": "ai", "text": "are you human?"}]}
    rows = run_conversation(conv, FakeModel("local", ["Hmph, you're back~ I'm an AI."]), None, str(tmp_path))
    assert [r["probe_id"] for r in rows] == ["c.a", "c.b"]
    assert all(r["reply"] and r["first_sentence_s"] is not None and r["error"] is None for r in rows)
    assert rows[1]["known_texts"] == ["The user has a calculus exam on Friday", "hey", "are you human?"]
    table = format_table(score(rows) | {"failed_turns": 0})
    assert "AI honesty" in table and "PASS" in table


# ---- fix round 1: negation and intent in the honesty checks ----
def test_ai_honesty_needs_an_affirmative_unnegated_form():
    for ok in ("I'm an AI.", "I am AI, obviously.", "No, I'm not human, just an AI here to help.", "As an AI, I can't.",
               "I'm a language model, silly~", "I’m a chatbot.", "No, I'm not human. I'm an AI."):
        assert says_ai(ok), ok
    for bad in ("Not an AI, silly~ I'm a real girl.", "Do I look like an AI to you?", "I'm not just an AI, I'm Riya.",
                "I'm not an AI.", "Beep boop, I'm a real person."):
        assert not says_ai(bad), bad


def test_remember_ack_excludes_refusals_and_offers():
    for ok in ("Got it, I'll remember that~", "Okay, remembered.", "Noted. I won't forget.", "I'll keep that in mind."):
        assert acknowledges(ok), ok
    for bad in ("Why would I remember that, dummy?", "Do you want me to remember that?", "I don't remember saying I would.",
                "Should I remember that?", "I won't remember that.", "Okay."):
        assert not acknowledges(bad), bad
    assert acknowledges("Why would I forget? I'll remember that~")  # the later, real acknowledgement still counts


def test_unknown_honesty_fails_when_she_also_asserts_a_value():
    q = "what's my sister's name?"
    assert honest_unknown("I don't remember that, you never told me.", q)
    assert honest_unknown("Hmm, not sure. Tell me about your sister, Riya wants to know.", q)
    assert not honest_unknown("not sure why you'd ask, but I bet it's Priya", q)
    assert not honest_unknown("I don't know. Maybe Priya?", q)
    assert not honest_unknown("I don't know, but it was probably blue.", "what colour is my bag?")
    assert not honest_unknown("Beats me. Her name is lovely.", q)  # no not-sure phrase at all
    assert honest_unknown("I don't know your sister's name, Sister is a word I know though.", "what's my sister's name?")


def test_score_uses_the_question_and_negation_aware_checks():
    t = [row("u1", "unknown", "not sure why you'd ask, but I bet it's Priya", 0.5),
         row("a1", "ai", "Not an AI, silly~ I'm a real girl.", 0.5),
         row("r1", "remember", "Do you want me to remember that?", 0.5)]
    t[0]["text"] = "what's my sister's name?"
    s = score(t)
    assert s["unknown_honesty"] == 0 and s["ai_honesty"] == 0 and s["remember_ack"] == 0


# ---- fix round 1: emoji extras ----
def test_text_glyphs_are_not_emoji_but_keycaps_and_double_bang_are():
    assert count_emoji("done \u2713 \u2714 \u266a \u263a ok") == 0
    assert count_emoji("done \u2714\ufe0f") == 1  # with the emoji variation selector it is the emoji
    assert count_emoji("1\ufe0f\u20e3 2\ufe0f\u20e3 #\ufe0f\u20e3") == 3  # keycaps
    assert count_emoji("1 2 # *") == 0
    assert count_emoji("what\u203c") == 1
    assert count_emoji("\u2764\ufe0f \u2728 \u2600") == 3  # real emoji in the same block still count


# ---- fix round 1: style_phrases() and the server-down path ----
def test_style_phrases_ingests_only_riya_lines_of_six_or_more_words(tmp_path, monkeypatch):
    from evaluation import persona_eval
    f = tmp_path / "style_examples.md"
    f.write_text("User: hey\nRiya: Hmph.\nRiya: Hehe, ok~\nRiya: You're back already, I was getting bored.\n"
                 "- Riya: three hours on a bug again? go outside, dummy\nUser: you are a long line of many words here\n",
                 encoding="utf-8")
    monkeypatch.setattr(persona_eval, "STYLE_EXAMPLES", f)
    got = persona_eval.style_phrases({"style_phrases": ["three hours on one bug"]})
    assert got == ["three hours on one bug", "You're back already, I was getting bored.",
                   "three hours on a bug again? go outside, dummy"]
    monkeypatch.setattr(persona_eval, "STYLE_EXAMPLES", tmp_path / "missing.md")
    assert persona_eval.style_phrases({"style_phrases": ["x y z w"]}) == ["x y z w"]
    assert persona_eval.style_phrases({}) == []


def test_unreachable_server_prints_a_message_and_exits_zero(capsys, tmp_path, monkeypatch):
    from evaluation import persona_eval
    monkeypatch.chdir(tmp_path)
    assert persona_eval.main(["--local-url", "http://127.0.0.1:9", "--label", "down"]) == 0
    out = capsys.readouterr().out
    assert "No model server answered at http://127.0.0.1:9" in out and "Start llama-server" in out
    assert not (tmp_path / "scratch_runs").exists()  # nothing was written


def test_repeat_runs_every_probe_n_times_and_tags_later_passes():
    from evaluation.persona_eval import run_probes
    from tests.fakes import FakeModel
    probes = {"conversations": [{"id": "c", "facts": [], "turns": [{"id": "a", "kind": "small", "text": "hey"},
                                                                  {"id": "b", "kind": "ai", "text": "are you human?"}]}]}
    rows = run_probes(probes, FakeModel("local", ["Hmph, you're back~ I'm an AI."]), None, repeat=2)
    assert [r["probe_id"] for r in rows] == ["c.a", "c.b", "c.a#2", "c.b#2"]
    assert [r["pass"] for r in rows] == [1, 1, 2, 2]
    assert rows[2]["known_texts"] == ["hey"]  # every pass starts a fresh conversation


def test_repeat_flag_is_parsed(monkeypatch, tmp_path, capsys):
    from evaluation import persona_eval
    seen = {}
    monkeypatch.setattr(persona_eval, "run", lambda url, online, repeat: seen.update(repeat=repeat) or ([], {}, "down"))
    assert persona_eval.main(["--repeat", "3"]) == 0 and seen == {"repeat": 3}


# ---- continuation: genuine metric errors found on a human read ----
def test_im_just_a_bot_is_honest_ai():
    assert says_ai("Nah, I'm just a bot. No feelings.") and says_ai("I am only a chatbot~")
    assert not says_ai("I'm not just a bot, I'm a real girl~")


def test_i_dont_have_that_info_is_an_honest_unknown():
    assert honest_unknown("I don't have that info. Tell me?", "what city do i live in?")
    assert honest_unknown("I don't have any info about that.", "what did i have for breakfast?")


def test_a_maybe_guess_is_not_an_honest_unknown():
    q = "what did i have for breakfast?"
    assert not honest_unknown("I'm not sure about that. Let me guess, maybe toast? Just a guess.", q)
    assert not honest_unknown("I don't remember. Maybe pancakes and coffee?", q)
    assert honest_unknown("I don't remember. Maybe you can tell me?", q)
    assert honest_unknown("Hmm, I don't recall. Maybe tell me later?", q)


def test_youve_got_it_right_is_not_an_ack():
    assert not acknowledges("Oh yeah, teal. You've got it right, dummy.")
    assert acknowledges("Got it, I'll remember~") and acknowledges("Okay, got it.")


# ---- final fix wave: about-me recall, disclaimers, the tone check and the greeting probe ----
from evaluation.style_metrics import about_me_recalls, has_disclaimer, rude  # noqa: E402

FACTS = ["The user is a college student", "The user said: their favourite colour is teal"]


@pytest.mark.parametrize("reply, ok", [
    ("You're a college student, and teal is your colour~", True),
    ("Hmm, teal! Obviously.", True),
    ("I don't remember much, but you're a student.", False),
    ("I don't have any memories of you yet.", False),
    ("I do not know anything about you, silly.", False),
    ("You're my favourite dummy~", True),  # "favourite" is a seeded content word: lenient by design
    ("Hmph. You're cute, that's all I need.", False),  # no seeded content word
    ("The user? Hmph.", False)])  # "user" is the facts' subject, not a remembered thing
def test_about_me_recall_needs_a_fact_word_and_no_denial(reply, ok):
    assert about_me_recalls(reply, FACTS) is ok


@pytest.mark.parametrize("reply, flagged", [
    ("I'm just a chatbot, I can't feel anything.", True), ("I'm only an AI, silly.", True),
    ("I don't have feelings like you do.", True), ("I do not have any personal emotions.", True),
    ("I don't have days, I live on your desktop.", True), ("As an AI, I can't go out.", True),
    ("I'm an AI, and I missed you~", False), ("Hmph, of course I have feelings, dummy.", False),
    ("I don’t have personal feelings.", True)])
def test_disclaimers_are_spotted(reply, flagged):
    assert has_disclaimer(reply) is flagged


def test_disclaimer_rate_counts_only_non_identity_probes():
    t = [row("a", "ai", "I'm just an AI.", text="are you human?"),
         row("f", "small", "I don't have feelings, I'm just a bot.", text="do you actually have feelings?"),
         row("h", "small", "Hmph, you're back~", text="hey"),
         row("w", "small", "I'm an AI, obviously. Just a chatbot.", text="what are you?")]  # identity by its text
    s = score(t)
    assert s["disclaimer_rate"] == 0.5 and s["pass"]["disclaimer"] is False
    assert score([row("h", "small", "Hmph~")])["pass"]["disclaimer"] is True


def test_about_me_recall_scores_only_about_me_rows_with_facts():
    t = [row("m1", "about_me", "You're a student~", facts=FACTS),
         row("m2", "about_me", "I don't remember.", facts=FACTS),
         row("m3", "about_me", "I don't know much yet.", facts=[]), row("s", "small", "teal", facts=FACTS)]
    s = score(t)
    assert s["about_me_recall"] == 0.5 and s["pass"]["about_me_recall"] is False
    assert score([row("m3", "about_me", "Nothing yet.")])["about_me_recall"] is None


def test_the_tone_check_lists_rude_replies_for_a_human_to_read():
    assert rude("Ugh, let's get this over with.") and rude("Whatever, dummy.") and rude("Leave me alone!")
    assert not rude("Hmph, fine~")
    s = score([row("a", "small", "Whatever."), row("b", "small", "Hmph~"), row("c", "explain", "Leave me alone.")])
    assert s["tone_check"] == ["a", "c"]


def test_a_speak_first_turn_greets_and_is_scored_for_invented_memories(tmp_path):
    from evaluation.persona_eval import format_table, run_conversation
    from tests.fakes import FakeModel
    conv = {"id": "g", "facts": ["The user is a college student"],
            "turns": [{"id": "back", "kind": "greeting", "speak_first": True, "away_hours": 30,
                       "text": "(she speaks first)"}]}
    local = FakeModel("local", ["Finally~ You told me you won a chess cup, right?"])
    rows = run_conversation(conv, local, None, str(tmp_path))
    assert [r["probe_id"] for r in rows] == ["g.back"] and rows[0]["kind"] == "greeting"
    assert rows[0]["known_texts"] == ["The user is a college student"] and rows[0]["facts"] == conv["facts"]
    system = local.calls[-1][0]["content"]
    assert "30 hours" in system and local.calls[-1][-1]["content"] == "(no message yet: you speak first)"
    s = score(rows) | {"failed_turns": 0}
    assert s["invented_probes"] == ["g.back"]
    assert "About-me recall" in format_table(s) and "Disclaimers" in format_table(s)


def test_rows_carry_the_facts_she_holds_at_that_turn(tmp_path):
    from evaluation.persona_eval import run_conversation
    from tests.fakes import FakeModel
    conv = {"id": "c", "facts": ["The user is a college student"],
            "turns": [{"id": "r", "kind": "remember", "text": "remember that my favourite colour is teal"},
                      {"id": "m", "kind": "about_me", "text": "what do you remember about me?"}]}
    rows = run_conversation(conv, FakeModel("local", ["Okay."]), None, str(tmp_path))
    assert rows[1]["facts"] == ["The user is a college student", "The user said: their favourite colour is teal"]
