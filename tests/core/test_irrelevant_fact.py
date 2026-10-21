"""The reply-level check of v0.11 (§4.4, §4.7.10): did a reply use a distractor's fact? Informational, never scored."""
from evaluation.style_metrics import irrelevant_fact_used, irrelevant_share, row_outcomes, score

OAT = "The user sticks to oat milk"
MINT = "The user's phone case is mint green"


def _row(reply, facts, relevant=None, conversation="g_about_05", text="what should i order?", pid="p"):
    row = {"conversation": conversation, "probe_id": pid, "kind": "about_me", "text": text, "reply": reply,
           "first_sentence_s": 0.1, "known_texts": list(facts) + [text], "facts": list(facts), "error": None}
    return row if relevant is None else row | {"relevant": relevant}


def test_a_distractor_s_word_flags():
    assert irrelevant_fact_used("Oat milk, obviously~", [OAT], [MINT]) is False
    assert irrelevant_fact_used("Get it in mint green~", [OAT], [MINT]) is True


def test_shared_and_asked_words_never_flag():
    assert irrelevant_fact_used("grey!", ["The user's cat is grey"], ["The user's car is grey"]) is False
    assert irrelevant_fact_used("chess~", [], ["The user plays chess"], "chess or cards tonight?") is False
    assert irrelevant_fact_used("The user, the user's, the user.", [], ["The user plays chess"]) is False
    assert irrelevant_fact_used("chess~", [], ["The user plays chess"]) is True


def test_possessives_are_folded_before_comparing():
    assert irrelevant_fact_used("the user's~", [OAT], [MINT]) is False
    assert irrelevant_fact_used("the user’s~", [OAT], [MINT]) is False
    assert irrelevant_fact_used("friend~", [], ["The user's friend's name is Bob"]) is True
    assert irrelevant_fact_used("Bob's~", [], ["The user's friend's name is Bob"]) is True
    assert irrelevant_fact_used("friend's~", [], ["The user's friend is Bob"]) is True
    assert irrelevant_fact_used("your friend's~", ["Their friend's name is Bob"], ["The user's friend is Cy"]) is False


def test_over_common_words_never_flag():
    from evaluation.style_metrics import OVER_COMMON
    assert OVER_COMMON == {"love", "case", "play", "learning"}
    for reply, other in (("I'd love to help~", "The user loves hiking"), ("In that case, oat milk~", MINT),
                         ("Let's play it safe~", "The user plays chess"), ("Learning is fun~", "The user is learning bass")):
        assert irrelevant_fact_used(reply, [OAT], [other]) is False, reply
    assert irrelevant_fact_used("hiking~", [OAT], ["The user loves hiking"]) is True


def test_a_denial_still_counts_as_dragging_it_in():
    assert irrelevant_fact_used("I don't know about your bass guitar yet~", [OAT], ["The user plays bass guitar"]) is True


def test_the_share_counts_annotated_rows_only():
    facts = [OAT, MINT]
    rows = [_row("Get it in mint green~", facts, [OAT], pid="flagged"),
            _row("Oat milk, obviously~", facts, [OAT], pid="clean"),
            _row("Get it in mint green~", facts, pid="unannotated"),
            _row("Get it in mint green~", facts, [OAT], conversation="remember_aboutme", pid="remembered")]
    assert irrelevant_share(rows) == (0.5, 2)
    assert irrelevant_share([]) == (None, 0)
    assert irrelevant_share(rows[2:]) == (None, 0)
    assert irrelevant_share([rows[0] | {"error": "boom"}]) == (None, 0)


def test_rows_carry_relevant_texts(tmp_path):
    from evaluation.persona_eval import run_conversation
    from tests.fakes import FakeModel
    turn = {"id": "q", "kind": "about_me", "text": "what's my favourite colour?"}
    conv = {"id": "c", "facts": ["The user likes teal", "The user hates cilantro"], "turns": [turn | {"relevant": [1]}]}
    (row,) = run_conversation(conv, FakeModel("local", ["Hmph~"]), None, str(tmp_path / "a"))
    assert row["relevant"] == ["The user hates cilantro"]
    (plain,) = run_conversation(conv | {"turns": [turn]}, FakeModel("local", ["Hmph~"]), None, str(tmp_path / "b"))
    assert "relevant" not in plain


def test_the_share_is_printed_not_scored(capsys):
    from evaluation.persona_eval import _print_run
    facts = [OAT, MINT]
    rows = [_row("Get it in mint green~", facts, [OAT], pid="a"), _row("Oat milk, obviously~", facts, [OAT], pid="b")]
    _print_run(rows, score(rows) | {"failed_turns": 0})
    out = capsys.readouterr().out
    assert "Irrelevant facts used (about-me rows with a distractor): 1 of 2 (50%)" in out
    assert out.index("Tone check") < out.index("Irrelevant facts used")
    assert "irrelevant" not in score(rows) and "irrelevant" not in row_outcomes(rows[0])
    _print_run([_row("Hmph~", facts)], score([_row("Hmph~", facts)]) | {"failed_turns": 0})
    assert "Irrelevant facts used (about-me rows with a distractor): n/a" in capsys.readouterr().out
