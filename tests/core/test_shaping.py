import pytest

from riya.core.shaping import BUDGETS, CAPS, cap_sentences, count_emoji, limit_emoji, tts_text, turn_kind


@pytest.mark.parametrize("about_me, thinking, expected", [
    (False, False, "small"), (False, True, "explain"), (True, False, "about_me"), (True, True, "about_me")])
def test_turn_kind_table(about_me, thinking, expected):
    assert turn_kind("anything", about_me=about_me, needs_thinking=thinking) == expected


def test_budgets_and_caps_are_the_specified_numbers():
    assert BUDGETS == {"small": 110, "about_me": 160, "explain": 380, "greeting": 90, "bid": 40}
    assert CAPS == {"small": 3, "about_me": 6, "explain": 8, "greeting": 3}


class Source:
    """An upstream sentence stream that records how far it was read and whether it was closed."""

    def __init__(self, sentences):
        self.taken, self.closed, self._it = 0, False, self._gen(sentences)

    def _gen(self, sentences):
        try:
            for s in sentences:
                self.taken += 1
                yield s
        finally:
            self.closed = True

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self._it.close()


def test_cap_stops_at_n_and_closes_upstream_without_reading_more():
    src = Source(["One.", "Two.", "Three.", "Four.", "Five."])
    assert list(cap_sentences(src, 3)) == ["One.", "Two.", "Three."]
    assert src.taken == 3 and src.closed


def test_cap_closes_upstream_as_the_last_sentence_is_handed_over():
    src = Source(["One.", "Two.", "Three."])
    it = cap_sentences(src, 2)
    assert [next(it), next(it)] == ["One.", "Two."] and src.closed and src.taken == 2


def test_cap_passes_short_streams_through_and_n_zero_reads_nothing():
    assert list(cap_sentences(iter(["Only one."]), 3)) == ["Only one."]
    src = Source(["x."])
    assert list(cap_sentences(src, 0)) == [] and src.taken == 0


def test_cap_keeps_a_code_fence_whole():
    code = "```py\nx = 1.\ny = 2.\n```"
    parts = ["Here you go.", "```py\nx = 1.", "y = 2.\n```", "Neat.", "More."]
    out = list(cap_sentences(iter(parts), 2))
    assert out == ["Here you go.", code]
    assert list(cap_sentences(iter(parts), 3)) == ["Here you go.", code, "Neat."]


def test_cap_with_an_unclosed_fence_yields_what_there_is():
    assert list(cap_sentences(iter(["```py", "x = 1."]), 3)) == ["```py\nx = 1."]


def test_limit_emoji_keeps_the_first_and_tidies_spaces():
    assert limit_emoji("Hi \U0001F600 there \U0001F44D and \U0001F525!") == "Hi \U0001F600 there and!"
    assert limit_emoji("\U0001F600 Hi", 0) == "Hi"
    assert limit_emoji("one \U0001F600 two", 2) == "one \U0001F600 two"
    assert count_emoji(limit_emoji("\U0001F600\U0001F600\U0001F600")) == 1


@pytest.mark.parametrize("text", ["வணக்கம், எப்படி இருக்கீங்க?", "Café naïve résumé — señor", "plain  text, 1 < 2 ~ ok",
                                  "a  b"])
def test_tamil_and_accents_pass_through_unchanged(text):
    assert limit_emoji(text) == text
    assert tts_text(text) == " ".join(text.split())


def test_tts_text_strips_emoji_and_stage_directions():
    assert tts_text("Hmph \U0001F624 *giggles* you're late. (laughs) Fine!") == "Hmph you're late. Fine!"
    assert tts_text("*sighs*") == ""
    assert tts_text("\U0001F600\U0001F44D") == ""
    assert tts_text("a\n\n  b\tc") == "a b c"


def test_tts_text_strips_only_narrow_stage_directions():
    out = tts_text("Compute 3 * 4 and 5 * 6 = f(x) (the cell's powerhouse) *really*.")
    assert out == "Compute 3 * 4 and 5 * 6 = f(x) (the cell's powerhouse)."
    assert tts_text("the cell (the powerhouse) makes energy") == "the cell (the powerhouse) makes energy"
    assert tts_text("Okay (sighs).") == "Okay."
    assert tts_text("Fine (laughs softly) then.") == "Fine then."
    assert tts_text("(blushing) Stop it. (Hmph!) Fine.") == "Stop it. Fine."
    assert tts_text("*giggles* Okay *Teasing*") == "Okay"
    assert tts_text("Hm (laughs at the long way round) ok") == "Hm (laughs at the long way round) ok"  # 5 words
    assert tts_text("It is (about 5 dollars) ok") == "It is (about 5 dollars) ok"
    assert tts_text("It costs *a very long aside that goes on for far too many words* ok") == (
        "It costs *a very long aside that goes on for far too many words* ok")
    assert tts_text("2**3 is **bold** maths") == "2**3 is bold maths"


def test_tts_text_replaces_a_fenced_block_and_leaves_it_alone():
    assert tts_text("Try this. ```py\nx = (1 * 2) # (laughs)\n``` Done.") == "Try this. here's the code Done."
    assert tts_text("Here: ```py\nx = 1") == "Here: here's the code"


def test_limit_emoji_never_touches_a_fenced_block():
    code = "```py\nprint('\U0001F600')   \U0001F44D\n```"
    assert limit_emoji(f"Hi \U0001F600 {code} bye \U0001F44D", 1) == f"Hi \U0001F600 {code} bye"
