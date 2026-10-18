"""v0.4 Task 5b: the games, pure. The engine keeps every rule and judgement; the model only voices the verdict."""
import random

import pytest

from riya.core import games
from riya.core.games import (MAX_TURNS, RIDDLES, THINGS, WORDS, detect_start, game_situation, guard, is_accept,
                             is_stop, mentions, new_game, opening, play, stop)


def rng():
    return random.Random(7)


# --- start and stop ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("let's play word chain", "word_chain"), ("Lets play Word-Chain!", "word_chain"),
    ("can we play 20 questions?", "twenty_questions"), ("play twenty questions", "twenty_questions"),
    ("could we play riddles", "riddles"), ("let's play a riddle", "riddles"),
    ("wanna play some riddles?", "riddles"), ("let's play a game of 20 questions", "twenty_questions")])
def test_detect_start_phrases(text, kind):
    assert detect_start(text, None) == kind


@pytest.mark.parametrize("text", ["word chain is fun", "I asked 20 questions today", "riddles are hard",
                                  "let's play minecraft", "hello", ""])
def test_detect_start_ignores_other_text(text):
    assert detect_start(text, None) is None


@pytest.mark.parametrize("text", ["sure", "Yeah!", "ok", "okay let's go", "let's go", "yes please", "sure, why not",
                                  "yep", "let's do it"])
def test_accepting_an_offer_starts_the_offered_game(text):
    assert is_accept(text)
    assert detect_start(text, "riddles") == "riddles"
    assert detect_start(text, None) is None


@pytest.mark.parametrize("text", ["no", "not now", "nah", "sure is hot today and I'm tired of work honestly",
                                  "yeah no", "ok but later"])
def test_declining_or_unrelated_text_is_no_accept(text):
    assert detect_start(text, "riddles") is None


def test_a_named_game_wins_over_the_offered_one():
    assert detect_start("sure, let's play word chain", "riddles") == "word_chain"


@pytest.mark.parametrize("text", ["stop", "Stop.", "quit", "enough", "stop the game", "let's stop", "I give up",
                                  "ok stop", "stop please", "enough!"])
def test_stop_phrases(text):
    assert is_stop(text)


@pytest.mark.parametrize("text", ["don't stop", "is it a bus stop?", "stop sign", "i quit my job", "enough food?"])
def test_stop_is_only_a_whole_message(text):
    assert not is_stop(text)


# --- the lists -----------------------------------------------------------------------------------------------------

def test_curated_lists_are_big_enough():
    assert len(THINGS) >= 55 and len(RIDDLES) >= 30
    common = "abcdefghijklmnoprstuvwy"
    assert all(len(WORDS[c]) >= 10 for c in common)
    assert all(w.isalpha() and w.islower() and w[0] == c for c, ws in WORDS.items() for w in ws)
    assert all(r.answers for r in RIDDLES)


# --- word chain ----------------------------------------------------------------------------------------------------

def chain(word="apple"):
    s = new_game("word_chain", rng())
    v = opening(s)
    assert v.kind == "start" and v.word == s.word and v.word in s.used
    assert v.say and v.key == v.word
    s.word, s.used = word, {word}
    return s


def test_word_chain_valid_word_gets_her_word():
    s = chain("apple")
    v = play(s, "Elephant!")
    assert v.kind == "ok" and v.word.startswith("t") and v.word in WORDS["t"]
    assert {"apple", "elephant", v.word} <= s.used and s.word == v.word and v.key == v.word
    assert f'"{v.word}"' in game_situation(s, v)


def test_word_chain_takes_the_word_out_of_a_sentence():
    s = chain("apple")
    assert play(s, "um, my word is eagle").kind == "ok"
    assert "eagle" in s.used


@pytest.mark.parametrize("text,reason", [("banana", "wrong letter"), ("e4gle", "not a word"), ("", "not a word"),
                                         ("xzqrt", "not a word"), ("e", "not a word"), ("apple", "wrong letter")])
def test_word_chain_invalid_words(text, reason):
    s = chain("apple")
    v = play(s, text)
    assert v.kind == "invalid" and v.reason == reason and s.word == "apple" and s.used == {"apple"}
    assert "starting with \"e\"" in v.line


def test_word_chain_repeat_is_invalid():
    s = chain("apple")
    s.used.add("eagle")
    v = play(s, "eagle")
    assert v.kind == "invalid" and v.reason == "used"


def test_word_chain_she_concedes_when_she_has_no_word():
    s = chain("apple")
    v = play(s, "equinox")                    # ends in x: she has one x word
    assert v.kind == "ok" and v.word == "xylophone"
    v = play(s, "eggbox")                     # e again, x again: none left
    assert v.kind == "win" and v.over and s.over and v.word is None


def test_word_chain_her_word_starts_with_their_last_letter():
    s = chain("apple")
    play(s, "egg")
    assert s.word[0] == "g"


# --- 20 questions --------------------------------------------------------------------------------------------------

def twenty(secret="banana"):
    s = new_game("twenty_questions", rng())
    assert s.secret in {t.name for t in THINGS}
    v = opening(s)
    assert v.kind == "start"
    s.secret, s.aliases = secret, next(t.aliases for t in THINGS if t.name == secret)
    return s


def test_twenty_questions_a_question_is_ok_and_counts():
    s = twenty()
    v = play(s, "is it alive?")
    assert v.kind == "ok" and s.questions == 1 and not s.over
    sit = game_situation(s, v)
    assert "banana" in sit and "never say" in sit.lower()


def test_twenty_questions_naming_it_wins():
    s = twenty()
    v = play(s, "Is it a BANANA?")
    assert v.kind == "win" and v.over and v.answer == "banana"


def test_twenty_questions_an_alias_or_plural_wins():
    s = twenty("bicycle")
    assert play(s, "bikes?").kind == "win"


def test_twenty_questions_the_20th_question_loses():
    s = twenty()
    s.questions = 19
    v = play(s, "is it blue?")
    assert v.kind == "loss" and v.over and v.answer == "banana" and "banana" in (v.say or "")


def test_giving_up_reveals_the_secret_but_stop_doesnt():
    s = twenty()
    v = stop(s, "I give up")
    assert v.kind == "ended" and v.reason == "gave up" and v.answer == "banana" and s.over
    assert "banana" in game_situation(s, v)
    s = twenty()
    v = stop(s, "stop")
    assert v.kind == "ended" and v.reason == "stopped" and v.answer is None
    assert "banana" not in game_situation(s, v)


# --- riddles -------------------------------------------------------------------------------------------------------

def riddles():
    s = new_game("riddles", rng())
    v = opening(s)
    assert v.kind == "start" and v.riddle == RIDDLES[s.riddle].question and v.say == v.riddle and v.key is None
    return s


def test_riddles_right_answer_moves_on():
    s = riddles()
    first = s.riddle
    v = play(s, f"is it {RIDDLES[first].answers[0]}?")
    assert v.kind == "right" and s.solved == 1 and s.riddle != first
    assert v.riddle == RIDDLES[s.riddle].question and v.say == v.riddle and v.answer == RIDDLES[first].answers[0]


def test_riddles_wrong_answer_keeps_the_riddle_then_reveals_after_three():
    s = riddles()
    first = s.riddle
    for _ in range(2):
        v = play(s, "a potato")
        assert v.kind == "wrong" and s.riddle == first and v.say is None
    v = play(s, "a potato")
    assert v.kind == "wrong" and v.answer == RIDDLES[first].answers[0] and s.riddle != first
    assert v.say == RIDDLES[s.riddle].question


def test_riddle_answers_match_whole_words_only():
    s = riddles()
    s.riddle = next(i for i, r in enumerate(RIDDLES) if r.answers == ("m",))
    assert play(s, "I'm not sure").kind == "wrong"
    assert play(s, "the letter M").kind == "right"


def test_riddles_run_out_is_a_win():
    s = riddles()
    s.left = []
    v = play(s, RIDDLES[s.riddle].answers[0])
    assert v.kind == "win" and v.over


def test_giving_up_a_riddle_reveals_its_answer():
    s = riddles()
    v = stop(s, "i give up")
    assert v.answer == RIDDLES[s.riddle].answers[0]


# --- the turn limit and the guard ----------------------------------------------------------------------------------

def test_the_15th_turn_ends_the_game():
    s = twenty()
    for _ in range(MAX_TURNS - 1):
        assert play(s, "is it alive?").kind == "ok"
    v = play(s, "is it alive?")
    assert v.kind == "ended" and v.reason == "turns" and s.over and v.answer == "banana"
    assert "banana" in v.say and "over" in game_situation(s, v)


def test_a_win_on_the_15th_turn_is_still_a_win():
    s = twenty()
    s.turns = MAX_TURNS - 1
    assert play(s, "banana").kind == "win"


def test_word_chain_15th_turn_ends_without_her_word():
    s = chain("apple")
    s.turns = MAX_TURNS - 1
    v = play(s, "eagle")
    assert v.kind == "ended" and v.reason == "turns" and v.word is None and v.say is None


def test_guard_replaces_sentences_naming_the_secret_while_live():
    s = twenty("ice cream")
    out = guard(s, "Haha, it's ice cream, obviously!")
    assert "ice cream" not in out.lower() and out in games.TEASES
    assert guard(s, "Ice-creams are yummy.") in games.TEASES
    assert guard(s, "Yes, it's cold!") == "Yes, it's cold!"
    s.over = True
    assert guard(s, "It was ice cream!") == "It was ice cream!"


def test_guard_only_matters_for_twenty_questions():
    s = chain("apple")
    assert guard(s, "Elephant is a nice word.") == "Elephant is a nice word."


def test_mentions_is_whole_words():
    assert mentions("My word: Tiger!", "tiger") and not mentions("tigers are big", "tiger")
    assert mentions("anything", None) is False


def test_situation_keeps_the_rules_and_one_sentence():
    for kind in ("word_chain", "twenty_questions", "riddles"):
        s = new_game(kind, rng())
        sit = game_situation(s, opening(s))
        assert "one short" in sit and "playing" in sit
