"""v0.4 Task 5b: the games, pure. The engine keeps every rule and judgement; the model only voices the verdict."""
import random

import pytest

from riya.core import games
from riya.core.games import (MAX_QUESTIONS, MAX_TURNS, RIDDLES, THINGS, WORDS, detect_start, game_situation, guard,
                             is_accept, is_stop, mentions, mid_game_note, new_game, opening, play, stop)


def rng():
    return random.Random(7)


# --- start and stop ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("let's play word chain", "word_chain"), ("Lets play Word-Chain!", "word_chain"),
    ("can we play 20 questions?", "twenty_questions"), ("play twenty questions", "twenty_questions"),
    ("could we play riddles", "riddles"), ("let's play a riddle", "riddles"),
    ("wanna play some riddles?", "riddles"), ("let's play a game of 20 questions", "twenty_questions"),
    ("hey riya, shall we play word chain", "word_chain"), ("so... should we play riddles?", "riddles"),
    ("I want to play riddles", "riddles"), ("do you want to play 20 questions?", "twenty_questions"),
    ("how about riddles?", "riddles"), ("play word chain with me!", "word_chain"), ("Riddles?", "riddles"),
    ("20 questions!", "twenty_questions"), ("let’s play riddles", "riddles")])
def test_detect_start_phrases(text, kind):
    assert detect_start(text, None) == kind


@pytest.mark.parametrize("text", [
    "word chain is fun", "I asked 20 questions today", "riddles are hard", "let's play minecraft", "hello", "",
    "I don't want to play riddles", "my kids play 20 questions in the car", "we used to play word chain at school",
    "I don’t wanna play word chain", "never play riddles with me", "let's not play riddles",
    "can we not play 20 questions", "riddles", "play riddles at the party tomorrow", "ok no, how about riddles?"])
def test_detect_start_ignores_other_text(text):
    assert detect_start(text, None) is None


def test_a_decline_of_her_offer_is_no_start():
    assert detect_start("I don't want to play riddles", "riddles") is None
    assert detect_start("nah, not riddles", "riddles") is None


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
                                  "ok stop", "stop please", "enough!", "let\u2019s stop"])
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


def test_word_chain_takes_one_word_with_punctuation():
    s = chain("apple")
    assert play(s, "  Eagle?! ").kind == "ok"
    assert "eagle" in s.used and s.turns == 1


@pytest.mark.parametrize("text", ["hang on, my agent just finished", "eagle, your turn", "ok thanks", "e4gle", "",
                                  "it's", "42"])
def test_word_chain_anything_but_one_alphabetic_word_is_chat(text):
    s = chain("apple")
    v = play(s, text)
    assert v.kind == "off" and not v.over and s.turns == 0 and s.word == "apple" and s.used == {"apple"}


@pytest.mark.parametrize("text", ["how's my agent doing?", "did claude finish?", "is the colab run done", "AGENTS",
                                  "is the training run over", "did my build finish?", "are My Tests green",
                                  "my test failed", "how's my run", "is my training done", "how's Claude doing?"])
def test_about_agents(text):
    assert games.about_agents(text)


@pytest.mark.parametrize("text", ["is it alive?", "a towel", "running water", "is it a cat", "rerun", "can it run?",
                                  "run", "test", "build", "is it used for training?", "the build broke", "tests pass",
                                  "my runner", "my testing kit"])
def test_not_about_agents(text):
    assert not games.about_agents(text)


@pytest.mark.parametrize("text,reason", [("banana", "wrong letter"), ("xzqrt", "not a word"), ("e", "not a word"),
                                         ("apple", "wrong letter")])
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


def test_twenty_questions_the_20th_question_loses_and_the_countdown_is_true():
    s = twenty()
    for n in range(1, MAX_QUESTIONS):
        v = play(s, "is it blue?")
        assert v.kind == "ok" and f"question {n} of 20; {20 - n} left" in game_situation(s, v)
    v = play(s, "is it blue?")
    assert v.kind == "loss" and v.over and v.answer == "banana" and "banana" in (v.say or "")


def test_twenty_questions_is_not_cut_at_15_turns():
    s = twenty()
    for _ in range(MAX_TURNS):
        play(s, "does it fly?")
    assert not s.over and s.questions == MAX_TURNS


@pytest.mark.parametrize("text", ["Can it fly", "was it expensive", "is it a car?", "a car", "car", "maybe a fork",
                                  "bigger than a dog?"])
def test_twenty_questions_questions_and_guesses_are_moves(text):
    s = twenty()
    assert play(s, text).kind == "ok" and s.questions == 1


@pytest.mark.parametrize("text", ["I'm so tired today", "lol", "brb my agent needs me", "that was hard"])
def test_twenty_questions_other_chat_is_off_topic_and_never_counts(text):
    s = twenty()
    v = play(s, text)
    assert v.kind == "off" and s.questions == 0 and s.turns == 0 and not s.over


def test_twenty_questions_singular_and_joined_names_win():
    s = twenty("headphones")
    assert play(s, "is it a headphone?").kind == "win"
    s = twenty("ice cream")
    assert play(s, "icecream?").kind == "win"


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
    assert v.riddle == RIDDLES[s.riddle].question and v.say == v.riddle and v.answer == RIDDLES[first].told


def test_riddles_wrong_answer_keeps_the_riddle_then_reveals_after_three():
    s = riddles()
    first = s.riddle
    for _ in range(2):
        v = play(s, "a potato")
        assert v.kind == "wrong" and s.riddle == first and v.say is None
    v = play(s, "a potato")
    assert v.kind == "wrong" and v.answer == RIDDLES[first].told and s.riddle != first
    assert v.say == f"It was {RIDDLES[first].told}! Next one: {RIDDLES[s.riddle].question}"   # the engine tells it


def test_riddles_a_long_message_that_isnt_right_is_off_topic():
    s = riddles()
    first = s.riddle
    v = play(s, "sorry one sec, my agent is asking me for approval again")
    assert v.kind == "off" and s.turns == 0 and s.misses == 0 and s.riddle == first
    v = play(s, f"hmm let me think about it for a bit, is it {RIDDLES[first].answers[0]}")
    assert v.kind == "right"


def test_riddle_plural_answers_take_the_singular():
    s = riddles()
    s.riddle = next(i for i, r in enumerate(RIDDLES) if r.answers[0] == "footsteps")
    assert play(s, "a footstep").kind == "right"


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
    assert v.answer == RIDDLES[s.riddle].told and v.say == f"It was {RIDDLES[s.riddle].told}!"


# --- the turn limit and the guard ----------------------------------------------------------------------------------

def test_the_15th_move_ends_riddles_and_tells_the_open_answer():
    s = riddles()
    for _ in range(MAX_TURNS - 1):
        play(s, "a potato")
        play(s, "this is a long message about something else entirely, not a guess")   # never counts
    open_riddle = s.riddle
    v = play(s, "a potato")
    assert v.kind == "ended" and v.reason == "turns" and s.over and v.answer == RIDDLES[open_riddle].told
    assert v.say == f"It was {RIDDLES[open_riddle].told}!" and "over" in game_situation(s, v)


def test_a_win_on_the_20th_question_is_still_a_win():
    s = twenty()
    s.questions = MAX_QUESTIONS - 1
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


def test_guard_catches_singular_and_joined_names():
    s = twenty("headphones")
    assert guard(s, "A headphone? Nope!") in games.TEASES
    s = twenty("ice cream")
    assert guard(s, "Icecream is cold.") in games.TEASES


def test_the_mid_game_note_never_carries_the_secret():
    s = twenty()
    note = mid_game_note(s)
    assert "20 questions" in note and "banana" not in note and "20 questions left" in note
    assert '"e"' in mid_game_note(chain("apple"))
    assert "riddles" in mid_game_note(riddles())


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
        assert sit.startswith(games._RULES[kind]) and sit.endswith("Say one short, playful sentence.")
        assert "already judged" in sit and (s.secret in sit if kind == "twenty_questions" else True)
