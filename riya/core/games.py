"""Games she plays while their agents work: word chain, 20 questions and riddles. Pure: the rules and every judgement
live here (and the state in the engine); the small local model only voices the verdict in one playful sentence.
See docs/superpowers/specs/2026-10-18-riya-v0.4-agent-babysitting-design.md §2.5 and §4.4."""
import random
import re
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["word_chain", "twenty_questions", "riddles"]
GAMES: tuple[Kind, ...] = ("word_chain", "twenty_questions", "riddles")
NAMES = {"word_chain": "word chain", "twenty_questions": "20 questions", "riddles": "riddles"}
MAX_TURNS = 15  # user turns in one game
MAX_QUESTIONS = 20
RIDDLE_TRIES = 3  # wrong guesses before she tells the answer and moves on
TEASES = ("Nice try, I'm not telling~", "Hehe, my lips are sealed.", "Nope, you have to guess it yourself~")

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_START = re.compile(r"\bplay\s+(?:a\s+game\s+of\s+|a\s+|some\s+)?(word[\s-]?chain|20\s+questions|twenty\s+questions"
                    r"|riddles?)\b", re.I)
_STOP = re.compile(r"(?:ok |okay |alright )?(?:stop|quit|enough|stop the game|stop playing|let's stop|lets stop"
                   r"|i give up|give up|i quit)(?: please| now)?")
_GIVE_UP = re.compile(r"\bgive up\b")
_ACCEPT = re.compile(r"(?:yes|yeah|yea|yep|yup|sure|ok|okay|alright|absolutely|definitely|totally|let's go|lets go"
                     r"|let's do it|lets do it|let's play|why not|sounds good|i'm in|go on)\b")
_DECLINE = frozenset("no nah nope not later but don't dont maybe".split())
_ACCEPT_MAX_WORDS = 5


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _norm(text: str) -> str:
    return " ".join(_tokens(text))


def mentions(text: str, key: str | None, plural: bool = False) -> bool:
    """key (one or more words) appears in text as whole words; plural also takes a trailing s/es on its last word."""
    if not key:
        return False
    words, toks = _tokens(key), _tokens(text)
    n = len(words)
    last = {words[-1], words[-1] + "s", words[-1] + "es"} if plural else {words[-1]}
    return any(toks[i:i + n - 1] == words[:-1] and toks[i + n - 1] in last for i in range(len(toks) - n + 1))


# ---- the curated lists ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Thing:
    name: str
    aliases: tuple[str, ...]  # every name that counts as naming it, the name included


def _thing(spec: str) -> Thing:
    names = tuple(n.strip() for n in spec.split("/"))
    return Thing(names[0], names)


THINGS = tuple(_thing(s) for s in (
    "apple", "banana", "pizza", "chair", "umbrella", "bicycle/bike", "guitar", "toothbrush", "pillow",
    "cat/kitten/kitty", "dog/puppy", "elephant", "giraffe", "penguin", "laptop/computer",
    "phone/smartphone/cellphone/mobile", "book", "clock", "mirror", "spoon", "fork", "cup/mug", "car", "airplane/plane",
    "train", "boat/ship", "tree", "sun", "moon", "rainbow", "cloud", "candle", "key", "shoe/sneaker", "sock", "hat",
    "backpack/rucksack", "pencil", "scissors", "lamp", "refrigerator/fridge", "television/tv", "sandwich",
    "ice cream", "cookie/biscuit", "egg", "carrot", "watermelon", "coffee", "balloon", "kite", "football/soccer ball",
    "teddy bear/teddy", "snowman", "butterfly", "bee", "fish", "door", "bed", "camera", "headphones", "bathtub/tub",
    "toaster", "keyboard"))


@dataclass(frozen=True)
class Riddle:
    question: str
    answers: tuple[str, ...]  # keywords: any one in their guess is right; the first is the one she tells


RIDDLES = (
    Riddle("What has keys but can't open locks?", ("piano", "keyboard")),
    Riddle("What has to be broken before you can use it?", ("egg",)),
    Riddle("What gets wetter the more it dries?", ("towel",)),
    Riddle("What has hands but can't clap?", ("clock", "watch")),
    Riddle("What has a neck but no head?", ("bottle",)),
    Riddle("What has one eye but can't see?", ("needle",)),
    Riddle("What goes up but never comes down?", ("age",)),
    Riddle("What can you catch but not throw?", ("cold",)),
    Riddle("What has many teeth but can't bite?", ("comb", "zipper")),
    Riddle("What has words but never speaks?", ("book",)),
    Riddle("What runs but never walks?", ("river", "water", "tap")),
    Riddle("What has a head and a tail but no body?", ("coin",)),
    Riddle("The more you take, the more you leave behind. What are they?", ("footsteps", "steps", "footprints")),
    Riddle("What can travel around the world while staying in a corner?", ("stamp",)),
    Riddle("What has legs but doesn't walk?", ("table", "chair")),
    Riddle("What is full of holes but still holds water?", ("sponge",)),
    Riddle("What belongs to you, but other people use it more than you do?", ("name",)),
    Riddle("What can fill a room but takes up no space?", ("light", "air")),
    Riddle("What has a thumb and four fingers but isn't alive?", ("glove",)),
    Riddle("What month of the year has 28 days?", ("all", "every", "each")),
    Riddle("What is always in front of you but can't be seen?", ("future",)),
    Riddle("What has cities but no houses, forests but no trees, and water but no fish?", ("map",)),
    Riddle("I speak without a mouth and hear without ears. What am I?", ("echo",)),
    Riddle("What comes once in a minute, twice in a moment, but never in a thousand years?", ("m",)),
    Riddle("What goes through cities and fields but never moves?", ("road", "street")),
    Riddle("What kind of band never plays music?", ("rubber",)),
    Riddle("What has a bottom at the top?", ("leg", "legs")),
    Riddle("What can you hold in your left hand but not in your right?", ("elbow", "right hand")),
    Riddle("What gets bigger the more you take away from it?", ("hole",)),
    Riddle("What is so fragile that saying its name breaks it?", ("silence",)),
    Riddle("What has four wheels and flies?", ("garbage", "trash", "rubbish", "bin")),
    Riddle("What building has the most stories?", ("library",)),
)

_WORDS = """
apple anchor arrow artist acorn alarm album angel ankle antelope apron attic avocado axe award
banana basket bridge button butter blanket bottle breeze bucket bubble badge bamboo beach bicycle biscuit
candle carrot castle cherry circle cloud cookie copper cotton crayon crystal cushion camera cactus canyon
daisy dragon drum desert dolphin doorbell diamond dinner donkey doughnut dentist dew dice duck dune
eagle earth elbow engine envelope eraser echo emerald elephant evening easel egg ember eel elf
feather forest falcon fence fiddle flame flower fossil fountain frog fudge funnel fabric ferry fig
garden giraffe glove goose grape guitar gravel ginger glacier goblin gondola gate gem globe gecko
hammer harbor helmet honey horse hotel hammock hazel heart hedge hippo hill hive hook hurricane
igloo island iron ivory insect icicle ink ivy iceberg idea image inch index inn icon
jacket jaguar jam jar jelly jewel jigsaw journal juice jungle jasmine jeep jet joke judge
kangaroo kettle kitten kite koala kayak kernel key kiwi knight knot keyboard kingdom kitchen kale
ladder lantern lemon lettuce library lizard lobster locket lotus lily lake leaf lime llama lamp
magnet mango marble meadow melon mirror monkey muffin mushroom mountain maple medal mitten moth map
napkin necklace needle nest noodle notebook nutmeg narwhal nectar net nickel night nail nose novel
ocean octopus olive onion orange orchard otter owl oven oyster oak oasis oar opal orbit
paddle panda parrot peach pencil pepper piano pillow pirate planet pocket pumpkin puzzle pebble penguin
queen quilt quiz quail quartz
rabbit radio rainbow raven ribbon river robot rocket ruby rose ruler raft raisin reef rhino
saddle salmon sandal scarf shell spider spoon squirrel star sugar sunflower sweater seal snail sock
table tiger tomato turtle tunnel teapot thunder ticket toast tulip tractor trumpet tent toad tree
umbrella unicorn uniform utensil urchin udon ukulele universe upstairs urn
valley vase velvet violin volcano vulture vanilla van village vine visor voyage vest viper vinegar
wagon walnut walrus wand wave whale wheel whistle willow window wizard wolf worm wreath waffle
xylophone
yacht yak yard yarn yeti yogurt yolk yoyo yam yellow yodel youth
zebra zipper zoo zucchini zero
"""
WORDS: dict[str, tuple[str, ...]] = {}
for _w in _WORDS.split():
    WORDS[_w[0]] = WORDS.get(_w[0], ()) + (_w,)
_STARTERS = "abcdefghlmprstw"  # her opening word starts with an easy letter


# ---- state and verdicts --------------------------------------------------------------------------------------------

@dataclass
class GameState:
    kind: Kind
    rng: random.Random
    turns: int = 0  # user turns played (the start doesn't count)
    over: bool = False
    secret: str | None = None  # 20 questions: what she thought of
    aliases: tuple[str, ...] = ()
    questions: int = 0
    word: str | None = None  # word chain: the last word said
    used: set[str] = field(default_factory=set)
    riddle: int | None = None  # riddles: the one being asked (index into RIDDLES)
    left: list[int] = field(default_factory=list)  # the ones still to ask, in order
    misses: int = 0
    solved: int = 0


@dataclass
class Verdict:
    """kind: start, ok (a fine move), invalid (word chain: try again), right / wrong (a riddle guess), win (the user
    won), loss (the user lost) or ended (stopped, gave up, or the turn limit). line is the engine's judgement for the
    model; say is a line the engine says itself right after hers, unless key was already in what she said."""
    kind: Literal["start", "ok", "invalid", "right", "wrong", "win", "loss", "ended"]
    line: str
    reason: str | None = None  # invalid: not a word / wrong letter / used; ended: stopped / gave up / turns
    word: str | None = None  # word chain: her word
    riddle: str | None = None  # riddles: the riddle she asks next
    answer: str | None = None  # an answer or secret she may now say
    say: str | None = None
    key: str | None = None

    @property
    def over(self) -> bool:
        return self.kind in ("win", "loss", "ended")


def _a(thing: str) -> str:
    return thing if thing.endswith("s") else ("an " if thing[0] in "aeiou" else "a ") + thing


def new_game(kind: Kind, rng: random.Random) -> GameState:
    s = GameState(kind, rng)
    if kind == "twenty_questions":
        t = rng.choice(THINGS)
        s.secret, s.aliases = t.name, t.aliases
    elif kind == "riddles":
        s.left = rng.sample(range(len(RIDDLES)), len(RIDDLES))
        s.riddle = s.left.pop()
    else:
        s.word = rng.choice(WORDS[rng.choice(_STARTERS)])
        s.used = {s.word}
    return s


def opening(s: GameState) -> Verdict:
    """The first move, hers: her first word, the secret picked, or the first riddle."""
    if s.kind == "word_chain":
        return Verdict("start", f'You start with the word "{s.word}". Next they need a word starting with '
                                f'"{_last(s.word)}".', word=s.word, say=f"I'll start: {s.word}!", key=s.word)
    if s.kind == "twenty_questions":
        return Verdict("start", "You've picked your secret thing. Tell them you've got something in mind and they "
                                "can start asking yes/no questions.")
    q = RIDDLES[s.riddle].question
    return Verdict("start", "Say you're ready with a riddle. Don't ask it yourself: it's asked right after you.",
                   riddle=q, say=q)


def _last(word: str) -> str:
    return next(c for c in reversed(word) if c.isalpha())


def _is_word(w: str) -> bool:
    return w.isalpha() and len(w) >= 2 and any(c in "aeiouy" for c in w) and not re.search(r"(.)\1\1", w)


def _chain(s: GameState, text: str) -> Verdict:
    need, toks = _last(s.word), _tokens(text)
    word = toks[0] if len(toks) == 1 else next((t for t in reversed(toks) if t.startswith(need)),
                                               toks[-1] if toks else "")
    bad = ((f'"{word}" isn\'t a real word.', "not a word") if not _is_word(word)
           else (f'"{word}" doesn\'t start with "{need}".', "wrong letter") if word[0] != need
           else (f'"{word}" was already used.', "used") if word in s.used else None)
    if bad and s.turns >= MAX_TURNS:
        return _ended(s, bad[0])
    if bad:
        return Verdict("invalid", f'{bad[0]} They should try again with a word starting with "{need}".',
                       reason=bad[1])
    s.used.add(word)
    s.word = word
    if s.turns >= MAX_TURNS:
        return _ended(s, f'Their word "{word}" is fine.')
    options = [w for w in WORDS.get(_last(word), ()) if w not in s.used]
    if not options:
        s.over = True
        return Verdict("win", f'Their word "{word}" is fine, and you can\'t think of any word starting with '
                              f'"{_last(word)}": you give up, they win!')
    mine = s.rng.choice(options)
    s.used.add(mine)
    s.word = mine
    return Verdict("ok", f'Their word "{word}" is fine. Your word is "{mine}"; next they need a word starting with '
                         f'"{_last(mine)}".', word=mine, say=f"My word: {mine}!", key=mine)


def _twenty(s: GameState, text: str) -> Verdict:
    if any(mentions(text, a, plural=True) for a in s.aliases):
        s.over = True
        return Verdict("win", f"They guessed it: it's {_a(s.secret)}! They win.", answer=s.secret)
    s.questions += 1
    if s.questions >= MAX_QUESTIONS:
        s.over = True
        return Verdict("loss", f"That was their {MAX_QUESTIONS}th question and they didn't get it, so you win. "
                               f"Tell them it was {_a(s.secret)}.", answer=s.secret,
                       say=f"It was {_a(s.secret)}!", key=s.secret)
    if s.turns >= MAX_TURNS:
        return _ended(s, "")
    return Verdict("ok", f"That was question {s.questions} of {MAX_QUESTIONS}; {MAX_QUESTIONS - s.questions} left.")


def _riddles(s: GameState, text: str) -> Verdict:
    r = RIDDLES[s.riddle]
    right = any(mentions(text, a, plural=True) for a in r.answers)
    s.misses = 0 if right else s.misses + 1
    if right:
        s.solved += 1
        line = f'Their answer is right, it\'s "{r.answers[0]}"!'
    elif s.misses >= RIDDLE_TRIES:
        line = f'Their guess is wrong again; tell them the answer was "{r.answers[0]}".'
    else:
        if s.turns >= MAX_TURNS:
            return _ended(s, "Their guess is wrong.")
        return Verdict("wrong", "Their guess is wrong. Tease them gently and let them guess again; don't give the "
                                "answer.")
    answer = r.answers[0]
    if s.turns >= MAX_TURNS:
        return _ended(s, line, answer)
    if not s.left:
        s.over = True
        return Verdict("win", f"{line} That was your last riddle: they solved {s.solved}. They win!", answer=answer)
    s.riddle, s.misses = s.left.pop(), 0
    q = RIDDLES[s.riddle].question
    return Verdict("right" if right else "wrong", f"{line} The next riddle is asked right after you; don't ask it "
                                                  "yourself.", answer=answer, riddle=q, say=q)


def _ended(s: GameState, line: str, answer: str | None = None) -> Verdict:
    """The turn limit: the move was judged (line), and the game is over; the secret or the riddle's answer is told."""
    s.over = True
    say = tell = None
    if s.kind == "twenty_questions":
        answer, say, tell = s.secret, f"It was {_a(s.secret)}!", f"Tell them it was {_a(s.secret)}."
    elif s.kind == "riddles":
        answer = answer or RIDDLES[s.riddle].answers[0]
        tell = f'Tell them the answer was "{answer}".'
    end = f"That was turn {MAX_TURNS}, so the game is over now: thank them for playing."
    return Verdict("ended", " ".join(filter(None, [line, end, tell])), reason="turns", answer=answer, say=say,
                   key=s.secret if say else None)


def play(s: GameState, text: str) -> Verdict:
    """The user's move: advances the state and returns the engine's verdict."""
    s.turns += 1
    return {"word_chain": _chain, "twenty_questions": _twenty, "riddles": _riddles}[s.kind](s, text)


def stop(s: GameState, text: str) -> Verdict:
    """The user stopped (or gave up): the game ends; giving up hears the secret or the riddle's answer."""
    s.over = True
    if _GIVE_UP.search(_norm(text)):
        if s.kind == "twenty_questions":
            return Verdict("ended", f"They gave up. Tell them it was {_a(s.secret)}.", reason="gave up",
                           answer=s.secret, say=f"It was {_a(s.secret)}!", key=s.secret)
        if s.kind == "riddles":
            a = RIDDLES[s.riddle].answers[0]
            return Verdict("ended", f'They gave up. Tell them the answer was "{a}".', reason="gave up", answer=a)
        return Verdict("ended", "They gave up, so you win this one.", reason="gave up")
    return Verdict("ended", "They want to stop the game. Say okay and thank them for playing.", reason="stopped")


# ---- the words around it -------------------------------------------------------------------------------------------

_RULES = {
    "word_chain": "You're playing word chain with the user: each word must start with the last letter of the word "
                  "before, and no word may repeat.",
    "twenty_questions": "You're playing 20 questions with the user: you thought of something and they ask yes/no "
                        "questions to guess it.",
    "riddles": "You're playing riddles with the user: you ask a riddle and they guess the answer.",
}


def game_situation(s: GameState, v: Verdict) -> str:
    """The rules, the secret while it's live (or when she may now tell it), and this turn's verdict, for the model.
    Never written to the transcript."""
    parts = [_RULES[s.kind]]
    if s.kind == "twenty_questions" and not s.over:
        parts.append(f"Your secret thing is {_a(s.secret)}. If they asked a yes/no question, answer it truthfully "
                     "with yes or no and maybe a tiny hint, but never say what the thing is.")
    parts += [f"The game engine already judged this turn; don't change it: {v.line}"]
    if v.word and v.kind in ("start", "ok"):
        parts.append(f'Say your word "{v.word}" in your sentence.')
    parts.append("Say one short, playful sentence.")
    return " ".join(parts)


def guard(s: GameState, sentence: str) -> str:
    """While 20 questions is live, a sentence of hers naming the secret is swapped for a teasing refusal."""
    if s.kind != "twenty_questions" or s.over or not any(mentions(sentence, a, plural=True) for a in s.aliases):
        return sentence
    return s.rng.choice(TEASES)


def is_stop(text: str) -> bool:
    """The whole message is a stop phrase: stop, quit, enough, stop the game, let's stop, i give up..."""
    return bool(_STOP.fullmatch(_norm(text)))


def is_accept(text: str) -> bool:
    """A short yes: sure, yeah, ok, let's go... (anything with a no, not or later in it isn't)."""
    toks = _tokens(text)
    said = " ".join(toks)
    return bool(toks) and len(toks) <= _ACCEPT_MAX_WORDS and not _DECLINE & set(said.replace("why not", "").split()) \
        and bool(_ACCEPT.match(said))


def detect_start(text: str, offered: str | None) -> Kind | None:
    """The game they asked for ("let's play word chain", "can we play 20 questions", "play riddles"), else the offered
    one when this is a yes to the offer, else None."""
    if m := _START.search(text):
        g = m.group(1).lower()
        return "word_chain" if g.startswith("word") else "riddles" if g.startswith("riddle") else "twenty_questions"
    return offered if offered in GAMES and is_accept(text) else None  # type: ignore[return-value]


def offer_situation(kind: Kind) -> str:
    return f"Their agent is busy working, so offer to play {NAMES[kind]} with you while they wait."


def offer_line(kind: Kind) -> str:
    return f"Wanna play {NAMES[kind]} while your agent works?"


def offers(text: str, kind: Kind) -> bool:
    """The line names the game ("word chain", "20 questions" or "twenty questions", "riddle")."""
    names = {"word_chain": ("word chain",), "twenty_questions": ("20 questions", "twenty questions"),
             "riddles": ("riddle",)}[kind]
    return any(mentions(text, n, plural=True) for n in names)
