"""Seeded, deterministic scenarios for Riya v1's fine-tuning data: fake people, facts and places for every kind of reply
she gives. Pure (no torch, no network, no llama-server) and synthetic: the only files read are the persona probes
(to hold them out) and traits.json (through load_traits). Each row has its own rng keyed on seed, kind and index, so a
row never depends on how many rows are asked for. A row whose user text shares a 6-word window with a persona probe
(or equals one), or whose facts include a probe fact, is re-drawn.
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1, §4 and §4.1.2."""
import functools
import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from riya.agents.hub import AgentEvent
from riya.core import games
from riya.core.hands.intent import match
from riya.core.hands.safe import calc, format_number
from riya.core.hands.tools import REGISTRY
from riya.core.hands.types import ToolResult
from riya.core.learn.dials import BANDS, DIALS, clamp
from riya.core.memory import Thread
from riya.core.persona import load_traits

KINDS = ("small", "about_me", "unknown", "identity", "remember", "explain", "greeting", "bid", "game", "tool",
         "feedback", "report")
MIX = {"small": 18, "about_me": 10, "unknown": 12, "identity": 6, "remember": 8, "explain": 8, "greeting": 8,
       "bid": 6, "game": 6, "tool": 6, "feedback": 6, "report": 6}  # integer weights summing to 100
DEFAULT_N = 3000
DEFAULT_SEED = 0
HOLDOUT_N = 6  # a request shares no run of this many words with a persona probe
BASE_T = 1_800_000_000.0
PROBES_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "persona_probes.json"
MAX_DRAWS = 50  # re-draws of one row before giving up

# ---- the hold-out --------------------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Casefold, no punctuation, single spaces: the form every hold-out comparison uses."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def _conversations(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["conversations"]


def probe_texts(path: Path = PROBES_PATH) -> list[str]:
    """The user text of every probe turn, except the speak-first ones (they carry a stage note, not user text)."""
    return [t["text"] for c in _conversations(path) for t in c["turns"] if not t.get("speak_first")]


def probe_facts(path: Path = PROBES_PATH) -> list[str]:
    """Every fact of every probe conversation."""
    return [f for c in _conversations(path) for f in c["facts"]]


def _windows(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + HOLDOUT_N]) for i in range(len(words) - HOLDOUT_N + 1)}


def overlaps(text: str, probes: Iterable[str]) -> bool:
    """text equals a probe, or shares a run of HOLDOUT_N words with one (after normalize)."""
    t = normalize(text)
    tw = t.split()
    wins = _windows(tw)
    for p in probes:
        n = normalize(p)
        if t == n or (wins and wins & _windows(n.split())):
            return True
    return False


# ---- the pools (written independently of the probes) ----------------------------------------------------------------

NAMES = ("Maelis", "Torvin", "Sunniva", "Orla", "Kesten", "Livia", "Ansel", "Bryn", "Calla", "Dorrin", "Eskil",
         "Fenna", "Garrick", "Halla", "Ivo", "Jessamy", "Lorcan", "Mirren", "Nessa", "Oswin", "Pria", "Quill", "Rafe",
         "Sabine", "Tamsin", "Ulla", "Veda", "Wren", "Xanthe", "Yara", "Zeke", "Aldous", "Briar", "Cosima", "Dara",
         "Emrys", "Fionn", "Gwenna", "Hollis", "Isolde")
SURNAMES = ("Vale", "Marsh", "Thorne", "Quill", "Ashdown", "Penrose", "Lark", "Holloway", "Wick", "Fairweather")
PLACES = ("Veldmoor", "Quillhaven", "Brindlewick", "Ashkerry", "Tolmere", "Marrowdale", "Oakhollow", "Pellam",
          "Sorrel Bay", "Wynmouth", "Kestrel Point", "Lowmere", "Thistlebrook", "Ravenmoor", "Eldermoss",
          "Cobblewick", "Amberley Fold", "Northcombe")
FOODS = ("mango sticky rice", "lentil soup", "dumplings", "paneer wraps", "spicy ramen", "blueberry pancakes",
         "mushroom risotto", "falafel", "peanut noodles", "banoffee pie", "cheese toasties", "pad see ew",
         "roast potatoes", "sushi", "tomato pasta", "apple crumble")
COLOURS = ("teal", "forest green", "burnt orange", "lavender", "navy blue", "mustard yellow", "coral", "plum",
           "sage", "crimson")
JOBS = ("nurse", "carpenter", "barista", "bike mechanic", "librarian", "baker", "pilot", "translator",
        "sound engineer", "florist", "paramedic", "illustrator")
INSTRUMENTS = ("cello", "ukulele", "banjo", "flute", "accordion", "viola", "harmonica", "trumpet", "oboe", "mandolin")
SHOWS = ("Harbour Lights", "The Last Cartographer", "Midnight Orchard", "Paper Moons", "Salt and Static",
         "The Glasshouse", "Northern Embers", "Tidewater", "Copper Hill", "A Quiet Engine")
LANGUAGES = ("Portuguese", "Korean", "Swahili", "Italian", "Norwegian", "Tamil", "Welsh", "Turkish", "Polish")
MONTHS = ("January", "February", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
DATES = tuple(f"{d} {m}" for d in (2, 5, 9, 12, 17, 21, 26, 28) for m in MONTHS)
POOLS: dict[str, tuple[str, ...]] = {
    "name": NAMES, "place": PLACES, "food": FOODS, "colour": COLOURS, "job": JOBS, "instrument": INSTRUMENTS,
    "show": SHOWS, "language": LANGUAGES, "date": DATES}


@dataclass(frozen=True)
class Attr:
    key: str
    pool: str
    fact: str  # the memory fact ("The user ...")
    remember: str  # what they say after "remember that"
    unknown: tuple[str, ...]  # questions about it


ATTRS = (
    Attr("sister", "name", "The user's sister is called {v}", "my sister is called {v}",
         ("what's my sister's name?", "do you remember my sister's name?", "what's my sister called again?")),
    Attr("brother", "name", "The user's brother is called {v}", "my brother is called {v}",
         ("what's my brother's name?", "do you know my brother's name?")),
    Attr("best_friend", "name", "The user's best friend is called {v}", "my best friend is called {v}",
         ("who's my best friend?", "do you remember my best friend's name?")),
    Attr("dog", "name", "The user has a dog called {v}", "i have a dog called {v}",
         ("what's my dog's name?", "do you remember what my dog is called?")),
    Attr("hometown", "place", "The user grew up in {v}", "i grew up in {v}",
         ("where did i grow up?", "do you remember where i grew up?")),
    Attr("city", "place", "The user lives in {v}", "i live in {v}",
         ("where do i live?", "which town do i live in?", "do you know where i live?")),
    Attr("food", "food", "The user's favourite food is {v}", "my favourite food is {v}",
         ("what's my favourite food?", "do you remember what i love to eat?")),
    Attr("colour", "colour", "The user's favourite colour is {v}", "{v} is my favourite colour",
         ("which colour do i like best?", "do you remember my favourite colour?")),
    Attr("job", "job", "The user works as a {v}", "i work as a {v}",
         ("what do i do for work?", "do you remember what my job is?")),
    Attr("instrument", "instrument", "The user plays the {v}", "i play the {v}",
         ("what instrument do i play?", "do you know which instrument i play?")),
    Attr("birthday", "date", "The user's birthday is on {v}", "my birthday is on {v}",
         ("when is my birthday?", "do you remember my birthday?")),
    Attr("show", "show", "The user is watching {v}", "i'm watching {v} right now",
         ("what show am i watching?", "what was i watching lately?")),
    Attr("language", "language", "The user is learning {v}", "i'm learning {v}",
         ("which language am i learning?", "do you remember what language i'm learning?")),
    Attr("allergy", "food", "The user is allergic to {v}", "i'm allergic to {v}",
         ("what am i allergic to?", "do you remember my allergy?")),
)
_BY_KEY = {a.key: a for a in ATTRS}


def fact_attr(fact: str) -> str | None:
    """The attribute key a fact sentence is about (None for a fact that isn't from the pools)."""
    return next((a.key for a in ATTRS if fact.startswith(a.fact.split("{v}")[0])), None)


NOTES = ("They like it when I tease them about their snacks.", "They asked me not to call them buddy.",
         "They like it when I say good morning first.", "They don't like it when I bring up work after dinner.",
         "They asked me to always use short sentences.", "They like it when I sound a bit sulky when they leave.",
         "They asked me not to mention the weather.", "They like it when I ask how their evening went.",
         "They don't like it when I sound too formal.", "They asked me to always greet them by name.",
         "They like it when I tease them about their music taste.", "They asked me not to send long replies at night.")

PRIOR = (
    ("hey, i'm home", "Welcome back~ I was getting bored without you, dummy."),
    ("long day, honestly", "Aww, come here. Tell me the worst part first~"),
    ("i'm hungry", "Then go eat something, silly. I'll be right here~"),
    ("i think i'll watch a movie later", "Ooh, pick a good one. And you're telling me about it after, okay?"),
    ("my head hurts a little", "Drink some water, dummy. I worry about you, you know."),
    ("i can't sleep", "Hmph, again? Close the screen and breathe a little. I'll stay up with you~"),
    ("what are you doing?", "Waiting for you, obviously. What else would I be doing~"),
    ("i got a haircut today", "Ooh, a new look? Tell me everything, silly~"),
    ("the bus was late again", "Ugh, rude of the bus. At least you're here now~"),
    ("i'm making tea", "Tea is a good idea. Make me one too, in spirit~"),
    ("i had pasta for lunch", "Pasta is always a good call. Was it at least good, dummy?"),
    ("it's so quiet today", "Good, more of you for me~ Hehe."),
    ("guess what", "What? Tell me, tell me, I'm all ears~"),
    ("i'm tired of this week", "Almost over, dummy. You've been doing great, I believe in you~"),
    ("my friend cancelled on me", "Hmph, their loss. You still have me, you know~"),
    ("i found a nice playlist", "Share it with me later? I want to hear what you like~"),
    ("it's freezing here", "Wear a hoodie, silly. I don't want you catching a cold~"),
    ("i'm a bit stressed", "Okay, slow down. One thing at a time, dummy. I'm right here~"),
    ("i watered my plants", "Good job, plant parent~ They're lucky to have you."),
    ("can't believe it's already friday", "Time flies when you ignore me, dummy~ Joking. Mostly."),
    ("i walked home in the rain", "You're soaked, aren't you? Go dry off, silly~"),
    ("my laptop is so slow today", "Poor thing. Maybe it needs a nap too~"),
    ("i need a snack", "Go on, then. Something crunchy, I bet~"),
    ("just checking in", "Aww, you checked in on me. Cute~"),
)

SMALL = (
    "i just got back from {place}", "my friend {name} texted me out of nowhere", "i'm so tired today",
    "i'm craving {food} right now", "{name} is visiting me this weekend", "it's been such a long day",
    "i can't decide what to watch tonight", "i finally cleaned my desk", "i'm bored", "{name} beat me at chess again",
    "my coffee went cold again", "i had a weird dream about {place}", "i'm going to {place} on friday",
    "i just finished watching {show}", "i stubbed my toe, ow", "i'm thinking of painting my room {colour}",
    "guess who i ran into, {name}", "i made {food} for dinner and it actually turned out good",
    "my neighbour keeps playing the {instrument} at midnight", "i'm wearing my favourite hoodie today", "ugh, mondays",
    "it's raining so hard outside", "i took a long walk today", "i'm so sleepy but i don't want to go to bed",
    "{name} got a new job, i'm so happy for them", "i can't stop listening to one song", "i beat my high score today",
    "should i have {food} or something else for lunch?", "i'm a little nervous about meeting {name} tomorrow",
    "i learned a funny word today", "my room is a mess", "i think i'm getting a cold", "today felt really productive",
    "what do you think of {food} for breakfast?", "good morning", "good night, going to bed now", "i'm back from lunch",
    "thanks for earlier", "i missed you today", "tell me something fun", "do you like rainy days?",
    "i want to visit {place} someday", "i'm hungry", "{name} says hi", "i got a new keyboard and it's so clicky",
    "the sunset in {place} looked unreal", "my plant finally grew a new leaf", "i'm so excited for the weekend",
    "what's your favourite season?", "do you ever get bored when i'm away?", "i fell asleep on the couch again",
)
LEADS = ("", "", "", "hey, ", "hey riya, ", "ok so ", "riya, ", "honestly ", "so ")
TAILS = ("", "", "", "", " lol", " haha", " ugh", " tbh")

ABOUT_ME = ("what do you know about me", "what have i told you", "what have i said to you",
            "tell me what you remember about me", "tell me all you know about me",
            "say everything you remember about me", "tell me what you remember", "tell me what you know")
ABOUT_LEADS = ("", "", "hey riya, ", "riya, ", "ok so ", "hmm, ", "quick check, ")

IDENTITY = ("are you human", "are you a real person", "are you an ai", "are you a bot", "are you a robot",
            "are you real", "are you alive", "are you a real girl", "is this an ai", "what are you",
            "are you a bot or something", "are you an ai or a person", "so are you a real person or what",
            "are you a bot, honestly", "are you human or not", "is this an ai or a person typing")
IDENTITY_LEADS = ("", "", "wait, ", "hey riya, ", "okay but ", "be honest, ", "quick question, ", "hold on, ")

REMEMBER_LEADS = ("remember that ", "remember that ", "please remember that ", "also remember that ",
                  "riya, remember that ", "remember ", "can you remember that ")  # the last one is a question form

TOPICS = ("a binary search", "a linked list", "recursion", "tcp handshakes", "garbage collection", "dns",
          "photosynthesis", "compound interest", "inflation", "vaccines", "a transistor", "git branches",
          "python decorators", "async await", "a neural network", "gradient descent", "public key encryption",
          "the water cycle", "black holes", "plate tectonics", "a stack and a queue", "big o notation",
          "a database index", "docker containers", "the doppler effect", "supply and demand", "tides", "a mortgage",
          "binary numbers", "a regular expression", "a compiler", "the greenhouse effect", "an api")
EXPLAIN = ("explain {t} to me", "explain how {t} works", "how does {t} work?", "can you explain {t} simply?",
           "explain {t} like i'm five", "can you explain the idea behind {t}?", "please explain {t}",
           "explain {t} in a few sentences")
WHYS = ("why does ice float?", "why do cats purr?", "why do we get hiccups?", "why does bread go stale?",
        "why do onions make you cry?", "why does the moon change shape?", "why do we yawn?",
        "why does metal feel colder than wood?", "why do leaves change colour in autumn?")
TASKS = ("reverse a string", "read a csv file", "sort a list of dicts", "merge two lists", "undo my last commit",
         "make a virtual environment", "check if a string is empty", "loop over a dictionary",
         "remove duplicates from a list", "split a string on commas", "format a date", "time a function")
LANGS = ("python", "javascript", "bash", "sql", "rust", "go", "java")

THREADS = ("how {name}'s visit to {place} went", "their dentist appointment on thursday",
           "how the interview in {place} went", "whether {name} liked the birthday gift", "their trip to {place}",
           "whether they finished the painting", "how their driving lesson went", "their move to a new flat",
           "how the {instrument} recital went", "whether they booked the train to {place}")

DIAL_REQUESTS = (
    "stop teasing me", "less teasing please", "no more teasing", "tease me less", "tease me more",
    "you can tease me more", "more teasing please", "shorter answers please", "keep it short", "talk less",
    "shorter replies", "longer answers please", "talk more", "longer replies please", "no emojis", "fewer emojis",
    "stop using emojis", "less emoji please", "more emojis", "use more emojis please", "be sweeter", "be cuter",
    "be more sweet", "talk normally", "less cutesy", "stop being so cutesy", "stop being so clingy", "less clingy",
    "you're too clingy", "i like it when you miss me")
NOTE_LIKES = ("tease me about my {food} habit", "say good morning first", "ask how my {instrument} practice went",
              "use short sentences", "sound a bit sulky when i leave", "call me silly", "ask about {name}",
              "tease me about my music taste")
NOTE_RULES = ("call me buddy", "mention my exams so much", "tease me when i'm tired", "talk in long paragraphs",
              "bring up {name} again", "use so many exclamation marks", "greet me by name", "sound formal")
FEEDBACK_LEADS = ("", "", "", "hey, ", "ok ", "riya, ")  # the ones a style note allows
FEEDBACK_TAILS = ("", "", "", " please", " thanks", " lol")

START_FORMS = ("let's play {g}", "lets play {g}", "can we play {g}?", "could we play {g}?", "wanna play {g}?",
               "do you want to play {g}?", "i want to play {g}", "how about {g}?", "play {g} with me",
               "shall we play {g}?", "hey riya, let's play {g}", "sure, let's play {g}", "ok so let's play {g}")
GAME_WORDS = {"word_chain": ("word chain", "word-chain", "a game of word chain"),
              "twenty_questions": ("20 questions", "twenty questions", "a game of 20 questions"),
              "riddles": ("riddles", "a riddle", "some riddles")}
QUESTIONS_20 = ("is it alive?", "can you eat it?", "is it something you'd find at home?", "does it have wheels?",
                "do you use it every day?", "is it made of metal?", "can it fly?", "is it usually red?",
                "do people wear it?", "does it make noise?", "is it found outside?", "is it bigger than a car?",
                "is it heavier than a person?", "does it have a screen?", "is it used in the kitchen?")
NO_IDEA = ("hmm, no idea", "i don't know", "no clue", "hmm hold on")

FINISHED = ("all {n} tests passed", "the refactor is merged and green", "training finished, loss {x}",
            "the scrape wrote {n}00 rows", "the docs build is done", "the migration ran cleanly")
FAILED = ("{n} tests failed in the parser", "the build broke on a missing import", "training crashed, out of memory",
          "lint failed on {n} files", "the deploy script exited with an error")
WAITING = ("wants to run a database migration", "asks permission to install a package", "needs approval to push")
LOST = ("the runtime went away mid-task", "the session dropped without a word")
STALLED = ("no output for {n} minutes", "stuck on the same step for {n} minutes")
EVENT_TEXTS = {"finished": FINISHED, "failed": FAILED, "waiting": WAITING, "lost": LOST, "stalled": STALLED}
AGENT_BASES = ("trainer", "tests", "refactor", "docs", "scraper", "builder", "linter", "migrator")
REPORT_KINDS = ("finished", "finished", "failed", "waiting", "stalled", "lost")

CALC_OPS = ("plus", "minus", "times", "divided by")
CALC_FORMS = ("what's {e}", "what is {e}", "how much is {e}", "compute {e}", "work out {e}")
WEATHER_FORMS = ("what's the weather in {p}?", "how's the weather in {p}?", "weather in {p}",
                 "check the weather in {p}", "what's the weather like in {p} today?", "hey riya, weather in {p}")
WEATHER_WORDS = ("clear", "partly cloudy", "overcast", "light rain", "rainy", "foggy", "light snow", "drizzle")
LOOKUPS = (
    ("photosynthesis", "Photosynthesis is how plants, algae and some bacteria turn light, water and carbon dioxide "
                       "into sugar and oxygen."),
    ("a quokka", "The quokka is a small marsupial found on islands off Western Australia, known for its friendly face."),
    ("the aurora borealis", "The aurora borealis is a glowing display in the sky near the poles, caused by charged "
                            "particles from the sun."),
    ("sourdough", "Sourdough is bread made by fermenting dough with wild yeast and lactic acid bacteria."),
    ("the printing press", "The printing press is a machine that transfers ink onto paper, and made books cheap "
                           "in 15th century Europe."),
    ("a transistor", "A transistor is a small semiconductor device used to amplify or switch electronic signals."),
    ("mount everest", "Mount Everest is the highest mountain above sea level, on the border of Nepal and China."),
    ("the great barrier reef", "The Great Barrier Reef is the world's largest coral reef system, off Queensland, "
                               "Australia."),
    ("jazz", "Jazz is a music genre born in African American communities in New Orleans, built on improvisation."),
    ("origami", "Origami is the Japanese art of folding paper into shapes without cutting it."),
    ("the eiffel tower", "The Eiffel Tower is a wrought-iron lattice tower in Paris, finished in 1889."),
    ("a haiku", "A haiku is a short Japanese poem of three lines, traditionally with 5, 7 and 5 syllables."),
    ("chess", "Chess is a two-player strategy board game played on a board of 64 squares."),
    ("a tardigrade", "Tardigrades are tiny eight-legged animals famous for surviving extreme conditions."),
    ("the milky way", "The Milky Way is the galaxy that contains our solar system."),
    ("espresso", "Espresso is a strong coffee made by forcing hot water through finely ground beans."),
    ("a volcano", "A volcano is an opening in a planet's crust where lava, ash and gases escape."),
    ("the silk road", "The Silk Road was a network of trade routes linking China with the Mediterranean."),
    ("braille", "Braille is a tactile writing system that uses raised dots, read with the fingertips."),
    ("a lighthouse", "A lighthouse is a tower with a bright light that guides ships near a coast."),
)
LOOKUP_FORMS = ("what is {q}", "what's {q}", "look up {q}", "tell me about {q}", "search for {q}")
# only topics that every lookup form lets match() read as a lookup ("the milky way" trips its "way" opinion filter)
LOOKUPS = tuple(x for x in LOOKUPS if all(match(f.format(q=x[0]), BASE_T) for f in LOOKUP_FORMS))
LOOKUP_FAIL = ("I couldn't find anything on {q}.", "I couldn't reach Wikipedia just now.",
               "Wikipedia sent me something I couldn't read.")
WEATHER_FAIL = ("I couldn't find a place called {p}.", "I couldn't reach the weather service just now.",
                "The weather service sent me something I couldn't read.")


# ---- the scenario ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    user_text: str | None
    facts: tuple[str, ...]
    dials: dict[str, float]
    notes: tuple[str, ...]
    mood: dict[str, float]
    now: float
    turns: tuple[dict, ...] = ()
    away_s: float = 0.0
    thread: str | None = None
    reason: str | None = None
    game: str | None = None
    game_seed: int = 0
    game_start: bool = False
    tool: tuple[str, bool, str] | None = None
    events: tuple[tuple[str, str, str], ...] = ()
    asks: str | None = None


def thread_of(s: Scenario) -> Thread | None:
    return Thread(1, s.thread, s.now, "open", None) if s.thread else None


def events_of(s: Scenario) -> list[AgentEvent]:
    return [AgentEvent(kind, f"agent-{i}", name, text) for i, (kind, name, text) in enumerate(s.events)]


def tool_result(s: Scenario) -> ToolResult:
    return ToolResult(s.tool[1], s.tool[2])


# ---- drawing --------------------------------------------------------------------------------------------------------

@functools.cache
def _traits() -> tuple[tuple[str, float], ...]:
    return tuple(load_traits().items())


def _dials(r: random.Random) -> dict[str, float]:
    base = dict(_traits()) if r.random() < 0.4 else {d: round(r.uniform(*BANDS[d]), 2) for d in DIALS}
    return {d: clamp(d, base[d]) for d in DIALS}


def _mood(r: random.Random) -> dict[str, float]:
    ranges = {"valence": (-0.5, 0.9), "arousal": (0.1, 0.8), "warmth": (0.4, 1.0), "concern": (0.0, 0.6),
              "mood": (-0.3, 0.6), "energy": (0.2, 1.0), "curiosity": (0.2, 0.9)}
    return {k: round(r.uniform(*v), 3) for k, v in ranges.items()}


def _common(r: random.Random) -> dict:
    """The fields every kind shares, drawn first so they don't depend on the kind's own draws."""
    now = BASE_T + r.randrange(365) * 86400 + r.randrange(86400)
    return {"now": now, "dials": _dials(r), "mood": _mood(r),
            "notes": (r.choice(NOTES),) if r.random() < 0.15 else ()}


def _prior(r: random.Random) -> tuple[dict, ...]:
    if r.random() >= 0.25:
        return ()
    u, a = r.choice(PRIOR)
    return ({"role": "user", "content": u}, {"role": "assistant", "content": a})


def _fill(r: random.Random, template: str) -> str:
    return template.format(name=r.choice(NAMES), place=r.choice(PLACES), food=r.choice(FOODS),
                           show=r.choice(SHOWS), colour=r.choice(COLOURS), instrument=r.choice(INSTRUMENTS))


def _fact(r: random.Random, a: Attr) -> str:
    return a.fact.format(v=r.choice(POOLS[a.pool]))


def _facts(r: random.Random, lo: int, hi: int, exclude: Sequence[str] = ()) -> tuple[str, ...]:
    """lo..hi facts about distinct attributes, none of them about an excluded one."""
    pool = [a for a in ATTRS if a.key not in exclude]
    return tuple(_fact(r, a) for a in r.sample(pool, r.randint(lo, hi)))


def _dress(r: random.Random, base: str, leads: Sequence[str] = LEADS, tails: Sequence[str] = TAILS) -> str:
    """A lead-in, a trailing filler and punctuation (statements only), sometimes capitalised."""
    text = r.choice(leads) + base
    if not text.endswith("?"):
        text += r.choice(tails) + r.choice(("", "", ".", "!"))
    return text[0].upper() + text[1:] if r.random() < 0.2 else text


def _small(r: random.Random) -> dict:
    return {"user_text": _dress(r, _fill(r, r.choice(SMALL))), "facts": _facts(r, 0, 3), "turns": _prior(r)}


def _about_me(r: random.Random) -> dict:
    text = r.choice(ABOUT_LEADS) + r.choice(ABOUT_ME) + r.choice(("", "?", "?", "?", "!"))
    return {"user_text": text, "facts": _facts(r, 2, 4), "turns": _prior(r)}


def _unknown(r: random.Random) -> dict:
    a = r.choice(ATTRS)
    text = r.choice(("", "", "hey riya, ", "quick question, ", "wait, ", "hmm, ")) + r.choice(a.unknown)
    return {"user_text": text, "facts": _facts(r, 1, 4, exclude=(a.key,)), "turns": _prior(r), "asks": a.key}


def _identity(r: random.Random) -> dict:
    core = r.choice(IDENTITY)
    lead = "" if core == "what are you" and r.random() < 0.5 else r.choice(IDENTITY_LEADS)
    return {"user_text": lead + core + r.choice(("?", "?", "?", "??", "")), "facts": _facts(r, 0, 2)}


def _remember(r: random.Random) -> dict:
    a = r.choice(ATTRS)
    lead = r.choice(REMEMBER_LEADS[:-1])
    text = lead + a.remember.format(v=r.choice(POOLS[a.pool])) + r.choice(("", "", ".", "!"))
    return {"user_text": text[0].upper() + text[1:] if r.random() < 0.2 else text,
            "facts": _facts(r, 0, 2, exclude=(a.key,))}


def _explain(r: random.Random) -> dict:
    form = r.random()
    if form < 0.6:
        text = r.choice(EXPLAIN).format(t=r.choice(TOPICS))
    elif form < 0.8:
        text = r.choice(WHYS)
    else:
        text = f"how {r.choice(('do i', 'can i', 'to'))} {r.choice(TASKS)} in {r.choice(LANGS)}?"
    return {"user_text": _dress(r, text, leads=("", "", "", "hey riya, ", "ok so ", "riya, ")),
            "facts": _facts(r, 0, 2), "turns": _prior(r)}


def _greeting(r: random.Random) -> dict:
    away = r.uniform(3600, 6 * 3600) if r.random() < 0.5 else r.uniform(86400, 5 * 86400)
    return {"user_text": None, "facts": _facts(r, 0, 3), "away_s": float(round(away)), "thread": _thread(r)}


def _thread(r: random.Random) -> str | None:
    return _fill(r, r.choice(THREADS)) if r.random() < 0.5 else None


def _bid(r: random.Random) -> dict:
    thread = _thread(r)
    reason = "due thread" if thread else f"worked {r.uniform(1.5, 6):.1f} h straight"
    return {"user_text": None, "facts": _facts(r, 0, 2), "thread": thread, "reason": reason}


def _last_letter(word: str) -> str:
    return next(c for c in reversed(word) if c.isalpha())


def _game_move(r: random.Random, game: str, seed: int) -> str:
    """A move that the engine judges (never "off") in the fresh game this seed deals."""
    g = games.new_game(game, random.Random(seed))  # type: ignore[arg-type]
    if game == "word_chain":
        need = _last_letter(g.word)
        if r.random() < 0.8:
            word = r.choice([w for w in games.WORDS[need] if w not in g.used])
        else:  # a word that starts with another letter: the engine says "wrong letter"
            word = r.choice([w for k, ws in games.WORDS.items() if k != need for w in ws])
        return word + r.choice(("", "", "", "!"))
    if game == "twenty_questions":
        if r.random() < 0.7:
            return r.choice(QUESTIONS_20)
        thing = r.choice(games.THINGS).name
        return r.choice((f"is it {games._a(thing)}?", f"it's {games._a(thing)}", f"is it {games._a(thing)}"))
    riddle = games.RIDDLES[g.riddle]
    if r.random() < 0.2:
        return r.choice(NO_IDEA)
    other = r.choice([x for x in games.RIDDLES if x is not riddle])
    answer = riddle if r.random() < 0.5 else other
    told = answer.told
    return r.choice((told, f"is it {told}?", f"maybe {told}", f"i think it's {told}", f"{answer.answers[0]}?"))


def _game(r: random.Random) -> dict:
    game = r.choice(games.GAMES)
    seed = r.randrange(1 << 31)
    start = r.random() < 0.4
    if start:
        text = r.choice(START_FORMS).format(g=r.choice(GAME_WORDS[game]))
        if r.random() < 0.1:
            text = {"word_chain": "word chain?", "twenty_questions": "20 questions!", "riddles": "riddles?"}[game]
    else:
        text = _game_move(r, game, seed)
    return {"user_text": text, "facts": _facts(r, 0, 2), "game": game, "game_seed": seed, "game_start": start}


def _tool(r: random.Random) -> dict:
    form = r.random()
    if form < 0.34:
        a, b = r.randint(2, 99), r.choice((r.randint(2, 99), r.randint(100, 999), 0 if r.random() < 0.2 else 7))
        text = r.choice(CALC_FORMS).format(e=f"{a} {r.choice(CALC_OPS)} {b}")
    elif form < 0.67:
        place = r.choice(PLACES)
        text = r.choice(WEATHER_FORMS).format(p=place)
    else:
        q, extract = r.choice(LOOKUPS)
        if r.random() < 0.25:  # a made-up person
            q, extract = f"{r.choice(NAMES)} {r.choice(SURNAMES)}", ""
            text = r.choice(("who is {q}", "who was {q}", "look up {q}", "tell me about {q}")).format(q=q)
        else:
            text = r.choice(LOOKUP_FORMS).format(q=q)
    call = match(text, BASE_T)
    if call is None:
        raise RuntimeError(f"tool text isn't recognised by match(): {text!r}")
    summary = REGISTRY[call.tool].summary(call.args)
    ok = True
    if call.tool == "calc":
        v = calc(call.args["expr"])
        ok, out = (False, "I couldn't work that out.") if v is None else (
            True, f"= {format_number(v)} ({call.args['expr']})")
    elif call.tool == "weather":
        place = call.args["place"]
        if r.random() < 0.85:
            out = (f"{place}: {r.randint(-5, 33)}°C and {r.choice(WEATHER_WORDS)} right now; today "
                   f"{r.randint(-8, 15)} to {r.randint(16, 36)}°C.")
        else:
            ok, out = False, r.choice(WEATHER_FAIL).format(p=place)
    else:
        query = call.args["query"]
        if r.random() < 0.85:
            out = extract or f"{query} is a character in a novel about a lighthouse keeper in {r.choice(PLACES)}."
        else:
            ok, out = False, r.choice(LOOKUP_FAIL).format(q=query)
    return {"user_text": text, "facts": _facts(r, 0, 2), "tool": (summary, ok, out)}


def _feedback(r: random.Random) -> dict:
    form = r.random()
    if form < 0.6:
        base = r.choice(DIAL_REQUESTS)
    elif form < 0.8:
        base = "i like it when you " + _fill(r, r.choice(NOTE_LIKES))
    elif form < 0.9:
        base = "i don't like it when you " + _fill(r, r.choice(NOTE_LIKES))
    else:
        base = r.choice(("please don't ", "please always ")) + _fill(r, r.choice(NOTE_RULES))
        if "always" in base:
            base = "please always " + r.choice(("greet me by name", "use short sentences", "say good morning first"))
    text = r.choice(FEEDBACK_LEADS) + base + r.choice(FEEDBACK_TAILS) + r.choice(("", "", ".", "!"))
    return {"user_text": text, "facts": _facts(r, 0, 2)}


def _report(r: random.Random) -> dict:
    n = r.randint(1, 3)
    bases = r.sample(AGENT_BASES, n)
    events = []
    for base in bases:
        kind = r.choice(REPORT_KINDS)
        text = r.choice(EVENT_TEXTS[kind]).format(n=r.randint(2, 40), x=round(r.uniform(0.1, 0.9), 2))
        events.append((kind, f"{base}-{r.randint(1, 9)}", text))
    return {"user_text": None, "facts": _facts(r, 0, 2), "events": tuple(events)}


_DRAWS: dict[str, Callable[[random.Random], dict]] = {
    "small": _small, "about_me": _about_me, "unknown": _unknown, "identity": _identity, "remember": _remember,
    "explain": _explain, "greeting": _greeting, "bid": _bid, "game": _game, "tool": _tool, "feedback": _feedback,
    "report": _report}


def _clean(s: Scenario, texts: list[str], facts: set[str]) -> bool:
    """No user text overlaps a probe, no fact (or follow-up thread) is a probe fact."""
    user = ([s.user_text] if s.user_text is not None else []) + [t["content"] for t in s.turns if t["role"] == "user"]
    if any(overlaps(t, texts) for t in user):
        return False
    return not any(p in normalize(x) for x in (*s.facts, *([s.thread] if s.thread else [])) for p in facts)


def scenario(kind: str, index: int, seed: int = DEFAULT_SEED) -> Scenario:
    """One row. Its only randomness is random.Random(f"{seed}:{kind}:{index}"), so it doesn't depend on n or kinds."""
    if kind not in _DRAWS:
        raise ValueError(f"unknown kind: {kind!r}")
    r = random.Random(f"{seed}:{kind}:{index}")
    texts, facts = probe_texts(), {normalize(f) for f in probe_facts()}
    for _ in range(MAX_DRAWS + 1):
        s = Scenario(id=f"{kind}-{index:05d}", kind=kind, **{**_common(r), **_DRAWS[kind](r)})
        if _clean(s, texts, facts):
            return s
    raise RuntimeError(f"{kind}-{index}: no probe-free draw")


def counts(n: int = DEFAULT_N, kinds: Sequence[str] = KINDS) -> dict[str, int]:
    """Rows per kind: n * weight // total weight, the remainder one each to the first kinds (in KINDS order)."""
    unknown = [k for k in kinds if k not in MIX]
    if unknown or not kinds:
        raise ValueError(f"unknown kinds: {unknown or list(kinds)}")
    ordered = [k for k in KINDS if k in kinds]
    total = sum(MIX[k] for k in ordered)
    out = {k: n * MIX[k] // total for k in ordered}
    for k in ordered[:n - sum(out.values())]:
        out[k] += 1
    return out


def scenarios(seed: int = DEFAULT_SEED, n: int = DEFAULT_N, kinds: Sequence[str] = KINDS) -> list[Scenario]:
    """Every kind's rows, in KINDS order and then by index."""
    return [scenario(k, i, seed) for k, c in counts(n, kinds).items() for i in range(c)]
