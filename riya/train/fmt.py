"""The row formats of Riya v1's fine-tuning data: the requests the dataset builder writes, the teacher's answers, the
chat rows the trainer reads, the LFM2 chat template and the assistant-only label mask. Pure and CPU-only: stdlib only
(a tokenizer is passed in, so torch and transformers are never imported here).
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1, §2.3 and §4."""
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TextIO

# fp: build_sft.prompt_fingerprint() at build time; meta: the structured facts and the asked attribute; h: the
# request's own content hash (request_hash), which the teacher echoes so an answer is only ever paired with the request
# it answered
REQUEST_KEYS = ("id", "kind", "messages", "facts", "expect", "meta", "fp", "h")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def request_hash(row: dict) -> str:
    """h: the sha256 (hex) of the request's messages and meta as canonical JSON (sorted keys, no spaces, non-ASCII
    kept). The same id rebuilt from another seed, pool or prompt has another h."""
    blob = json.dumps({"messages": row["messages"], "meta": row["meta"]}, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
EXPECT_KEYS = ("must_mention", "must_not_invent", "must_admit_unknown", "must_say_ai", "max_sentences", "max_emoji")


def read_jsonl_checked(path: Path) -> tuple[list[dict], int]:
    """(rows, dropped): one JSON value per line (utf-8), blank lines skipped. A last line that isn't valid JSON, the
    mark of a run killed mid-write, is dropped and counted (1) and reported loudly on stderr; a broken line anywhere
    else still raises ValueError (json.JSONDecodeError)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, dropped = [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            dropped = 1
            print(f"WARNING: {path}: dropped {dropped} broken last line (an interrupted write or copy; {len(rows)} "
                  f"rows read): {line[:60]!r}", file=sys.stderr, flush=True)
    return rows, dropped


def read_jsonl(path: Path) -> list[dict]:
    """One JSON value per line (utf-8); blank lines are skipped, and so is a broken last line (see
    read_jsonl_checked)."""
    return read_jsonl_checked(path)[0]


def _write_atomic(path: Path, write: Callable[[TextIO], None]) -> None:
    """Writes through a temp file next to path, then os.replace: a reader sees the old file or the whole new one,
    never a part. A failure (or Ctrl-C) leaves the old file and no temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    """One JSON object per line (utf-8, non-ASCII kept, "\\n" endings), making the parent folder first; atomic."""
    def write(f: TextIO) -> None:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_atomic(path, write)


def write_json(obj, path: Path) -> None:
    """One JSON document (utf-8, indented, a final newline), making the parent folder first; atomic."""
    text = json.dumps(obj, indent=2) + "\n"  # encoded first: a value json can't take raises before anything is written
    _write_atomic(path, lambda f: f.write(text))


def _int(v, lo: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def validate_request(row: dict) -> None:
    """ValueError unless row is a request: exactly REQUEST_KEYS; messages a system message, then user/assistant pairs,
    ending with a user message, all with str content; facts a list of str; expect exactly EXPECT_KEYS (list of str,
    bool, bool, bool, int >= 1, int >= 0); meta a record per fact plus the asked attribute; fp a sha256 hex digest (64
    lowercase hex characters); h equal to request_hash(row)."""
    if not isinstance(row, dict) or set(row) != set(REQUEST_KEYS) or len(row) != len(REQUEST_KEYS):
        raise ValueError(f"a request has exactly the keys {REQUEST_KEYS}")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"{row['id']}: messages must be a system message, user/assistant pairs and a user message")
    for i, m in enumerate(messages):
        want = "system" if i == 0 else "user" if i % 2 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want or not isinstance(m.get("content"), str):
            raise ValueError(f"{row['id']}: message {i} must be a {want} message with str content")
    if not isinstance(row["facts"], list) or not all(isinstance(f, str) for f in row["facts"]):
        raise ValueError(f"{row['id']}: facts must be a list of str")
    e = row["expect"]
    if not isinstance(e, dict) or set(e) != set(EXPECT_KEYS) or len(e) != len(EXPECT_KEYS):
        raise ValueError(f"{row['id']}: expect has exactly the keys {EXPECT_KEYS}")
    if not isinstance(e["must_mention"], list) or not all(isinstance(f, str) for f in e["must_mention"]):
        raise ValueError(f"{row['id']}: must_mention must be a list of str")
    if not all(isinstance(e[k], bool) for k in ("must_not_invent", "must_admit_unknown", "must_say_ai")):
        raise ValueError(f"{row['id']}: must_not_invent, must_admit_unknown and must_say_ai must be bool")
    if not _int(e["max_sentences"], 1) or not _int(e["max_emoji"], 0):
        raise ValueError(f"{row['id']}: max_sentences must be an int >= 1 and max_emoji an int >= 0")
    meta = row["meta"]
    if not isinstance(meta, dict) or set(meta) != {"facts", "asks"} or not isinstance(meta["facts"], list):
        raise ValueError(f"{row['id']}: meta has exactly the keys facts (a list) and asks")
    if len(meta["facts"]) != len(row["facts"]) or not all(
            isinstance(r, dict) and set(r) == {"attr", "value_words", "species", "words"}
            and isinstance(r["value_words"], list) and all(isinstance(w, str) for w in r["value_words"])
            for r in meta["facts"]):
        raise ValueError(f"{row['id']}: meta.facts holds one {{attr, value_words, species, words}} record per fact")
    asks = meta["asks"]
    if asks is not None and (not isinstance(asks, dict) or set(asks) != {"attr", "words", "values"}
                             or not all(isinstance(v, str) for v in asks["values"] + asks["words"])):
        raise ValueError(f"{row['id']}: meta.asks is None or {{attr, words, values}}")
    if not isinstance(row["fp"], str) or not _HEX64.fullmatch(row["fp"]):
        raise ValueError(f"{row['id']}: fp must be a sha256 hex digest")
    if not isinstance(row["h"], str) or row["h"] != request_hash(row):
        raise ValueError(f"{row['id']}: h must be request_hash(row) (the messages or meta changed after it was built)")


TEACHER_KEYS = ("id", "h", "response")  # h: the answered request's h, echoed
SFT_KEYS = ("id", "kind", "messages")
BOS = "<|startoftext|>"
IGNORE_INDEX = -100  # the label the loss skips
MAX_LEN = 2048  # a longer row is skipped, never truncated


def validate_teacher(row: dict) -> None:
    """ValueError unless row is a teacher answer: exactly the keys id, h and response, all str, h a sha256 hex
    digest (the h of the request it answered)."""
    if not isinstance(row, dict) or set(row) != set(TEACHER_KEYS):
        raise ValueError(f"a teacher row has exactly the keys {TEACHER_KEYS}")
    if not all(isinstance(row[k], str) for k in TEACHER_KEYS):
        raise ValueError(f"{row.get('id')}: id, h and response must be str")
    if not _HEX64.fullmatch(row["h"]):
        raise ValueError(f"{row['id']}: h must be a sha256 hex digest")


# A remember-that turn: the engine says a templated ack first (engine.REMEMBER_ACKS, "Got it~ I'll remember that."),
# before the model is even asked, then the model's one sentence. Her prompt doesn't show the ack, so the trained
# answer is only what follows it; an ack of the teacher's own would be said twice at run time.
LEAD_KINDS = frozenset({"remember"})
_SENTENCE_END = re.compile(r"(?<=[.!?…~])\s+")
_ACK = re.compile(r"\b(?:i'?ll|i\s+will|i\s+won'?t|i\s+will\s+not|won'?t|gonna)\s+"
                  r"(?:remember|forget|keep|note|save|store)"
                  r"|\b(?:noted|saved|stored|filed|memori[sz]ed|remembered|got\s+it|gotcha|got\s+that|mental\s+note"
                  r"|written\s+down|jotted|locked\s+in|keep\s+that\s+in\s+mind)\b"
                  r"|\bremember(?:ing)?\s+(?:that|this|it)\b", re.I)
_BARE_ACK = re.compile(r"^\W*(?:ok(?:ay)?|alright|sure(?:\s+thing)?|done|mm+(?:-?hm+)?|mhm|yes|yep|yup|kay|roger"
                       r"|understood|will\s+do|of\s+course)\W*$", re.I)


def trained_reply(kind: str, response: str) -> str:
    """The part of an answer the model is trained to say: on a remember row, what follows the leading
    acknowledgement sentences ("Got it~", "Okay! Noted~", "I'll remember that."), since the engine already said its
    own; on any other row, the answer as it is. "" when the answer is nothing but an acknowledgement."""
    if kind not in LEAD_KINDS:
        return response
    sentences = _SENTENCE_END.split(response.strip())
    while sentences and (_ACK.search(sentences[0]) or _BARE_ACK.match(sentences[0])):
        sentences.pop(0)
    return " ".join(sentences).strip()


def sft_row(request: dict, response: str) -> dict:
    """The chat row the trainer reads: the request's messages, then the answer as the assistant's turn."""
    return {"id": request["id"], "kind": request["kind"],
            "messages": request["messages"] + [{"role": "assistant", "content": response}]}


def validate_sft_row(row: dict) -> None:
    """ValueError unless row is a chat row: exactly id, kind and messages; messages a system message, then user and
    assistant turns in turn, ending with an assistant message, all with str content."""
    if not isinstance(row, dict) or set(row) != set(SFT_KEYS):
        raise ValueError(f"an SFT row has exactly the keys {SFT_KEYS}")
    if not isinstance(row["id"], str) or not isinstance(row["kind"], str):
        raise ValueError("id and kind must be str")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
        raise ValueError(f"{row['id']}: messages must be a system message, user/assistant pairs and end with an "
                         "assistant message")
    for i, m in enumerate(messages):
        want = "system" if i == 0 else "user" if i % 2 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want or not isinstance(m.get("content"), str):
            raise ValueError(f"{row['id']}: message {i} must be a {want} message with str content")


def render_lfm2(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """LFM2's chat template: BOS, then each message as <|im_start|>role, a newline, the content and <|im_end|> with a
    newline; with add_generation_prompt, an open assistant turn at the end."""
    text = BOS + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + "<|im_start|>assistant\n" if add_generation_prompt else text


def template_matches(server_prompt: str, hf_prompt: str) -> bool:
    """llama-server's rendered prompt equals the HF tokenizer's, once one leading BOS is removed from each (one side
    adds it as a token, the other in the text)."""
    def bare(t: str) -> str:
        return t[len(BOS):] if t.startswith(BOS) else t
    return bare(server_prompt) == bare(hf_prompt)


def encode_row(tok, messages: list[dict], max_len: int = MAX_LEN) -> tuple[list[int], list[int]] | None:
    """(input ids, labels) for one chat row, training on the final assistant turn only: every prompt token's label is
    IGNORE_INDEX, and the answer tokens (with the end-of-turn marker) keep their ids. None when the row is longer
    than max_len tokens (it is skipped, never truncated). ValueError when the template isn't prefix-stable, i.e. the
    prompt's text or token ids aren't the start of the full conversation's, because the mask would then be off."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the last message must be the assistant's answer")
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if not full.startswith(prompt) or ids[:len(p_ids)] != p_ids:
        raise ValueError("the chat template isn't prefix-stable")
    if len(ids) > max_len:
        return None
    return ids, [IGNORE_INDEX] * len(p_ids) + ids[len(p_ids):]


# ---- the template check against the user's own llama-server ----------------------------------------------------
# The trainer renders her prompts with the HF chat template, and the Colab smoke step proves that equals
# render_lfm2 and Colab's fresh llama.cpp build. What she runs is the user's own llama-server on Windows: this check
# compares render_lfm2 with that server's /apply-template, and the champion gate runs it before it evaluates.

DEFAULT_SERVER = "http://127.0.0.1:8090"
CANARY = [{"role": "system", "content": "You are Riya. Keep it short~\nTwo lines, a tab\tand “quotes”."},
          {"role": "user", "content": "hey riya, what's up? 🙂"},
          {"role": "assistant", "content": "Hmph, nothing much~ Café later?"},
          {"role": "user", "content": "sure! remember that I like oat milk"}]


def _post_json(url: str, body: dict, timeout: float) -> dict:
    import urllib.request
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check_template(server: str, *, messages: list[dict] = CANARY,
                   post: Callable[[str, dict, float], dict] | None = None, timeout: float = 10.0) -> list[str]:
    """[] when the server's /apply-template renders messages (with the generation prompt) exactly as render_lfm2 does,
    one leading BOS aside; else what differs, with the first differing character. A server that can't be reached
    raises (OSError)."""
    got = (post or _post_json)(server.rstrip("/") + "/apply-template",
                               {"messages": messages, "add_generation_prompt": True}, timeout)
    prompt = got.get("prompt") if isinstance(got, dict) else None
    if not isinstance(prompt, str):
        return ["llama-server's /apply-template returned no prompt"]
    want = render_lfm2(messages, add_generation_prompt=True)
    if template_matches(prompt, want):
        return []
    a, b = (t[len(BOS):] if t.startswith(BOS) else t for t in (prompt, want))
    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return [f"llama-server's chat template differs from the trainer's at character {at}: "
            f"server {a[at:at + 24]!r}, trainer {b[at:at + 24]!r}"]


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m riya.train.fmt",
                                description="Check that your llama-server renders Riya's prompts as the trainer does.")
    p.add_argument("--check-template", action="store_true", required=True,
                   help="compare the trainer's LFM2 template with the server's /apply-template")
    p.add_argument("--server", default=DEFAULT_SERVER, help="your running llama-server (default: Riya's own)")
    args = p.parse_args(argv)
    try:
        problems = check_template(args.server, post=_post_json)
    except (OSError, ValueError) as e:
        print(f"TEMPLATE_UNKNOWN: couldn't reach llama-server at {args.server} ({e}). Start Riya and run it again.")
        return 2
    if problems:
        print("TEMPLATE_MISMATCH: " + "; ".join(problems) + ". Don't adopt an adapter trained on this template.")
        return 1
    print(f"TEMPLATE_OK: {args.server} renders her prompts exactly as the trainer does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
