"""The row formats of Riya v1's fine-tuning data: the requests the dataset builder writes, the teacher's answers, the
chat rows the trainer reads, the LFM2 chat template and the assistant-only label mask. Pure and CPU-only: stdlib only
(a tokenizer is passed in, so torch and transformers are never imported here).
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.1, §2.3 and §4."""
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# fp: build_sft.prompt_fingerprint() at build time; meta: the structured facts and the asked attribute
REQUEST_KEYS = ("id", "kind", "messages", "facts", "expect", "meta", "fp")
_HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECT_KEYS = ("must_mention", "must_not_invent", "must_admit_unknown", "must_say_ai", "max_sentences", "max_emoji")


def read_jsonl_checked(path: Path) -> tuple[list[dict], int]:
    """(rows, dropped): one JSON value per line (utf-8), blank lines skipped. A last line that isn't valid JSON, the
    mark of a run killed mid-write, is dropped and counted (1) and reported on stderr; a broken line anywhere else
    still raises ValueError (json.JSONDecodeError)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, dropped = [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            dropped = 1
            print(f"{path}: ignoring a broken last line (an interrupted write): {line[:60]!r}", file=sys.stderr)
    return rows, dropped


def read_jsonl(path: Path) -> list[dict]:
    """One JSON value per line (utf-8); blank lines are skipped, and so is a broken last line (see
    read_jsonl_checked)."""
    return read_jsonl_checked(path)[0]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    """One JSON object per line (utf-8, non-ASCII kept, "\\n" endings), making the parent folder first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _int(v, lo: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def validate_request(row: dict) -> None:
    """ValueError unless row is a request: exactly REQUEST_KEYS; messages a system message, then user/assistant pairs,
    ending with a user message, all with str content; facts a list of str; expect exactly EXPECT_KEYS (list of str,
    bool, bool, bool, int >= 1, int >= 0); meta a record per fact plus the asked attribute; fp a sha256 hex digest (64
    lowercase hex characters)."""
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


TEACHER_KEYS = ("id", "response")
SFT_KEYS = ("id", "kind", "messages")
BOS = "<|startoftext|>"
IGNORE_INDEX = -100  # the label the loss skips
MAX_LEN = 2048  # a longer row is skipped, never truncated


def validate_teacher(row: dict) -> None:
    """ValueError unless row is a teacher answer: exactly the keys id and response, both str."""
    if not isinstance(row, dict) or set(row) != set(TEACHER_KEYS):
        raise ValueError(f"a teacher row has exactly the keys {TEACHER_KEYS}")
    if not all(isinstance(row[k], str) for k in TEACHER_KEYS):
        raise ValueError(f"{row.get('id')}: id and response must be str")


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
