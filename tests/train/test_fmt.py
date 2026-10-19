"""v0.8 Task 3: the chat rows the trainer reads, the LFM2 template and the assistant-only label mask, tested on a toy
character-level tokenizer (no torch, no transformers)."""
import pytest

from riya.train.fmt import (BOS, IGNORE_INDEX, MAX_LEN, TEACHER_KEYS, encode_row, render_lfm2, sft_row,
                            template_matches, validate_request, validate_sft_row, validate_teacher)

SYSTEM = {"role": "system", "content": "S"}
USER = {"role": "user", "content": "Yo"}
ANSWER = {"role": "assistant", "content": "Hi~"}


class ToyTok:
    """Character-level: the template is render_lfm2 and each character is its own id."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return render_lfm2(messages, add_generation_prompt)

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False  # the template already carries BOS
        return {"input_ids": [ord(c) for c in text]}

    @staticmethod
    def decode(ids):
        return "".join(chr(i) for i in ids)


def good_request():
    return {"id": "r1", "kind": "small", "messages": [SYSTEM, USER], "facts": [],
            "expect": {"must_mention": [], "must_not_invent": False, "must_admit_unknown": False,
                       "must_say_ai": False, "max_sentences": 3, "max_emoji": 1}, "fp": "0" * 64}


def test_constants():
    assert (BOS, IGNORE_INDEX, MAX_LEN, TEACHER_KEYS) == ("<|startoftext|>", -100, 2048, ("id", "response"))


def test_render_lfm2_golden():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert render_lfm2(msgs, add_generation_prompt=True) == (
        "<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")
    assert render_lfm2(msgs) == "<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
    assert render_lfm2([]) == BOS


def test_template_matches():
    msgs = [SYSTEM, USER]
    with_bos = render_lfm2(msgs, add_generation_prompt=True)
    assert template_matches(with_bos, with_bos[len(BOS):])
    assert template_matches(with_bos[len(BOS):], with_bos)
    assert template_matches(with_bos, with_bos)
    assert not template_matches(with_bos, with_bos[len(BOS):].replace("user\n", "user"))
    assert not template_matches(with_bos, with_bos[len(BOS):] + "\n")
    assert not template_matches(BOS + BOS + "x", "x")  # only ONE leading BOS is removed from each


def test_encode_row_masks_all_but_the_answer():
    tok = ToyTok()
    messages = [SYSTEM, USER, ANSWER]
    ids, labels = encode_row(tok, messages)
    assert len(ids) == len(labels)
    n_prompt = len(tok(render_lfm2(messages[:-1], add_generation_prompt=True), add_special_tokens=False)["input_ids"])
    assert labels[:n_prompt] == [IGNORE_INDEX] * n_prompt
    assert all(label == i for label, i in zip(labels[n_prompt:], ids[n_prompt:]))
    assert tok.decode([label for label in labels if label != IGNORE_INDEX]) == "Hi~<|im_end|>\n"
    assert encode_row(tok, messages, max_len=10) is None
    assert encode_row(tok, messages, max_len=len(ids)) is not None  # the limit itself fits


def test_only_the_final_turn_is_trained():
    """An earlier assistant turn in the history is context, not a target."""
    tok = ToyTok()
    messages = [SYSTEM, USER, {"role": "assistant", "content": "Earlier~"}, {"role": "user", "content": "And?"}, ANSWER]
    ids, labels = encode_row(tok, messages)
    assert tok.decode([label for label in labels if label != IGNORE_INDEX]) == "Hi~<|im_end|>\n"
    assert "Earlier" in tok.decode(ids) and "Earlier" not in tok.decode([x for x in labels if x != IGNORE_INDEX])


def test_encode_row_refuses_an_unstable_template():
    class Rewriting(ToyTok):  # the full render rewrites the system message, so the prompt isn't its prefix
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return text if add_generation_prompt else text.replace("system\nS", "system\nS!")

    class Merging(ToyTok):  # same text, but a token merges across the prompt/answer boundary
        def __call__(self, text, add_special_tokens=True):
            ids = super().__call__(text, add_special_tokens)["input_ids"]
            if not text.endswith("assistant\n"):
                i = text.index("assistant\n") + len("assistant\n")
                ids = ids[:i - 1] + [-1] + ids[i + 1:]  # "\n" and the first answer character become one token
            return {"input_ids": ids}

    for tok in (Rewriting(), Merging()):
        with pytest.raises(ValueError, match="prefix-stable"):
            encode_row(tok, [SYSTEM, USER, ANSWER])


def test_encode_row_needs_an_assistant_answer():
    with pytest.raises(ValueError):
        encode_row(ToyTok(), [SYSTEM, USER])


def test_fmt_imports_only_the_stdlib():
    import ast
    import sys

    import riya.train.fmt as fmt
    tree = ast.parse(open(fmt.__file__, encoding="utf-8").read())
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert roots <= set(sys.stdlib_module_names)  # in particular no torch, transformers or peft, even in a function


def test_row_validation():
    request = good_request()
    row = sft_row(request, "Hi~")
    assert row == {"id": "r1", "kind": "small", "messages": [SYSTEM, USER, ANSWER]}
    assert request["messages"] == [SYSTEM, USER]  # the request is left alone
    validate_sft_row(row)
    validate_teacher({"id": "r1", "response": "Hi~"})

    for bad in ({"id": "r1"}, {"id": "r1", "response": 3}, {"id": "r1", "response": "x", "extra": 1}, "r1"):
        with pytest.raises(ValueError):
            validate_teacher(bad)
    with pytest.raises(ValueError):
        validate_sft_row({**row, "messages": [SYSTEM, USER]})  # must end with the assistant
    with pytest.raises(ValueError):
        validate_sft_row({**row, "messages": [SYSTEM, USER, USER]})
    with pytest.raises(ValueError):
        validate_sft_row({**row, "messages": [SYSTEM, USER, {"role": "assistant", "content": 3}]})
    with pytest.raises(ValueError):
        validate_sft_row({**row, "extra": 1})
    with pytest.raises(ValueError):
        validate_sft_row({"id": "r1", "kind": "small"})

    validate_request(request)
    with pytest.raises(ValueError):  # a request ends with a user message
        validate_request({**request, "messages": [SYSTEM, USER, ANSWER]})
