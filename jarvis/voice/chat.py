"""Chat template for SFT: plain-text role markers on top of the GPT-2 tokenizer.

  <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n...<|end|>\n

The markers are ordinary strings for now (GPT-2 splits them into a few tokens); they can become real special
tokens later without changing the data. `format_row` also returns the prompt prefix so the trainer can put the
loss on the assistant response only.
"""
SYSTEM, USER, ASSISTANT, END = "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
ROLE_MARKERS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def render_turn(role, text):
    return f"{ROLE_MARKERS[role]}\n{text.strip()}{END}\n"


def render_chat(turns, system=None):
    """turns: [{"role": "user"|"assistant", "text": ...}] -> one string."""
    head = render_turn("system", system) if system else ""
    return head + "".join(render_turn(t["role"], t["text"]) for t in turns)


def format_row(row, system=None):
    """Dataset row -> {"prompt", "completion", "text"}; text = prompt + completion, loss goes on completion."""
    prompt = render_chat(row["context"], system) + ASSISTANT + "\n"
    completion = row["response"].strip() + END + "\n"
    return {"prompt": prompt, "completion": completion, "text": prompt + completion}
