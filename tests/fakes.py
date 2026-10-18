import re

from riya.core.models import ModelUnavailable


class FakeModel:
    def __init__(self, name="fake", replies=("Okay.",), fail=None, fail_after=None):
        self.name, self.replies, self.fail, self.fail_after = name, list(replies), fail, fail_after
        self.calls = []

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.calls.append(messages)
        if self.fail:
            raise self.fail
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        for i, tok in enumerate(re.findall(r"\S+\s*", text)):
            if self.fail_after is not None and i == self.fail_after:
                raise ModelUnavailable("dropped mid-stream")
            yield tok
