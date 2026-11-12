"""On Colab: a larger uncensored teacher writes the replies Riya v1 is fine-tuned on.

Reads requests.jsonl (riya.train.build_sft: her exact local-model prompts, synthetic only), serves the teacher with
llama-server (8 parallel slots, the colab/gen_riya_data.py pattern) and appends {"id", "h", "response"} rows to
teacher.jsonl, h being the answered request's content hash. Resumable: a request is skipped only when --out already
holds an answer with its id and its h, so requests rebuilt under another seed, pool or prompt are answered again. The
filter (riya.train.filter_sft) runs afterwards, locally, and pairs answers on id and h.

    python -m colab.gen_riya_v1                       # starts the teacher itself
    python -m colab.gen_riya_v1 --url http://127.0.0.1:8090 --limit 24   # an already-running server, a few rows

The online model is a fallback for a request the teacher failed on. It is off unless TEACHER_ONLINE=1 and an online
model is available (OpenCode's free tier needs no key; RIYA_ONLINE_API_KEY lifts its limits), and then it is limited
to ONLINE_RPM requests a minute.
See docs/superpowers/specs/2026-10-19-riya-v0.8-riya-v1-design.md §2.2."""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from colab.gen_riya_data import find_teacher
from riya.train.fmt import read_jsonl, read_jsonl_checked, validate_request, write_jsonl

RIYA_V1_DIR = os.environ.get("RIYA_V1_DIR", "/content/drive/MyDrive/riya_v1")
PORT = 8095
SLOTS = 8
TEACHER_TOKENS = 400  # her replies are short, but a clipped answer would be filtered out as "truncated"
TEMPERATURE = 0.8
ONLINE_RPM = 10  # the free tier
SERVER = os.environ.get("LLAMA_SERVER", "/content/bin/llama-server")
SLOT_CTX = 4096  # per slot: her system prompt, examples and memory block, plus the answer
ONLINE_BASE_URL = os.environ.get("RIYA_ONLINE_BASE_URL", "https://opencode.ai/inference/openai/v1")
ONLINE_MODEL = os.environ.get("RIYA_ONLINE_MODEL", "glm-5.3-flash-free")


def teacher_body(messages: list[dict]) -> dict:
    """The chat request for one row: the teacher sees her exact prompt."""
    return {"messages": messages, "max_tokens": TEACHER_TOKENS, "temperature": TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False}}


def clean_response(text: str) -> str:
    """What follows the last </think>, whitespace-stripped, minus one pair of surrounding double quotes. Labels
    ("Riya: ...") stay: the filter rejects them."""
    text = text.rsplit("</think>", 1)[-1].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def todo(requests: list[dict], done: set[tuple[str, str]]) -> list[dict]:
    """The requests with no answer for their (id, h) yet, in order: an answer written for another request under the
    same id doesn't count."""
    return [r for r in requests if (r["id"], r["h"]) not in done]


def online_gap_s(rpm: int = ONLINE_RPM) -> float:
    """Seconds between online requests at rpm requests a minute."""
    return 60 / rpm


def online_enabled(environ: Mapping[str, str]) -> bool:
    """The online fallback needs TEACHER_ONLINE=1 and an online model: OpenCode's free tier needs no key."""
    from riya.core.models import online_available
    return environ.get("TEACHER_ONLINE") == "1" and online_available(environ)


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 300) -> str:
    from riya.core.models import USER_AGENT  # opencode.ai rejects urllib's default User-Agent with 403
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def _post_chat(url: str, body: dict, timeout: float = 300) -> str:
    """The raw reply text from a llama-server /v1/chat/completions endpoint."""
    return _post(url, body, timeout=timeout)


def _online_chat(environ: Mapping[str, str]) -> Callable[[dict], str]:
    key = (environ.get("RIYA_ONLINE_API_KEY") or "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    def chat(body: dict) -> str:
        # a reasoning model's thinking tokens count against max_tokens, so leave room or the reply comes back empty
        sent = {"model": ONLINE_MODEL, "messages": body["messages"], "max_tokens": body["max_tokens"] * 4,
                "temperature": body["temperature"]}  # no chat_template_kwargs: that is llama-server's
        return _post(f"{ONLINE_BASE_URL.rstrip('/')}/chat/completions", sent, headers, 120)
    return chat


class _Throttle:
    """At most one call every gap_s seconds across all threads (the clock and the sleep are injected)."""

    def __init__(self, gap_s: float, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.gap_s, self.clock, self.sleep = gap_s, clock, sleep
        self._lock = threading.Lock()
        self._last: float | None = None

    def wait(self) -> None:
        with self._lock:
            if self._last is not None:
                left = self.gap_s - (self.clock() - self._last)
                if left > 0:
                    self.sleep(left)
            self._last = self.clock()


def generate(requests: list[dict], out: Path, teacher_chat: Callable[[dict], str],
             online_chat: Callable[[dict], str] | None = None, *, gap_s: float | None = None,
             clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
             workers: int = SLOTS) -> tuple[int, int]:
    """Appends {"id", "h", "response"} to out for each request, answered by teacher_chat(body) and, when it fails or
    answers blank, by online_chat (throttled to one call per gap_s). Returns (answered, failed); a failed row is
    written nowhere, so the next run retries it."""
    throttle = _Throttle(online_gap_s() if gap_s is None else gap_s, clock, sleep)
    stopping = threading.Event()  # set on an interrupt: queued work must not go online or write

    def work(r: dict) -> dict | None:
        body = teacher_body(r["messages"])
        text = ""
        try:
            text = clean_response(teacher_chat(body))
        except Exception as e:
            print(f"{r['id']}: teacher failed ({e})", flush=True)
        if not text and online_chat is not None and not stopping.is_set():
            throttle.wait()
            try:
                text = clean_response(online_chat(body))
            except Exception as e:
                print(f"{r['id']}: online fallback failed ({e})", flush=True)
        return {"id": r["id"], "h": r["h"], "response": text} if text else None

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    t0 = time.time()
    ex = ThreadPoolExecutor(workers)
    try:
        with out.open("a", encoding="utf-8", newline="\n") as f:
            for row in ex.map(work, requests):
                if row is None:
                    failed += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()  # every row: a killed run loses at most the line being written
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{len(requests)} ({done / (time.time() - t0):.1f}/s)", flush=True)
    finally:
        stopping.set()
        ex.shutdown(wait=False, cancel_futures=True)  # on an error or Ctrl-C, drop the queue instead of draining it
    return done, failed


def resume_keys(out: Path) -> set[tuple[str, str]]:
    """The (id, h) pairs already answered in out (empty when it doesn't exist); a row without an h (written before h
    existed) answers nothing. A last line cut off by an interrupted write is reported and removed (the file is rewritten
    without it), so the rows appended next start on a clean line."""
    out = Path(out)
    if not out.exists():
        return set()
    rows, dropped = read_jsonl_checked(out)
    text = out.read_text(encoding="utf-8")
    if dropped or (text and not text.endswith("\n")):
        write_jsonl(rows, out)
    return {(r["id"], r["h"]) for r in rows if isinstance(r.get("h"), str)}


def _wait_healthy(base: str, proc: subprocess.Popen | None) -> None:
    for _ in range(600):
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"the teacher server exited early (code {proc.returncode}); see its log")
        try:
            urllib.request.urlopen(f"{base}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("the teacher server never became healthy")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Riya v1 teacher: requests.jsonl -> teacher.jsonl")
    ap.add_argument("--requests", default=f"{RIYA_V1_DIR}/requests.jsonl")
    ap.add_argument("--out", default=f"{RIYA_V1_DIR}/teacher.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N requests")
    ap.add_argument("--url", default=None, help="use this llama-server (http://host:port) and start none")
    args = ap.parse_args(argv)

    requests = read_jsonl(Path(args.requests))
    if args.limit is not None:
        requests = requests[:args.limit]
    for r in requests:
        validate_request(r)
    out = Path(args.out)
    done = resume_keys(out)
    pending = todo(requests, done)
    print(f"{len(requests) - len(pending)} done, {len(pending)} to generate", flush=True)

    online = _online_chat(os.environ) if online_enabled(os.environ) else None
    print("online fallback:", f"on, {ONLINE_RPM} requests/min" if online else "off", flush=True)
    proc = None
    base = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{PORT}"
    try:
        if pending and args.url is None:
            teacher = find_teacher()
            log_path = os.environ.get("TEACHER_LOG", "/content/runs/teacher.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            proc = subprocess.Popen([SERVER, "-m", teacher, "--port", str(PORT), "-ngl", "99",
                                     "-c", str(SLOTS * SLOT_CTX), "-np", str(SLOTS), "-fa", "on", "--no-webui",
                                     "--cache-ram", "0"],  # the default 8 GiB host prompt cache OOM-kills it on Colab
                                    stdout=subprocess.DEVNULL, stderr=open(log_path, "w"))
        if proc is not None:  # --url names a server that is already up
            _wait_healthy(base, proc)
        endpoint = f"{base}/v1/chat/completions"
        answered, failed = generate(pending, out, lambda body: _post_chat(endpoint, body), online)
    finally:
        if proc is not None:
            proc.terminate()
    if failed:
        print(f"GEN_PARTIAL: {failed} request(s) got no answer; run this cell again to retry them", flush=True)
        return 1
    print("GEN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
