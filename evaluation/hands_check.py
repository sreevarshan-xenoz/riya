"""Live check of her hands: the T0 tools run for real, the T1 and T2 ones only as a dry run.

  python -m evaluation.hands_check [--city London]

T0 (live): time, a 1 s timer on a throwaway HandsDB, calc, the weather for the city (Open-Meteo) and a lookup of
"Alan Turing" (Wikipedia). Only those two touch the network, with the Riya User-Agent; weather sends only the place and
lookup only the query. T1 and T2 are never run: for "open youtube", "volume up", "run backup" and "write a file called
x saying y" it shows match, validate_call, the summary and the confirmation question, against a throwaway Settings with a
sample `backup` command that is never run. Nothing calls `effects` (DryRunEffects only prints, and only if called).
The db is a temp file, never the one under %LOCALAPPDATA%\\Riya. The last line is the VERDICT; always exits 0; not part
of pytest.
"""
import argparse
import tempfile
from pathlib import Path

from riya.core.clock import SystemClock
from riya.core.hands.intent import match
from riya.core.hands.net import UrlHttp
from riya.core.hands.policy import CONFIRM_S, Pending, confirm_text, decide
from riya.core.hands.records import HandsDB
from riya.core.hands.tools import REGISTRY, TIERS, ToolCtx, validate_call
from riya.core.settings import Settings
from riya.core.store import open_db

BACKUP = {"name": "backup", "argv": ["robocopy", "C:\\Users\\me\\Documents", "D:\\backup", "/E"], "cwd": None,
          "timeout_s": 30}
DRY_TURNS = [("open", "open youtube"), ("volume", "volume up"), ("run", "run backup"),
             ("write", "write a file called x saying y")]


class DryRunEffects:
    """Only prints what it would have done. The check never calls it."""

    def _say(self, name: str, *a) -> None:
        print(f"  [dry run] would {name}{a!r}", flush=True)

    def open_url(self, url): self._say("open_url", url)
    def open_app(self, target): self._say("open_app", target)
    def media_key(self, key, presses=1): self._say("media_key", key, presses)
    def copy(self, text): self._say("copy", text)

    def run(self, argv, cwd, timeout_s):
        self._say("run", argv, cwd, timeout_s)
        return None, "dry run"


def _t0(city: str) -> list[tuple[str, str]]:
    return [("time", "what time is it"), ("timer", "set a timer for 1 second"), ("calc", "what's 12 times 12"),
            ("weather", f"what's the weather in {city}"), ("lookup", "who was Alan Turing")]


def run_t0(name: str, phrase: str, ctx: ToolCtx) -> bool:
    """Match, validate and run one T0 turn; print the line. True when the tool reported ok."""
    now = ctx.clock.now()
    call = match(phrase, now, tz=ctx.tz)
    call = validate_call(call, ctx.settings) if call else None
    if call is None or TIERS.get(call.tool) != 0:
        print(f"  {name}: FAIL (\"{phrase}\" did not match a T0 tool)")
        return False
    r = REGISTRY[call.tool].run(call.args, ctx)
    ok = r.ok
    if name == "timer":  # it is on the temp db, 1 s ahead: due exactly then
        ok = ok and r.data["due_at"] - now < 2 and any(x.id == r.data["id"] for x in ctx.db.due(r.data["due_at"]))
    print(f"  {name}: {'ok' if ok else 'FAIL'} [{call.tool}] {r.text}", flush=True)
    return ok


def run_dry(name: str, phrase: str, settings: Settings, now: float) -> bool:
    """Match, validate, summarize and build the confirmation for one T1/T2 turn. Runs nothing."""
    names = lambda items: [i["name"] for i in items]
    call = match(phrase, now, apps=names(settings.apps), commands=names(settings.commands))
    if call is None:
        print(f"  {name}: FAIL (\"{phrase}\" did not match)")
        return False
    valid = validate_call(call, settings)
    if valid is None:
        print(f"  {name}: FAIL (matched {call.tool} but it did not validate)")
        return False
    summary = REGISTRY[valid.tool].summary(valid.args)
    pending = Pending("dry-run", valid, summary, now + CONFIRM_S)
    print(f"  {name}: ok [{valid.tool} T{TIERS[valid.tool]}, {decide(valid, settings)}] {confirm_text(pending)}", flush=True)
    return True


def main(argv=None, *, http=None, effects=None) -> int:
    ap = argparse.ArgumentParser(prog="evaluation.hands_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--city", default="London")
    args = ap.parse_args(argv)
    http, effects = http if http is not None else UrlHttp(), effects if effects is not None else DryRunEffects()
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="riya-hands-check-") as tmp:
        conn = open_db(Path(tmp) / "hands.db")
        try:
            clock, settings = SystemClock(), Settings()
            ctx = ToolCtx(settings=settings, db=HandsDB(conn), effects=effects, http=http, clock=clock, riya_dir=tmp)
            print("T0, run for real (the network is used only for weather and lookup):")
            for name, phrase in _t0(args.city):
                try:
                    ok = run_t0(name, phrase, ctx)
                except Exception as e:
                    print(f"  {name}: FAIL ({type(e).__name__})")
                    ok = False
                if not ok:
                    failed.append(name)
            print("T1 and T2, dry run (matched and validated; nothing is run):")
            dry = Settings(commands=[BACKUP], tools_ask=["open_url", "volume"])
            dry.tools_enabled = {n: True for n in TIERS}
            for name, phrase in DRY_TURNS:
                try:
                    ok = run_dry(name, phrase, dry, clock.now())
                except Exception as e:
                    print(f"  {name}: FAIL ({type(e).__name__})")
                    ok = False
                if not ok:
                    failed.append(name)
        finally:
            conn.close()
    print("VERDICT: ok" if not failed else f"VERDICT: FAIL {' '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
