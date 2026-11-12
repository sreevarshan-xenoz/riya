"""The Colab side of Riya v1: the teacher and the trainer, tested through their pure helpers only (no GPU, no
network, no llama-server, no torch/peft/transformers)."""
import ast
import json
from pathlib import Path

from colab import gen_riya_v1 as gen
from colab import train_riya_v1 as tr
from riya.train import fmt
from riya.train.holdout import probe_texts

ROOT = Path(__file__).resolve().parents[2]


def test_lora_config():
    assert tr.peft_config_kwargs() == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                                       "task_type": "CAUSAL_LM", "target_modules": tr.TARGET_REGEX}
    assert tr.peft_config_kwargs() is not tr.LORA  # a copy
    s = tr.train_settings(False)
    assert (s["dtype"], s["epochs"], s["lr"], s["scheduler"], s["max_len"]) == ("float16", 2, 2e-4, "cosine", 2048)
    assert s["warmup_ratio"] == 0.03 and s["batch"] == 4 and s["grad_accum"] == 4 and s["seed"] == 0
    assert tr.train_settings(True)["dtype"] == "bfloat16"
    assert tr.TRAIN == {"epochs": 2, "lr": 2e-4, "scheduler": "cosine", "warmup_ratio": 0.03, "max_len": 2048,
                        "batch": 4, "grad_accum": 4, "seed": 0}  # train_settings doesn't mutate TRAIN


def test_targets_skip_the_short_conv_layers():
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj",
             "model.layers.0.self_attn.out_proj", "model.layers.0.self_attn.q_layernorm",
             "model.layers.0.feed_forward.w1", "model.layers.0.feed_forward.w2", "model.layers.0.feed_forward.w3",
             "model.layers.1.conv.in_proj", "model.layers.1.conv.out_proj", "model.layers.1.conv.conv",
             "lm_head", "model.embed_tokens"]
    assert tr.selected_modules(iter(names)) == [n for n in names[:8] if "layernorm" not in n]
    assert len(tr.selected_modules(names)) == 7
    assert tr.TARGET_MODULES == ("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3")
    assert tr.selected_modules(["model.layers.1.conv.out_proj", "x.conv.self_attn.q_proj.weight"]) == []


def test_resolve_base_order():
    def boom(_q):
        raise AssertionError("the search must not run when BASE_ID is set")

    assert tr.resolve_base({"BASE_ID": " x/y "}, boom) == ("x/y", True)  # pinned, but not the weights she serves
    assert tr.resolve_base({"BASE_ID": tr.DEFAULT_BASE}, boom) == (tr.DEFAULT_BASE, False)
    fork = "someone/huihui-lfm2.5-1.2b-instruct-abliterated"  # the same family name, another uploader
    assert tr.resolve_base({"BASE_ID": fork}, boom) == (fork, False)
    seen = []

    def hits(h):
        def search(q):
            seen.append(q)
            return h
        return search

    assert tr.resolve_base({}, hits(["a/b", tr.DEFAULT_BASE])) == (tr.DEFAULT_BASE, False)
    assert seen == [tr.BASE_SEARCH]
    assert tr.resolve_base({"BASE_ID": "  "}, hits(["a/b"])) == ("a/b", True)  # any other hit is a mismatch
    assert tr.resolve_base({}, hits(["a/b", fork])) == (fork, False)  # a family match beats a first hit
    assert tr.resolve_base({}, hits(["x/Huihui-LFM2.5-1.2B-Instruct-abliterated-v2"])) == (
        "x/Huihui-LFM2.5-1.2B-Instruct-abliterated-v2", True)
    assert tr.mismatch_banner("a/b").startswith("BASE MISMATCH: training on a/b,")
    assert tr.resolve_base({}, hits([])) == (tr.OFFICIAL_BASE, True)

    def raises(_q):
        raise OSError("offline")

    assert tr.resolve_base({}, raises) == (tr.OFFICIAL_BASE, True)
    assert tr.DEFAULT_BASE == "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated"
    assert tr.OFFICIAL_BASE == "LiquidAI/LFM2.5-1.2B-Instruct"


def test_convert_uses_the_resolved_base():
    cmd = tr.convert_cmd("L", "A", "B", "O/riya-v1-lora.gguf")
    assert cmd[1:] == ["L/convert_lora_to_gguf.py", "A", "--base", "B", "--outfile", "O/riya-v1-lora.gguf",
                       "--outtype", "f16"]


def test_smoke_problems():
    p = "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert tr.smoke_problems(["/x/riya-v1-lora.gguf"], "/y/riya-v1-lora.gguf", p, p) == []
    assert tr.smoke_problems([], "/y/riya-v1-lora.gguf", p, p) == ["llama-server didn't load the adapter"]
    assert tr.smoke_problems(["/x/other.gguf"], "/y/riya-v1-lora.gguf", p, p) == [
        "llama-server didn't load the adapter"]
    assert tr.smoke_problems(["/x/riya-v1-lora.gguf"], "/y/riya-v1-lora.gguf", p, p + "\n") == [
        "the HF chat template differs from llama-server's"]
    assert tr.smoke_problems(["C:\\x\\riya-v1-lora.gguf"], "/y/riya-v1-lora.gguf", p, p[len("<|startoftext|>"):]) == []
    assert len(tr.smoke_problems([], "/y/a.gguf", "a", "b")) == 2
    cmd = tr.smoke_server_cmd("srv", "base.gguf", "lora.gguf")
    assert cmd == ["srv", "-m", "base.gguf", "--lora", "lora.gguf", "--port", "8096", "-c", "4096", "-ngl", "99",
                   "--no-webui"]
    assert "--lora-scaled" not in cmd and cmd[cmd.index("--lora") + 1] == "lora.gguf"


def test_report():
    r = tr.make_report(base_id=tr.OFFICIAL_BASE, base_mismatch=True, rows=100, skipped=2,
                       losses=[(1, 2.5), (2, 1.25)], smoke=False, gpu="Tesla T4", settings=tr.train_settings(False))
    assert json.loads(json.dumps(r)) == r
    assert r["base_mismatch"] is True and r["base_id"] == tr.OFFICIAL_BASE
    assert r["loss_curve"] == [[1, 2.5], [2, 1.25]]
    assert r["lora"] == tr.peft_config_kwargs() and r["adapter"] == "riya-v1-lora.gguf"
    assert (r["rows"], r["skipped_too_long"], r["smoke"], r["gpu"]) == (100, 2, False, "Tesla T4")
    assert r["train"]["dtype"] == "float16"
    assert r["holdout_dropped"] == 0
    assert tr.make_report(base_id=tr.OFFICIAL_BASE, base_mismatch=False, rows=1, skipped=0, losses=[], smoke=False,
                          gpu="T4", settings=tr.train_settings(False), holdout_dropped=3)["holdout_dropped"] == 3


def _sft(i: int, user: str) -> dict:
    return {"id": f"r{i}", "kind": "small", "messages": [{"role": "system", "content": "sys"},
                                                          {"role": "user", "content": user},
                                                          {"role": "assistant", "content": "Hmph, hi~"}]}


def test_the_loader_drops_and_counts_a_probe_overlap(tmp_path):
    """A9 (CPU): the trainer drops rows that overlap a probe, before the smoke slice, and counts them."""
    rows = [_sft(0, "good morning to you"), _sft(1, probe_texts()[0]), _sft(2, "i made toast today")]
    p = tmp_path / "sft.jsonl"
    fmt.write_jsonl(rows, p)
    assert tr.load_sft(p) == ([rows[0], rows[2]], 1)
    many = [_sft(i, f"message number {i} here") for i in range(tr.SMOKE_ROWS + 4)]
    many.insert(1, _sft(99, probe_texts()[0]))
    fmt.write_jsonl(many, p)
    kept, n = tr.load_sft(p, smoke=True)
    assert n == 1 and len(kept) == tr.SMOKE_ROWS and "r99" not in {r["id"] for r in kept}  # dropped before slicing
    assert [r["id"] for r in kept] == [f"r{i}" for i in range(tr.SMOKE_ROWS)]


def test_the_run_uses_the_loader_and_reports_the_drop():
    src = (ROOT / "colab" / "train_riya_v1.py").read_text(encoding="utf-8")
    run = src.split("def _run(", 1)[1].split("\ndef ", 1)[0]
    assert "load_sft(" in run and "rows dropped as probe overlaps" in run and "holdout_dropped=" in run
    assert "riya.core" not in src


def test_teacher_helpers():
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    b = gen.teacher_body(m)
    assert b["messages"] is m and b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["max_tokens"] == gen.TEACHER_TOKENS >= 400  # ruling CONFLICT 4: room for an unclipped reply
    assert b["temperature"] == gen.TEMPERATURE == 0.8
    assert gen.clean_response('<think>hm</think>  "Hi~" ') == "Hi~"
    assert gen.clean_response("<think>a</think>x<think>b</think> ok ") == "ok"
    assert gen.clean_response('"quoted" and "more"') == 'quoted" and "more'  # one pair only
    assert gen.clean_response("Riya: hi") == "Riya: hi"  # labels are the filter's business
    reqs = [{"id": "a", "h": "1" * 64}, {"id": "b", "h": "2" * 64}, {"id": "c", "h": "3" * 64}]
    assert gen.todo(reqs, {("b", "2" * 64)}) == [reqs[0], reqs[2]]
    assert gen.todo(reqs, {("b", "9" * 64), ("c", "3" * 64)}) == reqs[:2]  # an answer to another b isn't b's
    assert gen.todo(reqs, set()) == reqs
    assert gen.online_gap_s() == 6.0 and gen.online_gap_s(30) == 2.0
    assert (gen.PORT, gen.SLOTS, gen.ONLINE_RPM) == (8095, 8, 10)


def test_batching_and_resume(tmp_path, monkeypatch):
    from riya.train import fmt

    reqs = [{"id": f"r{i}", "h": f"{i}" * 64,
             "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"u{i}"}]} for i in range(5)]
    req_path, out = tmp_path / "requests.jsonl", tmp_path / "teacher.jsonl"
    req_path.write_text("".join(json.dumps(r) + "\n" for r in reqs), encoding="utf-8")
    out.write_text(json.dumps({"id": "r0", "h": "0" * 64, "response": "old"}) + "\n", encoding="utf-8")
    calls = []

    def fake_post(url, body, timeout=300):
        calls.append((url, body))
        return '<think>x</think> "answer ' + body["messages"][-1]["content"] + '~"'

    monkeypatch.setattr(gen, "_post_chat", fake_post)
    monkeypatch.setattr(gen, "validate_request", lambda r: None)  # these toy rows aren't full requests
    assert gen.main(["--requests", str(req_path), "--out", str(out), "--url", "http://x:1", "--limit", "3"]) == 0
    rows = fmt.read_jsonl(out)
    assert [r["id"] for r in rows] == ["r0", "r1", "r2"]  # --limit 3 reads r0..r2; r0 was done, so it isn't redone
    assert rows[0]["response"] == "old" and rows[1]["response"] == "answer u1~"
    assert [r["h"] for r in rows] == ["0" * 64, "1" * 64, "2" * 64]  # each answer carries its request's h
    assert sorted(c[1]["messages"][-1]["content"] for c in calls) == ["u1", "u2"]
    assert {c[0] for c in calls} == {"http://x:1/v1/chat/completions"}
    assert all(c[1]["max_tokens"] == gen.TEACHER_TOKENS for c in calls)
    for r in rows:
        fmt.validate_teacher(r)
    assert gen.main(["--requests", str(req_path), "--out", str(out), "--url", "http://x:1", "--limit", "3"]) == 0
    assert len(calls) == 2  # a second run finds everything done


def test_online_fallback_is_off_unless_asked_and_throttled(tmp_path):
    reqs = [{"id": f"r{i}", "h": f"{i}" * 64, "messages": [{"role": "user", "content": f"u{i}"}]} for i in range(3)]

    def down(_body):
        raise OSError("teacher down")

    online_calls = []
    clock = {"t": 100.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    def online(body):
        online_calls.append(clock["t"])
        return "online " + body["messages"][-1]["content"]

    out = tmp_path / "t.jsonl"
    # off: a teacher failure writes nothing (the row is retried on the next run)
    assert gen.generate(reqs, out, down, None, workers=1) == (0, 3)
    assert not out.exists() or out.read_text() == ""
    # on: each failed request goes online, at most one every 6 s
    done, failed = gen.generate(reqs, out, down, online, gap_s=6.0, clock=lambda: clock["t"], sleep=sleep, workers=1)
    assert (done, failed) == (3, 0)
    assert all(b - a >= 6.0 - 1e-9 for a, b in zip(online_calls, online_calls[1:])) and len(online_calls) == 3
    assert sum(slept) >= 12.0 - 1e-9
    assert [json.loads(line)["response"] for line in out.read_text().splitlines()] == ["online u0", "online u1",
                                                                                         "online u2"]


def test_online_env_gate():
    assert gen.online_enabled({"TEACHER_ONLINE": "1"}) is True  # OpenCode's free tier needs no key
    assert gen.online_enabled({"TEACHER_ONLINE": "1", "RIYA_ONLINE_API_KEY": "k"}) is True
    assert gen.online_enabled({"TEACHER_ONLINE": "1", "RIYA_ONLINE_API_KEY": " "}) is False
    assert gen.online_enabled({"TEACHER_ONLINE": "1", "RIYA_ONLINE_BASE_URL": "https://llm.example.test/v1"}) is False
    assert gen.online_enabled({"RIYA_ONLINE_API_KEY": "k"}) is False
    assert gen.online_enabled({"TEACHER_ONLINE": "0", "RIYA_ONLINE_API_KEY": "k"}) is False


def test_no_gpu_stack_at_import():
    banned = {"torch", "peft", "transformers", "bitsandbytes", "accelerate"}
    for name in ("gen_riya_v1.py", "train_riya_v1.py"):
        src = (ROOT / "colab" / name).read_text(encoding="utf-8")
        for node in ast.parse(src).body:
            if isinstance(node, ast.Import):
                mods = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not mods & banned, (name, mods)
        for word in ("bitsandbytes", "load_in_4bit", "--lora-scaled"):
            assert word not in src, (name, word)


BASE_GGUF = "/content/models/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf"


def _runbook() -> str:
    return (ROOT / "colab" / "riya_v1" / "README.md").read_text(encoding="utf-8")


def test_runbook_cells():
    text = _runbook()
    pos = -1
    for needle in ("riya.train.build_sft", "sync_to_drive.py", "pip uninstall -y torchao", "colab.gen_riya_v1",
                   "riya.train.filter_sft", "colab.train_riya_v1 --smoke", "SMOKE_OK", "colab.train_riya_v1",
                   "base_mismatch",
                   "## 7. Colab: the gate's verdict", "git archive --format=zip", "/content/riya_code.zip",
                   "11731cb4", BASE_GGUF, "LLAMA_SERVER=/content/bin/llama-server",
                   "PYTHONPATH=/content/code", "--verdict-out /content/drive/MyDrive/riya_v1/verdict-$SHA8.json",
                   "--cache-ram", "LLAMA_ARG_CACHE_RAM=0",
                   "## 8. Colab: calibrating the gate", "gate.identity()",
                   "for n in 1;", "--base-vs-base", "gate/bank-$n.json", "for n in 2 3;",
                   "evaluation.gate_calibrate --bank scratch_runs/riya_v1/gate/bank",
                   "evaluation/gate_calibration.json",
                   "## 9. Local: adopt from the verdict",
                   "C:/Users/sreev/models/riya/riya-v1-candidate-lora.gguf",
                   "--check-verdict", "--adopt-verdict", "-ngl 0"):
        nxt = text.find(needle, pos + 1)
        assert nxt > pos, f"{needle!r} is missing or out of order"
        pos = nxt
    assert "drive.mount" in text and "pip install" in text and "peft" in text
    assert "riya.train.fmt --check-template" not in text  # the adopt checks her own build's template itself
    # a new adapter goes to the candidate path; only the gate installs it as the adopted file (C1)
    copies = [line for line in text.splitlines() if line.lstrip().startswith("Copy-Item")]
    to_models = [line for line in copies if "models/riya" in line]
    assert to_models and all(line.rstrip().endswith('riya-v1-candidate-lora.gguf"') for line in to_models)


def test_runbook_installs_the_pinned_embedder():
    """v0.11 §4.7.4: the gate's embedder is the pinned bge, so the runbook installs fastembed, points RIYA_MODEL_DIR at a
    kept folder and verifies the model before the first champion command; section 8 compares five code fields."""
    text = _runbook()
    first = text.index("python -m evaluation.champion")
    assert 0 <= text.index("pip install -q fastembed==0.9.0") < first
    assert 0 <= text.index("RIYA_MODEL_DIR=") < first
    assert text.index("FastEmbedder(verify=True)") < first
    sec8 = text.split("## 8. Colab: calibrating the gate", 1)[1].split("## 9.", 1)[0]
    ident = sec8.split("**Then the bank.**", 1)[0]
    assert "five code fields" in ident and "`embedder`" in ident
    assert "FastEmbedder(verify=True).embedder_id" in ident
    assert "five live" in sec8 and "four code fields" not in text and "three live" not in text
    assert "/tmp/riya/models" in text and "models--Qdrant--bge-small-en-v1.5-onnx-Q" in text
    # every champion command (and only those that run probes) carries the model dir
    for cmd in _joined_commands(text):
        if "python -m evaluation.champion" in cmd and ("--verdict-out" in cmd):
            assert "RIYA_MODEL_DIR=" in cmd, cmd
    # the bank's second block is separate, so the first two loops keep their needles
    assert text.index("for n in 1;") < text.index("for n in 2 3;") < text.index("for n in 4 5;")


def test_runbook_final_review_notes():
    """Final review M2-M4: a snapshot-only pin mismatch points to the offline copy, the laptop's embedder check uses
    the venv's python, and the filter step says --allow-stale is safe for the pre-v0.11 requests."""
    text = _runbook()
    sec7 = text.split("## 7. Colab: the gate's verdict", 1)[1].split("## 8.", 1)[0]
    assert "stop there" not in sec7 and "expected aa8f8b06" in sec7 and "offline copy" in sec7
    laptop = [c for c in _joined_commands(text) if "FastEmbedder(verify=True).embedder_id" in c and "%%bash" not in c
              and "/content" not in c]
    assert laptop and all(".venv\\Scripts\\python -c" in c for c in laptop), laptop
    sec4 = text.split("## 4. Local: filter the teacher's answers", 1)[1].split("## 5.", 1)[0]
    assert "--allow-stale" in sec4 and "v0.11" in sec4


def _joined_commands(text: str) -> list[str]:
    """The README's lines with every continuation (a trailing backslash in bash, a backtick in PowerShell) joined onto
    the next line, so a flag on a continuation line is checked like any other."""
    out, cur = [], ""
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\") or stripped.endswith("`"):
            cur += stripped[:-1] + " "
            continue
        out.append((cur + stripped).strip())
        cur = ""
    if cur:
        out.append(cur.strip())
    return out


def test_the_joiner_follows_both_continuations():
    text = "a --x \\\n  --y\nb --p `\n  --q\nc --z"
    assert _joined_commands(text) == ["a --x    --y", "b --p    --q", "c --z"]


def test_runbook_commands_match_the_code():
    import re
    from evaluation import champion
    text = _runbook()
    known = set(champion.parser()._option_string_actions)
    cmds = [c for c in _joined_commands(text) if "python -m evaluation.champion" in c]
    assert cmds
    for cmd in cmds:
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z-]*", cmd):
            assert flag in known, f"{flag} isn't a champion flag: {cmd}"
    wanted = {
        "a candidate run": lambda c: "--verdict-out" in c and "--base-vs-base" not in c and "--lora" in c,
        "base-vs-base": lambda c: "--base-vs-base" in c and "--verdict-out" in c,
        "check": lambda c: "--check-verdict" in c,
        "adopt": lambda c: "--adopt-verdict" in c,
    }
    for name, has in wanted.items():
        assert any(has(c) for c in cmds), f"no {name} command in the runbook"
    assert any("python -m evaluation.gate_calibrate" in c for c in _joined_commands(text))
    assert "gate.identity()" in text


def test_batching_helpers():
    from riya.train.fmt import IGNORE_INDEX

    ids, labels, mask = tr.pad_batch([([1, 2, 3], [-100, 2, 3]), ([4], [4])], pad_id=0)
    assert ids == [[1, 2, 3], [4, 0, 0]] and mask == [[1, 1, 1], [1, 0, 0]]
    assert labels == [[IGNORE_INDEX, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]
    b = tr.epoch_batches(10, 4, seed=0, epoch=0)
    assert [len(x) for x in b] == [4, 4, 2] and sorted(i for x in b for i in x) == list(range(10))
    assert b == tr.epoch_batches(10, 4, seed=0, epoch=0) and b != tr.epoch_batches(10, 4, seed=0, epoch=1)
    assert tr.total_steps(10, 2, 4, 4) == 2  # 3 micro-batches per epoch fit in one accumulation group
    assert tr.total_steps(100, 2, 4, 4) == 14  # 25 micro-batches -> 7 groups per epoch


def test_only_full_precision_repos_count_as_the_base():
    assert tr.plain_weights("huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated")
    for bad in ("mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF", "x/Huihui-LFM2.5-abliterated-MLX-4bit"):
        assert not tr.plain_weights(bad)


def test_precision_follows_the_compute_capability_not_bf16_emulation():
    assert tr.bf16_capable((7, 5)) is False  # a T4: torch.cuda.is_bf16_supported() says True there by emulation
    assert tr.bf16_capable((7, 0)) is False
    assert all(tr.bf16_capable(c) for c in ((8, 0), (8, 6), (8, 9), (9, 0)))
    assert tr.train_settings(tr.bf16_capable((7, 5)))["dtype"] == "float16"
    assert tr.train_settings(tr.bf16_capable((8, 0)))["dtype"] == "bfloat16"


def test_resume_survives_a_broken_trailing_line(tmp_path, monkeypatch, capsys):
    from riya.train import fmt

    reqs = [{"id": f"r{i}", "h": f"{i}" * 64, "messages": [{"role": "user", "content": f"u{i}"}]} for i in range(4)]
    req_path, out = tmp_path / "requests.jsonl", tmp_path / "teacher.jsonl"
    req_path.write_text("".join(json.dumps(r) + "\n" for r in reqs), encoding="utf-8")
    out.write_text(json.dumps({"id": "r0", "h": "0" * 64, "response": "a"}) + "\n"
                   + json.dumps({"id": "r1", "h": "1" * 64, "response": "b"})
                   + "\n" + '{"id": "r2", "respo', encoding="utf-8")  # the run was killed mid-write
    rows, dropped = fmt.read_jsonl_checked(out)
    assert [r["id"] for r in rows] == ["r0", "r1"] and dropped == 1
    assert fmt.read_jsonl(out) == rows
    assert "broken last line" in capsys.readouterr().err
    monkeypatch.setattr(gen, "_post_chat", lambda url, body, timeout=300: "fresh")
    monkeypatch.setattr(gen, "validate_request", lambda r: None)
    assert gen.main(["--requests", str(req_path), "--out", str(out), "--url", "http://x:1"]) == 0
    rows = fmt.read_jsonl(out)  # r2 and r3 were appended on clean lines
    assert [(r["id"], r["response"]) for r in rows] == [("r0", "a"), ("r1", "b"), ("r2", "fresh"), ("r3", "fresh")]
    assert out.read_text(encoding="utf-8").endswith("\n") and len(out.read_text(encoding="utf-8").splitlines()) == 4
    # a valid last line that merely lacks its newline is fixed too, and a broken middle line still raises
    line = json.dumps({"id": "r0", "h": "0" * 64, "response": "a"})
    out.write_text(line, encoding="utf-8")
    assert gen.resume_keys(out) == {("r0", "0" * 64)} and out.read_text(encoding="utf-8") == line + "\n"
    out.write_text('{"id": "r0"\n{"id": "r1", "response": "b"}\n', encoding="utf-8")
    with __import__("pytest").raises(json.JSONDecodeError):
        fmt.read_jsonl(out)
    assert gen.resume_keys(tmp_path / "missing.jsonl") == set()


def test_an_interrupt_drops_the_queue_and_keeps_the_flushed_rows(tmp_path, monkeypatch):
    import time

    shutdowns = []

    class Spy(gen.ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdowns.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(gen, "ThreadPoolExecutor", Spy)
    out = tmp_path / "t.jsonl"
    reqs = [{"id": f"r{i}", "h": f"{i}" * 64, "messages": [{"role": "user", "content": f"u{i}"}]} for i in range(6)]
    calls = []

    def chat(body):
        calls.append(body)
        if len(calls) == 3:  # Ctrl-C arrives once two rows are answered; each must already be on disk
            for _ in range(200):
                if out.exists() and len(out.read_text(encoding="utf-8").splitlines()) >= 2:
                    break
                time.sleep(0.01)
            raise KeyboardInterrupt
        if len(calls) > 3:
            time.sleep(0.3)  # a slow teacher: the main thread has time to cancel the queue
        return "ok"

    with __import__("pytest").raises(KeyboardInterrupt):
        gen.generate(reqs, out, chat, workers=1)
    assert shutdowns and shutdowns[-1] == (False, True)
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["r0", "r1"]
    assert len(calls) < 6  # the queued requests were cancelled, not drained


def test_filter_sft_reads_the_teacher_file_tolerantly():
    from riya.train import filter_sft, fmt

    assert filter_sft.read_jsonl_checked is fmt.read_jsonl_checked  # the one loader that skips a cut-off last line


def test_the_teacher_regenerates_after_a_rebuild(tmp_path, monkeypatch, capsys):
    """I1: requests rebuilt under another seed keep their ids but not their content. The teacher resumes only an
    answer whose h matches the request as it is now, so every changed request is answered again."""
    from riya.train import fmt
    from riya.train.build_sft import build

    old, new = build(0, 24, ("small",)), build(1, 24, ("small",))
    old_h = {r["id"]: r["h"] for r in old}
    changed = [r["id"] for r in new if r["h"] != old_h[r["id"]]]
    assert changed and [r["id"] for r in old] == [r["id"] for r in new]
    req_path, out = tmp_path / "requests.jsonl", tmp_path / "teacher.jsonl"
    fmt.write_jsonl(old, req_path)
    asked = []
    monkeypatch.setattr(gen, "_post_chat", lambda url, body, timeout=300: asked.append(body) or "Hmph, hi~")
    assert gen.main(["--requests", str(req_path), "--out", str(out), "--url", "http://x:1"]) == 0
    rows = fmt.read_jsonl(out)
    assert len(asked) == len(old) and all(r["h"] == old_h[r["id"]] for r in rows)
    for r in rows:
        fmt.validate_teacher(r)
    with out.open("a", encoding="utf-8") as f:  # an answer from before h existed is never taken as done
        f.write(json.dumps({"id": new[0]["id"], "response": "legacy"}) + "\n")
    fmt.write_jsonl(new, req_path)
    asked.clear()
    capsys.readouterr()
    assert gen.main(["--requests", str(req_path), "--out", str(out), "--url", "http://x:1"]) == 0
    assert sorted(b["messages"][-1]["content"] for b in asked) == sorted(
        r["messages"][-1]["content"] for r in new if r["id"] in changed)
    assert f"{len(new) - len(changed)} done, {len(changed)} to generate" in capsys.readouterr().out
    fresh = [r for r in fmt.read_jsonl(out) if r.get("h") in {x["h"] for x in new}]
    assert {r["id"] for r in fresh} == {r["id"] for r in new}


def test_every_json_write_is_atomic():
    """Final-review minor: riya/train, the v0.8 Colab scripts and the champion write JSON/JSONL only through
    fmt.write_jsonl / fmt.write_json (a temp file, then os.replace) or brain.write_record; the teacher's own
    append-and-flush is the one exception."""
    files = [*sorted((ROOT / "riya" / "train").glob("*.py")), ROOT / "colab" / "gen_riya_v1.py",
             ROOT / "colab" / "train_riya_v1.py", ROOT / "evaluation" / "champion.py"]
    for f in files:
        src = f.read_text(encoding="utf-8")
        if f.name == "fmt.py":
            src = src.split("def _write_atomic", 1)[0] + src.split("def write_json(", 1)[1].split("\ndef ", 1)[1]
        assert ".write_text(" not in src and "json.dump(" not in src, f.name
        assert not __import__("re").search(r"""\.open\(\s*["']w""", src), f.name
    gen_src = (ROOT / "colab" / "gen_riya_v1.py").read_text(encoding="utf-8")
    assert 'out.open("a"' in gen_src and "f.flush()" in gen_src  # the teacher appends and flushes each row
    train_src = (ROOT / "colab" / "train_riya_v1.py").read_text(encoding="utf-8")
    assert "write_json(" in train_src


# ---- final fix wave minor: the trainer aborts on a non-finite loss ----

def scripted(values):
    """A fake micro-step: each call returns the next loss and a backward() that records it."""
    seen, backward = [], []

    def micro(idx):
        v = values[len(seen)]
        seen.append(list(idx))
        return v, lambda: backward.append(v)
    return micro, seen, backward


def test_the_training_loop_groups_micro_batches_and_stops_at_steps():
    micro, seen, backward = scripted([2.0, 4.0, 1.0, 3.0, 9.0, 9.0])
    steps_done, logged = [], []
    losses = tr.train_loop(lambda epoch: [[0, 1], [2, 3], [4]], epochs=2, accum=2, steps=3, micro=micro,
                           opt_step=lambda: steps_done.append(len(seen)), log=lambda s, v: logged.append((s, v)))
    assert losses == [(1, 3.0), (2, 1.0), (3, 6.0)]  # [2, 4], the epoch's short last group [1], then [3, 9]
    assert steps_done == [2, 3, 5] and seen == [[0, 1], [2, 3], [4], [0, 1], [2, 3]] and logged == losses
    assert backward == [2.0, 4.0, 1.0, 3.0, 9.0]


def test_a_non_finite_loss_aborts_the_run():
    for bad in (float("nan"), float("inf"), float("-inf")):
        micro, seen, backward = scripted([2.0, 2.0, bad, 1.0])
        with __import__("pytest").raises(tr.NonFiniteLoss) as e:
            tr.train_loop(lambda epoch: [[0], [1], [2], [3]], epochs=1, accum=2, steps=10, micro=micro,
                          opt_step=lambda: None, log=lambda s, v: None)
        assert backward == [2.0, 2.0]  # the bad loss never reaches backward or the optimizer
        assert e.value.losses == [(1, 2.0)] and e.value.step == 2
        assert "non-finite loss" in str(e.value) and "step 2" in str(e.value)
    r = tr.make_report(base_id=tr.DEFAULT_BASE, base_mismatch=False, rows=4, skipped=0, losses=[(1, 2.0)],
                       smoke=False, gpu="Tesla T4", settings=tr.train_settings(False), aborted=tr.NON_FINITE)
    assert r["aborted"] == "non-finite loss" == tr.NON_FINITE and r["loss_curve"] == [[1, 2.0]]
    ok = tr.make_report(base_id=tr.DEFAULT_BASE, base_mismatch=False, rows=4, skipped=0, losses=[], smoke=False,
                        gpu="T4", settings=tr.train_settings(False))
    assert ok["aborted"] is None


def test_the_run_writes_the_aborted_report_before_stopping():
    src = (ROOT / "colab" / "train_riya_v1.py").read_text(encoding="utf-8")
    run = src.split("def _run(", 1)[1].split("\ndef ", 1)[0]
    assert "train_loop(" in run and "except NonFiniteLoss" in run and "aborted=NON_FINITE" in run
    assert run.index("except NonFiniteLoss") < run.index("save_pretrained")  # no adapter is exported after it


def test_online_post_sends_a_client_user_agent(monkeypatch):
    """opencode.ai's edge answers urllib's default UA with 403, so the teacher fallback must send ours."""
    from riya.core.models import USER_AGENT

    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(req, timeout=300):
        seen["ua"] = req.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(gen.urllib.request, "urlopen", fake_urlopen)
    assert gen._post("https://opencode.ai/inference/openai/v1/chat/completions", {"messages": []}) == "ok"
    assert seen["ua"] == USER_AGENT and not seen["ua"].startswith("Python-urllib")
