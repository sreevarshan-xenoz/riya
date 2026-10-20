"""The Colab side of Riya v1: the teacher and the trainer, tested through their pure helpers only (no GPU, no
network, no llama-server, no torch/peft/transformers)."""
import ast
import json
from pathlib import Path

from colab import gen_riya_v1 as gen
from colab import train_riya_v1 as tr

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

    assert tr.resolve_base({"BASE_ID": " x/y "}, boom) == ("x/y", False)
    seen = []

    def hits(h):
        def search(q):
            seen.append(q)
            return h
        return search

    assert tr.resolve_base({}, hits(["a/b", tr.DEFAULT_BASE])) == (tr.DEFAULT_BASE, False)
    assert seen == [tr.BASE_SEARCH]
    assert tr.resolve_base({"BASE_ID": "  "}, hits(["a/b"])) == ("a/b", False)
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
    assert gen.online_enabled({"TEACHER_ONLINE": "1", "RIYA_ONLINE_API_KEY": "k"}) is True
    assert gen.online_enabled({"TEACHER_ONLINE": "1", "RIYA_ONLINE_API_KEY": " "}) is False
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


def test_runbook_cells():
    text = (ROOT / "colab" / "riya_v1" / "README.md").read_text(encoding="utf-8")
    pos = -1
    for needle in ("riya.train.build_sft", "sync_to_drive.py", "colab.gen_riya_v1", "riya.train.filter_sft",
                   "colab.train_riya_v1 --smoke", "SMOKE_OK", "colab.train_riya_v1", "base_mismatch",
                   "C:/Users/sreev/models/riya/riya-v1-candidate-lora.gguf", "evaluation.champion"):
        nxt = text.find(needle, pos + 1)
        assert nxt > pos, f"{needle!r} is missing or out of order"
        pos = nxt
    assert "drive.mount" in text and "pip install" in text and "peft" in text
    # a new adapter goes to the candidate path; only the gate installs it as the adopted file (C1)
    copies = [line for line in text.splitlines() if line.lstrip().startswith("Copy-Item")]
    to_models = [line for line in copies if "models/riya" in line]
    assert to_models and all(line.rstrip().endswith('riya-v1-candidate-lora.gguf"') for line in to_models)


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

    assert filter_sft.read_jsonl is fmt.read_jsonl  # the one loader that skips a cut-off last line


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
