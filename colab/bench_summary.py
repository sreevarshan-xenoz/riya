import glob
import json
import os

rows = []
for path in sorted(glob.glob("/content/runs/bench/*.json")):
    r = json.load(open(path))
    lc = r.get("long_chat", {})
    rows.append(f"{os.path.basename(path)[:58]:58s} vram {r.get('vram_mb')!s:>5} | ttft {r.get('ttft_ms')!s:>5} ms | "
                f"{r.get('tok_s')!s:>5} tok/s | refusals {r.get('refusals', {}).get('rate')} | "
                f"long-chat errors {lc.get('errors')} ttft {lc.get('ttft_start_ms')}->{lc.get('ttft_end_ms')} | "
                f"gate {r.get('gate', {}).get('pass')}")
print("\n".join(rows) or "no bench results yet")
tail = open("/content/runs/pass1.out").read().splitlines()[-3:]
print("\n".join(tail))
