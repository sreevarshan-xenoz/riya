"""Make an uploadable copy of a run_world state.pt without the replay buffer (file conversion only, no compute).

  python colab/slim_state.py scratch_runs/bench/train-v0/state.pt   -> state_slim.pt next to it
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis.world import Replay  # noqa: E402

for path in sys.argv[1:]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    st["replay"] = Replay(capacity=st["replay"].capacity)  # refills from fresh experience after resume
    out = os.path.join(os.path.dirname(path), "state_slim.pt")
    torch.save(st, out)
    print(out, f"{os.path.getsize(out) / 1e6:.1f} MB")
