"""Bridge: run the world-model agent for a while, read its affect state, and let the LM speak from it.

The affect vector enters the LM through `affect_proj`, which is zero-initialised: after plain pretraining it has
no effect yet. It starts to matter after fine-tuning on affect-labelled dialogue (next stage of the roadmap).
"""
import argparse

import tiktoken
import torch

from jarvis.lm import HybridLM, LMConfig
from jarvis.world import AFFECT_DIM, AffectCore, Agent, GlobalWorkspace, SurvivalGrid, WorldModel

p = argparse.ArgumentParser()
p.add_argument("--lm", required=True)
p.add_argument("--world", required=True)
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--live_steps", type=int, default=60, help="steps the agent lives before speaking")
p.add_argument("--tokens", type=int, default=120)
a = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(a.world, map_location=device, weights_only=False)  # own checkpoint (contains numpy replay)
env = SurvivalGrid(seed=1)
wm = WorldModel(env.obs_shape).to(device)
ws = GlobalWorkspace(wm.dz, AFFECT_DIM, wm.n_actions).to(device)
wm.load_state_dict(ck["wm"]), ws.load_state_dict(ck["ws"])
affect = AffectCore()
agent = Agent(wm, ws, affect, wm.n_actions, device)
agent.memory = ck["memory"]
obs = env.reset()
for _ in range(a.live_steps):
    act, info = agent.act(obs)
    nxt, done, _ = env.step(act)
    err = float(wm.one_step_error(agent._t(obs["view"]), agent._t(obs["body"]), torch.tensor([act], device=device),
                                  agent._t(nxt["view"]), agent._t(nxt["body"])))
    affect.update(obs["body"], nxt["body"], err, info["s1_conf"])
    obs = env.reset() if done else nxt
print("internal state:", affect.describe())

lm_ck = torch.load(a.lm, map_location=device, weights_only=True)
model = HybridLM(LMConfig(**lm_ck["config"])).to(device).eval()
model.load_state_dict(lm_ck["model"])
enc = tiktoken.get_encoding("gpt2")
idx = torch.tensor([enc.encode_ordinary(a.prompt)], device=device)
out = model.generate(idx, a.tokens, affect=torch.as_tensor(affect.state.vector()[None], device=device))
print(enc.decode(out[0].tolist()))
