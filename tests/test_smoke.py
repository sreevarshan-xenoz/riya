import numpy as np
import torch

from jarvis.lm import PRESETS, HybridLM
from jarvis.lm.layers import chunk_gated_delta_rule, chunk_gated_linear_attention
from jarvis.lm.optim import build_optimizers
from jarvis.world import AFFECT_DIM, AffectCore, Agent, GlobalWorkspace, Replay, SurvivalGrid, WorldModel


def _inputs(B=2, H=3, T=37, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, generator=g), dim=-1)
    v = torch.randn(B, H, T, D, generator=g)
    log_a = -torch.rand(B, H, T, generator=g) * 0.3
    beta = torch.rand(B, H, T, generator=g)
    return q.double(), k.double(), v.double(), log_a.double(), beta.double()


def test_gated_linear_attention_matches_recurrence():
    q, k, v, log_a, _ = _inputs()
    S = torch.zeros(2, 3, 8, 8, dtype=torch.double)
    ref = []
    for t in range(q.shape[2]):
        S = log_a[:, :, t, None, None].exp() * S + k[:, :, t, :, None] * v[:, :, t, None, :]
        ref.append((q[:, :, t, None, :] @ S).squeeze(-2))
    o, S_out = chunk_gated_linear_attention(q, k, v, log_a, chunk=8)
    assert torch.allclose(o, torch.stack(ref, 2), atol=1e-8)
    assert torch.allclose(S_out, S, atol=1e-8)


def test_gated_delta_rule_matches_recurrence():
    q, k, v, log_a, beta = _inputs()
    D = q.shape[-1]
    S = torch.zeros(2, 3, D, D, dtype=torch.double)
    eye = torch.eye(D, dtype=torch.double)
    ref = []
    for t in range(q.shape[2]):
        kt, vt, bt, at = k[:, :, t], v[:, :, t], beta[:, :, t, None, None], log_a[:, :, t, None, None].exp()
        S = at * ((eye - bt * kt[..., :, None] * kt[..., None, :]) @ S) + bt * kt[..., :, None] * vt[..., None, :]
        ref.append((q[:, :, t, None, :] @ S).squeeze(-2))
    o, S_out = chunk_gated_delta_rule(q, k, v, log_a, beta, chunk=8)
    assert torch.allclose(o, torch.stack(ref, 2), atol=1e-8)
    assert torch.allclose(S_out, S, atol=1e-8)


def test_lm_forward_backward_and_generate():
    for pattern, bitnet in [("DRA", False), ("DDA", True)]:
        cfg = PRESETS["tiny"].__class__(dim=64, n_heads=4, pattern=pattern, chunk=16, bitnet=bitnet, vocab_size=128, block_size=40)
        model = HybridLM(cfg)
        x = torch.randint(0, 128, (2, 40))
        _, loss = model(x, x, affect=torch.zeros(2, cfg.affect_dim))
        loss.backward()
        opts = build_optimizers(model, 1e-3, 0.02)
        for o in opts:
            o.step()
        assert torch.isfinite(loss)
        assert model.generate(x[:, :5], 3).shape == (2, 8)


def test_lm_is_causal():
    cfg = PRESETS["tiny"].__class__(dim=64, n_heads=4, pattern="DRA", chunk=8, vocab_size=128)
    model = HybridLM(cfg).eval()
    x = torch.randint(0, 128, (1, 30))
    y = x.clone()
    y[0, 20:] = (y[0, 20:] + 1) % 128
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-5)


def test_context_belief_detects_rule_flip():
    from jarvis.world.context import ContextBelief, touched_channel
    from jarvis.world.env import FRUIT_A, N_CHANNELS

    view = np.zeros((N_CHANNELS, 5, 5), dtype=np.float32)
    view[FRUIT_A, 1, 2] = 1  # fruit A directly above the agent
    assert touched_channel(view, 1) == 0 and touched_channel(view, 2) is None
    cb = ContextBelief()
    healthy, hurt = np.array([0.7, 1.0]), np.array([0.7, 0.7])
    for _ in range(40):  # A is harmless for a long time
        cb.observe(view, 1, healthy, healthy)
    assert cb.p_harm()[0] < 0.15
    cps = [cb.observe(view, 1, healthy, hurt) for _ in range(5)]  # the rule flips: A now hurts
    assert max(cps) > 0.5  # a strong 'something changed' signal during the streak of contradictions
    assert cb.p_harm()[0] > 0.5  # belief revised within a handful of experiences


def test_world_loop_runs():
    env = SurvivalGrid(max_steps=40, seed=0)
    wm = WorldModel(env.obs_shape)
    ws = GlobalWorkspace(wm.dz, AFFECT_DIM, wm.n_actions)
    affect = AffectCore()
    agent = Agent(wm, ws, affect, wm.n_actions, "cpu", n_samples=32)
    replay = Replay(capacity=500, surprise_frac=0.5)
    for ep in range(3):
        obs, done = env.reset(), False
        while not done:
            a, info = agent.act(obs, random_policy=ep == 0)
            nxt, done, _ = env.step(a)
            affect.update(obs["body"], nxt["body"], 0.1, info["s1_conf"])
            replay.add(view=obs["view"].astype(np.uint8), body=obs["body"], action=np.int64(a),
                       next_view=nxt["view"].astype(np.uint8), next_body=nxt["body"],
                       dead=np.float32(0), episode=np.int64(ep), plan=np.int64(info["plan"]), affect=affect.state.vector(),
                       mem=info["mem"], next_valence=np.float32(affect.state.valence), surprise=np.float32(0.1))
            obs = nxt
    seq, starts = replay.sample_seq(16, 4, "cpu")
    loss, metrics, first_err = wm.loss(seq)
    loss.backward()
    replay.update_priority(starts, first_err.numpy())
    assert np.allclose(replay.priority[starts], first_err.numpy() + 1e-4)
    b = replay.sample(16, "cpu", ["view", "body", "plan", "affect", "mem", "next_valence", "surprise"])
    src = agent.sources(wm.encode(b["view"], b["body"]).detach(), b["affect"], b["mem"])
    ws_loss, _ = ws.loss(src, b["plan"].long(), b["plan"] >= 0, b["next_valence"], b["surprise"])
    ws_loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(ws_loss)
    assert isinstance(affect.describe(), str)
