"""Comment 3 trace: one EGAT-MAPPO training transition and one deterministic evaluation epoch, from the code.

Run from E:\\MESH:  PYTHONPATH=. python docs/paper/review/c3_trace.py > docs/paper/review/c3_trace.out.txt
CPU, n=20, S2, seed 0, sim settings airtime=domain, mac_eff=0.65, contention=0.0 (as runs/v10).
Final model: runs/v10/ckpt/egat_nogru_s0.pt (egat, use_gru=false), evaluated with switch_margin=2.0.
Nothing is trained or saved: the PPO buffer is filled for one episode and inspected, no update runs.
"""
import glob
import math
import re
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from mesh.model.agent import EGATAgent
from mesh.model.ko_prior import ko_action_idx
from mesh.rl.agents_io import K_SLOTS, N_DL_CH, decode, per_agent_reward, sequential_channels
from mesh.sim.env import NODE_FEATS, MeshEnv
from mesh.sim.scenarios import SUITE_A, SUITE_B, make_scenario

CKPT = "runs/v10/ckpt/egat_nogru_s0.pt"
SIM = dict(airtime="domain", mac_eff=0.65, contention=0.0)
MARGIN = 2.0
torch.set_grad_enabled(False)
np.set_printoptions(precision=3, suppress=True, linewidth=140)


def cfg_of(name, n=20):
    return replace(make_scenario(name, n=n), **SIM)


def load():
    a = EGATAgent.load(CKPT, device="cpu")
    assert not a.cfg.use_gru and a.phase == "ppo"
    return a


def stages(agent, p):
    """Re-run EGATNet.forward stage by stage (same modules, same order) and check it equals the net output."""
    net = agent.net
    t = {k: torch.as_tensor(p[k])[None] for k in ("x", "e", "adj", "ko_slot", "ko_chan", "slot_mask")}
    z = torch.relu(net.inp(t["x"]))
    for gat, norm in zip(net.gats, net.norms):
        z = torch.relu(norm(z + gat(z, t["e"], t["adj"])))
    route_raw, chan_raw = net.route(z)[0], net.chan(z)[0]
    kr, kc = nn.functional.softplus(net.beta_r).item(), nn.functional.softplus(net.beta_c).item()
    route_prior = route_raw - kr * torch.nan_to_num(t["ko_slot"][0], posinf=0.0)
    chan_prior = chan_raw - kc * torch.nan_to_num(t["ko_chan"][0], posinf=0.0)
    route_masked = route_prior.masked_fill(~t["slot_mask"][0], -1e9)
    out = agent._tensors([p], torch.zeros(1, len(p["x"]), agent.cfg.hidden))
    assert torch.allclose(out["route_logits"][0], route_masked) and torch.allclose(out["chan_logits"][0], chan_prior)
    return dict(route_raw=route_raw, chan_raw=chan_raw, kr=kr, kc=kc, route_prior=route_prior,
                chan_prior=chan_prior, route=route_masked, chan=chan_prior, value=out["value"][0], out=out)


def ent(logits):
    return torch.distributions.Categorical(logits=logits).entropy()


def pick_nodes(obs, p):
    """A gateway, a node with >=2 progress slots, and a node with exactly 1."""
    k = p["slot_mask"].sum(1)
    gw = int(obs.gateways[0])
    multi = [u for u in range(len(k)) if k[u] >= 2 and u not in obs.gateways]
    one = [u for u in range(len(k)) if k[u] == 1 and u not in obs.gateways and p["slot_mask"][u].any()]
    return [gw] + multi[:1] + one[:1]


def show_slots(obs, p, u):
    hop = obs.node_feat[:, NODE_FEATS.index("hop_to_gw")]
    s = p["slots"][u]
    print(f"    candidate slots (link nbrs by hop asc, SNR desc, id; -1 pad): {s.tolist()}")
    print(f"    slot hop_to_gw: {[int(hop[v]) if v >= 0 else None for v in s]}   own hop {int(hop[u])}"
          f"   current parent {obs.next_hop[u]}   current radio-1 channel {obs.channels[u, 1]}")
    print(f"    progress mask (valid & strictly closer; slot 0 forced if none): {p['slot_mask'][u].astype(int).tolist()}")


# ======================================================================================================
def training_transition():
    print("=" * 100)
    print("PART 1  TRAINING PATH  (agent.act(explore=True) -> env.step -> per_agent_reward -> agent.learn)")
    print("=" * 100)
    agent = load()
    env = MeshEnv(cfg_of("S2"), 0)
    obs = env.reset()
    lam = agent.reward_lam(100_000, 100_000)  # value of lambda_t at the end of training (0.5 -> 0.1)
    show_t = {10: "channel step", 11: "non-channel step"}
    teacher_total = teacher_dropped = 0
    while True:
        p = agent.prepare(obs)
        # teacher target as the BC phase would store it (Ko action mapped to slots), and whether BC uses it
        ko_a = ko_action_idx(obs, p["slots"], agent.ko)
        alive = p["alive"]
        allowed = p["slot_mask"][np.arange(len(ko_a)), ko_a % K_SLOTS]
        teacher_total += int(alive.sum())
        teacher_dropped += int((alive & ~allowed).sum())
        st = stages(agent, p) if obs.t in show_t else None
        action = agent.act(obs, explore=True)
        pend = agent._pending
        if st is not None:
            print(f"\n--- t={obs.t} ({show_t[obs.t]}; is_chan_step={p['is_chan_step']}, t_c={agent.cfg.t_c}) ---")
            print(f"  prior scales: kappa_route=softplus(beta_r)={st['kr']:.4f}  kappa_chan=softplus(beta_c)={st['kc']:.4f}")
            for u in pick_nodes(obs, p):
                print(f"  node {u}{' (gateway)' if u in obs.gateways else ''}:")
                show_slots(obs, p, u)
                print(f"    [1] route head raw logits z            : {st['route_raw'][u].numpy()}")
                print(f"    [2] Ko slot cost (row z-score, inf->0) : {np.nan_to_num(p['ko_slot'][u], posinf=0.0)}")
                print(f"    [3] after prior z - kappa*cost         : {st['route_prior'][u].numpy()}")
                print(f"    [4] after progress mask (-1e9)         : {st['route'][u].numpy()}")
                pr = torch.softmax(st["route"][u], -1).numpy()
                print(f"    route probs pi_p                       : {pr}   H_route={ent(st['route'][u]).item():.4f} nats")
                print(f"    [1] chan head raw logits               : {st['chan_raw'][u].numpy()}")
                print(f"    [2] Ko chan cost (row z-score)         : {p['ko_chan'][u]}")
                print(f"    [3] after prior (no channel mask)      : {st['chan'][u].numpy()}")
                pc = torch.softmax(st["chan"][u], -1).numpy()
                print(f"    chan probs pi_c                        : {pc}   H_chan={ent(st['chan'][u]).item():.4f} nats")
                j, c = int(pend["j"][u]), int(pend["c"][u])
                print(f"    sampled proposal: slot j={j}  channel idx c={c}"
                      f"{'  (head sample discarded: channel held, c=cur_chan)' if not p['is_chan_step'] else ''}")
                print(f"    executed: next_hop={action.next_hop[u]}  channels(radio0,radio1)={action.channels[u].tolist()}"
                      f"  (a=c*4+j={agent.last_action_idx[u]})")
                print(f"    stored logp_route=log pi_p(j)={pend['logp_route'][u]:.4f}  logp_chan=log pi_c(c)={pend['logp_chan'][u]:.4f}"
                      f"  -> PPO old logp = logp_route + is_chan*logp_chan = "
                      f"{pend['logp_route'][u] + float(p['is_chan_step']) * pend['logp_chan'][u]:.4f}")
                print(f"    value v_i = {pend['value'][u]:.3f}")
            # loss-side recomputation with the training function, same params -> ratio exactly 1
            n = len(p["x"])
            j_t, c_t = torch.as_tensor(pend["j"])[None].long(), torch.as_tensor(pend["c"])[None].long()
            is_chan = torch.as_tensor([float(p["is_chan_step"])])
            logp, H = agent._logp_entropy(st["out"], j_t, c_t, is_chan)
            old = pend["logp_route"] + float(p["is_chan_step"]) * pend["logp_chan"]
            print(f"  _logp_entropy (update path): max|logp - stored old| = {np.abs(logp[0].numpy() - old).max():.2e}"
                  f"  -> ratio = 1 at collection")
            print(f"  loss entropy term per node = H_route + is_chan*H_chan; nodes shown: "
                  f"{[round(float(H[0, u]), 4) for u in pick_nodes(obs, p)]}")
            print(f"  training executes the proposal unchanged: no hysteresis, no sequential decoding, no fallback"
                  f" (all gated by explore=False). Executed parent == slots[j] for every node with a valid slot: "
                  f"{bool(np.all((action.next_hop == p['slots'][np.arange(n), pend['j']]) | ~p['slot_mask'].any(1) | (action.next_hop == -1)))}")
        next_obs, r, info = env.step(action)
        rew = per_agent_reward(obs, action, info, r, env.w, lam=lam)
        agent.learn(obs, agent.last_action_idx, rew, next_obs, info["done"])
        obs = next_obs
        if info["done"]:
            break
    buf = agent.buf
    print(f"\n--- episode end: env.t={env.t} = episode_len {env.cfg.episode_len} (a time limit), done=True at the last"
          f" transition; buffer {len(buf)}/{agent.cfg.rollout} (no update fired) ---")
    rewards = np.stack([b["rewards"] for b in buf])
    values = np.stack([b["value"] for b in buf])
    dones = np.array([b["done"] for b in buf])
    last_v = np.zeros(values.shape[1])  # update(): dones[-1] -> zeros, no V(s_T) evaluated
    adv, ret = agent.gae(rewards, values, dones, last_v)
    u = pick_nodes(env._observe(None), agent.prepare(next_obs))[1]
    T = len(buf) - 1
    print(f"  node {u}: last step  r={rewards[T, u]:.3f}  v={values[T, u]:.3f}  return target={ret[T, u]:.3f}"
          f"  (== r: V(s_200) multiplied by (1-done)=0, i.e. the time limit is treated as termination)")
    print(f"  node {u}: step T-1   r={rewards[T-1, u]:.3f}  v={values[T-1, u]:.3f}  next v={values[T, u]:.3f}"
          f"  return target={ret[T-1, u]:.3f}  (bootstrapped: r + gamma*v_next + gamma*lambda*A_next)")
    print(f"  bootstrap mask b and trace mask c are both (1-done): identical; truncation is not distinguished.")
    print(f"  rollout {agent.cfg.rollout} % episode_len {env.cfg.episode_len} = {agent.cfg.rollout % env.cfg.episode_len}, "
          f"bc_steps {agent.cfg.bc_steps} % episode_len = {agent.cfg.bc_steps % env.cfg.episode_len}: every rollout ends on"
          f" done=True, so the _last_value bootstrap branch is never taken in these runs.")
    print(f"  dead/unalive nodes: reward 0 and loss weight 0 (alive mask); in S2 all {int(buf[0]['alive'].sum())} nodes alive.")
    print(f"  BC teacher targets this episode: {teacher_total} live-node targets, {teacher_dropped} "
          f"({100 * teacher_dropped / max(teacher_total, 1):.1f}%) fall on a progress-masked slot and get no route CE"
          f" (allowed=0); their channel CE still applies on channel steps.")


# ======================================================================================================
def eval_epoch(scen="S2", seed=0, verbose=True):
    agent = load()
    agent.cfg.switch_margin = MARGIN  # what mesh.eval.suite --agent-set switch_margin=2.0 does
    env = MeshEnv(cfg_of(scen), seed)
    obs = env.reset()
    s = dict(fires=0, fires_chan=0, decisions=0, chan_steps=0, maxH_chan=0.0, maxH_route=0.0,
             maxH_chan_m0=0.0, over2_m0=0, held_route=0, held_chan=0, seq_moved=0, route_ties=0, chan_ties=0,
             fb_ko_masked=0, fb_changed=0, max_sum_bound=0.0, fire_at=[])
    while True:
        p = agent.prepare(obs)
        st = stages(agent, p)
        rl, cl = st["route"].clone(), st["chan"].clone()
        cur = torch.as_tensor((p["slots"] == obs.next_hop[:, None]) & (obs.next_hop[:, None] >= 0))
        rlh = rl + MARGIN * cur
        clh = cl + MARGIN * nn.functional.one_hot(torch.as_tensor(p["cur_chan"]), N_DL_CH)
        j0, jh = rl.argmax(-1), rlh.argmax(-1)
        c_arg = clh.argmax(-1).numpy()
        c_seq = sequential_channels(obs, p["slots"], clh.numpy(), jh.numpy())
        Hr, Hc = ent(rlh), ent(clh)
        Hsum = Hr + (Hc if p["is_chan_step"] else 0.0)
        Hsum_m0 = ent(rl) + (ent(cl) if p["is_chan_step"] else 0.0)
        alive = p["alive"]
        unsure = (Hsum.numpy() > agent.cfg.fallback_entropy) & alive
        k = p["slot_mask"].sum(1)

        action = agent.act(obs, explore=False)  # the real evaluation call
        a_mine = ((torch.as_tensor(c_seq) if p["is_chan_step"] else torch.as_tensor(p["cur_chan"])) * K_SLOTS + jh).numpy()
        if unsure.any():
            ko = ko_action_idx(obs, p["slots"], agent.ko)
            s["fb_ko_masked"] += int((unsure & ~p["slot_mask"][np.arange(len(ko)), ko % K_SLOTS]).sum())
            s["fb_changed"] += int((unsure & (ko != a_mine)).sum())
            s["fire_at"] += [(obs.t, int(u), int(k[u]), round(float(Hsum[u]), 3)) for u in np.flatnonzero(unsure)]
            a_mine = np.where(unsure, ko, a_mine)
        assert np.array_equal(a_mine, agent.last_action_idx), "reconstruction differs from agent.act"
        assert abs(agent.fallback_rate - unsure[alive].mean()) < 1e-9

        s["decisions"] += int(alive.sum())
        s["fires"] += int(unsure.sum())
        s["chan_steps"] += int(p["is_chan_step"])
        if p["is_chan_step"]:
            s["fires_chan"] += int(unsure.sum())
            s["maxH_chan"] = max(s["maxH_chan"], float(Hsum[torch.as_tensor(alive)].max()))
            s["maxH_chan_m0"] = max(s["maxH_chan_m0"], float(Hsum_m0[torch.as_tensor(alive)].max()))
            s["over2_m0"] += int(((Hsum_m0.numpy() > 2.0) & alive).sum())
            s["held_chan"] += int(((clh.argmax(-1) != cl.argmax(-1)) & torch.as_tensor(alive)).sum())
            s["seq_moved"] += int((c_seq != c_arg)[alive].sum())
            srt = torch.sort(clh, -1, descending=True).values
            s["chan_ties"] += int(((srt[:, 0] - srt[:, 1]).abs() < 1e-6).numpy()[alive].sum())
            s["max_sum_bound"] = max(s["max_sum_bound"], float(np.log(np.maximum(k[alive], 1)).max() + math.log(N_DL_CH)))
        else:
            s["maxH_route"] = max(s["maxH_route"], float(Hsum[torch.as_tensor(alive)].max()))
        s["held_route"] += int(((jh != j0) & torch.as_tensor(alive)).sum())
        rs = torch.sort(rlh, -1, descending=True).values
        s["route_ties"] += int((((rs[:, 0] - rs[:, 1]).abs() < 1e-6).numpy() & alive & (k >= 2)).sum())

        if verbose and obs.t in (0, 10):
            print(f"\n--- eval t={obs.t} (is_chan_step={p['is_chan_step']}) ---")
            for u in pick_nodes(obs, p):
                print(f"  node {u}{' (gateway)' if u in obs.gateways else ''}:")
                show_slots(obs, p, u)
                print(f"    route logits (prior, mask)      : {rl[u].numpy()}")
                print(f"    + margin {MARGIN} on incumbent slot : {rlh[u].numpy()}  argmax -> j={int(jh[u])} (no margin: {int(j0[u])})")
                print(f"    chan logits (prior)             : {cl[u].numpy()}")
                print(f"    + margin on cur_chan {p['cur_chan'][u]}          : {clh[u].numpy()}  argmax c={c_arg[u]}  sequential c={c_seq[u]}")
                print(f"    fallback entropy H_route+{'H_chan' if p['is_chan_step'] else '0'} (biased) = {float(Hsum[u]):.4f}"
                      f"  (unbiased {float(Hsum_m0[u]):.4f}); threshold {agent.cfg.fallback_entropy} -> fire={bool(unsure[u])}")
                print(f"    executed: next_hop={action.next_hop[u]} channels={action.channels[u].tolist()}")
        obs, r, info = env.step(action)
        if info["done"]:
            break
    return s


def baseline_facts():
    print("\n" + "=" * 100)
    print("PART 3  TRAINING INTERACTIONS FROM runs/v10/ckpt/*.log and checkpoints")
    print("=" * 100)
    n_val = len(SUITE_A + SUITE_B) * 4 * 200  # validate(): 11 scenarios x 4 val seeds x 200 steps, no learning
    for f in sorted(glob.glob("runs/v10/ckpt/*_s0.log")):
        txt = open(f).read()
        last = re.findall(r"step (\d+)/(\d+)\s+episodes (\d+)", txt)[-1]
        vals = len(re.findall(r"val step", txt))
        print(f"  {f.split('/')[-1]:24s} env steps {last[0]}  episodes {last[2]}  agent-transitions {int(last[0]) * 20}"
              f"  validation calls {vals} x {n_val} = {vals * n_val} non-learning env steps")
    for name in ("yuan", "li", "q_neutral"):
        if name == "q_neutral":
            import pickle
            ag = pickle.load(open(f"runs/v10/ckpt/{name}_s0.pt", "rb"))
            print(f"  {name}: alpha {ag.alpha} gamma {ag.gamma} eps {ag.eps_start}->{ag.eps_end} over {ag.eps_decay_steps}"
                  f" steps, explore-steps {ag.steps}, table rows {len(ag.q)}")
        else:
            ck = torch.load(f"runs/v10/ckpt/{name}_s0.pt", map_location="cpu", weights_only=False)
            print(f"  {name}: hp {ck['hp']}  steps/updates {ck.get('steps', ck.get('updates'))}")


if __name__ == "__main__":
    training_transition()
    print("\n" + "=" * 100)
    print(f"PART 2  DETERMINISTIC EVALUATION EPOCH  (agent.act(explore=False), switch_margin={MARGIN}), S2 seed 0")
    print("=" * 100)
    s = eval_epoch()
    print(f"\n  S2 seed 0 summary: {s}")
    print(f"  ln4={math.log(4):.3f} ln6={math.log(6):.3f} ln24={math.log(24):.3f} ln2+ln6={math.log(12):.3f}")
    print("\n  Fallback count across suites A+B, eval seed 0 (same agent, margin 2):")
    tot = dict(fires=0, decisions=0, over2_m0=0, chan_steps=0, fb_changed=0)
    for sc in SUITE_A + SUITE_B:
        r = eval_epoch(sc, 0, verbose=False)
        for kk in tot:
            tot[kk] += r[kk]
        print(f"    {sc}: fires {r['fires']} / {r['decisions']} node-decisions  maxH(chan step, biased) {r['maxH_chan']:.3f}"
              f"  maxH(non-chan step) {r['maxH_route']:.3f}  maxH(chan step, margin 0) {r['maxH_chan_m0']:.3f}"
              f"  >2 at margin 0: {r['over2_m0']}  route held by margin {r['held_route']}  chan held {r['held_chan']}"
              f"  seq-decoder moved {r['seq_moved']}  fallback->Ko slot outside progress mask {r['fb_ko_masked']}"
              f"  fallback changed action {r['fb_changed']}  fires (t, node, #progress slots, H) {r['fire_at']}")
    print(f"    total: {tot}")
    baseline_facts()
