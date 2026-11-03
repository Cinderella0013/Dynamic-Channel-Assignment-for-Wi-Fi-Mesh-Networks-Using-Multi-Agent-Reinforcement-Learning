"""EGAT v2 (spec addendum 4, the user's design): GAT-Q scores -> RAES ordered channel reservation
-> migration refinement -> execute with queue preservation -> ACK feeds the next round."""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from mesh.model.agent import EGATAgent
from mesh.model.egat import EGATConfig
from mesh.model.ko_prior import ko_action_idx
from mesh.rl.agents_io import K_SLOTS, N_DL_CH, alive_mask, decode
from mesh.sim.env import NODE_FEATS, edge_matrix


def _z(x):
    """z-score over the last axis; a constant row becomes 0 (no preference)."""
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=-1, keepdims=True)
    return np.divide(x - x.mean(axis=-1, keepdims=True), sd, out=np.zeros_like(x), where=sd > 0)


def _z_global(m, valid):
    """z-score over all valid entries of a matrix: keeps the size of a cost across nodes, unlike a
    per-row z-score, which would charge a node with 0.01 Mb backlog the same as one at Q_MAX."""
    vals = m[valid]
    sd = vals.std() if vals.size else 0.0
    return np.where(valid, (m - vals.mean()) / sd, 0.0) if sd > 0 else np.zeros_like(m, dtype=float)


def _z_rows(x, dom):
    """Per-row z-score over the entries in dom; a row with < 2 entries in dom becomes 0."""
    x = np.where(dom, x, np.nan)
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(x, axis=1, keepdims=True), np.nanstd(x, axis=1, keepdims=True)
    z = np.where((sd > 0) & dom, (x - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    return np.nan_to_num(z)


def _route_score(obs, slots, q_rt, dom, g, downtime, cur_slot):
    """z(Q_rt) over the allowed slots minus g * (globally z-scored) cost of changing parent; -inf outside dom."""
    backlog = obs.node_feat[:, NODE_FEATS.index("backlog")]
    move = (np.arange(slots.shape[1])[None, :] != cur_slot[:, None]) * downtime * backlog[:, None]
    return np.where(dom, _z_rows(q_rt, dom) - g * _z_global(move, dom), -np.inf)


def _depth(next_hop):
    n = len(next_hop)
    depth = np.zeros(n, dtype=int)
    for u in range(n):
        v, d = u, 0
        while next_hop[v] >= 0 and d <= n:
            v, d = next_hop[v], d + 1
        depth[u] = d
    return depth


def _current_slot(obs, slots, fallback):
    hit = (slots == obs.next_hop[:, None]) & (slots >= 0)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), fallback)


def raes(obs, slots, q_ch, q_rt, route_mask, a=1.0, b=1.0, g=1.0, order="backlog", downtime=1):
    """Ordered channel reservation: each node, in turn, reserves
    argmax_c z(Q) - a z(interference from reserved neighbours) - b z(co-channel load) - g z(move cost)."""
    n = len(q_ch)
    alive = alive_mask(obs)
    nf = obs.node_feat
    backlog = nf[:, NODE_FEATS.index("backlog")]
    load = nf[:, NODE_FEATS.index("offered_load")]
    hop = nf[:, NODE_FEATS.index("hop_to_gw")]
    conflict = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
    cur = np.clip(obs.channels[:, 1] - 1, 0, N_DL_CH - 1)
    if order == "backlog":  # congested nodes reserve first
        seq = sorted(range(n), key=lambda u: (-backlog[u], hop[u], u))
    else:  # "parents": tree order of the current routes
        depth = _depth(obs.next_hop)
        seq = sorted(range(n), key=lambda u: (depth[u], u))
    move = (np.arange(N_DL_CH)[None, :] != cur[:, None]) * downtime * backlog[:, None]
    z_move = _z_global(move, np.repeat(alive[:, None], N_DL_CH, axis=1))
    c = np.full(n, -1, dtype=int)
    score = np.zeros((n, N_DL_CH))
    for u in seq:
        nb = np.flatnonzero(conflict[u])
        interf = np.zeros(N_DL_CH)
        for v in nb[c[nb] >= 0]:
            interf[c[v]] += 1.0 + backlog[v]
        cochan = np.zeros(N_DL_CH)
        np.add.at(cochan, cur[nb], load[nb])
        score[u] = _z(q_ch[u]) - a * _z(interf) - b * _z(cochan) - g * z_move[u]
        c[u] = int(np.argmax(score[u])) if alive[u] else cur[u]

    cur_slot = _current_slot(obs, slots, fallback=np.full(n, -1))
    j = _route_score(obs, slots, q_rt, route_mask, g, downtime, cur_slot).argmax(axis=1)
    return c, j, score


def refine(obs, slots, c, j, ch_score, q_rt, cooldown_left, margin=0.1, k_frac=0.1, route_mask=None, g=1.0,
           downtime=1):
    """Migration refinement: keep a proposed move only if its gain beats the margin, the node is not
    cooling down, no conflicting neighbour moves this round, and at most ceil(k_frac * n_alive) move."""
    n = len(c)
    alive = alive_mask(obs)
    conflict = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
    c_cur = np.clip(obs.channels[:, 1] - 1, 0, N_DL_CH - 1)
    j_cur = _current_slot(obs, slots, fallback=j)
    idx = np.arange(n)
    # route gain on the same scale RAES uses (allowed slots plus the current one), move cost included
    dom = (slots >= 0) if route_mask is None else route_mask.copy()
    dom[idx, j_cur] = True
    rs = _route_score(obs, slots, q_rt, dom, g, downtime, j_cur)
    gain = (ch_score[idx, c] - ch_score[idx, c_cur]) + (rs[idx, j] - rs[idx, j_cur])
    proposes = (c != c_cur) | (j != j_cur)
    cand = np.flatnonzero(proposes & alive & (gain > margin) & (np.asarray(cooldown_left) == 0))
    k_max = math.ceil(k_frac * alive.sum())
    moved = np.zeros(n, dtype=bool)
    for u in cand[np.argsort(-gain[cand], kind="stable")]:
        if moved.sum() >= k_max:
            break
        if not (conflict[u] & moved).any():
            moved[u] = True
    return np.where(moved, c, c_cur), np.where(moved, j, j_cur), moved


@dataclass
class V2Config(EGATConfig):
    # Double DQN on executed actions (the planner rewrites the argmax, so learning is off-policy)
    use_ko_prior: bool = False
    use_fallback: bool = False
    lam_start: float = 0.1
    lam_end: float = 0.1
    replay_size: int = 50_000
    batch: int = 32
    tau: float = 0.01
    eps_start: float = 0.3
    eps_end: float = 0.02
    eps_decay: int = 50_000
    dqfd_margin: float = 0.8
    # planner
    raes: bool = True
    migration_control: bool = True
    queue_preserve: bool = True
    raes_order: str = "backlog"
    a: float = 1.0
    b: float = 1.0
    g: float = 1.0
    margin: float = 0.1
    k_frac: float = 0.1
    cooldown: int = 3


class EGATv2Agent(EGATAgent):
    """GAT-Q + RAES + migration control + queue preservation. Q(u, c*K+j) = Q_ch(u,c) + Q_rt(u,j)."""
    config_cls = V2Config

    def __init__(self, seed=0, device=None, **cfg_overrides):
        super().__init__(seed=seed, device=device, **cfg_overrides)
        self.target = copy.deepcopy(self.net)
        self.target.requires_grad_(False)
        self.replay, self.replay_pos = [], 0
        self.cool = None
        self.rng = np.random.default_rng(seed)
        self._cache = (None, None)

    def _q(self, items, h, net=None):
        # Q heads see every slot; the planner and the TD target apply masks, so the executed
        # (possibly current, non-progress) parent always has a finite Q value
        items = [dict(it, slot_mask=np.ones_like(it["slot_mask"])) for it in items]
        return self._tensors(items, h, net=net)

    def _prep(self, obs):
        key, val = self._cache
        return val if key is obs else self.prepare(obs)

    def reward_lam(self, step, total):
        return self.cfg.lam_start

    def _eps(self):
        start = self.cfg.bc_steps if self.cfg.use_bc else 0
        f = min(max(self.steps - start, 0) / self.cfg.eps_decay, 1.0)
        return self.cfg.eps_start + f * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, obs, explore=False):
        cfg, n = self.cfg, len(obs.node_feat)
        if obs.t == 0 or self.h is None or self.h.shape[0] != n:
            self.h = torch.zeros(n, cfg.hidden, device=self.device)
            self.cool = np.zeros(n, dtype=int)
        p = self._prep(obs)
        h_prev = self.h
        self.net.eval()
        with torch.no_grad():
            out = self._q([p], h_prev[None])
        self.h = out["h"][0]
        q_ch = out["chan_logits"][0].cpu().numpy()
        q_rt = out["route_logits"][0].cpu().numpy()
        mask = p["slot_mask"]

        if self.phase == "bc" and explore:  # behave as Ko (DQfD demonstrations)
            a = ko_action_idx(obs, p["slots"], self.ko)
        else:
            if explore:
                noisy = self.rng.random(n) < self._eps()
                q_ch[noisy] = self.rng.normal(size=(noisy.sum(), N_DL_CH))
                q_rt[noisy] = self.rng.normal(size=(noisy.sum(), K_SLOTS))
            if cfg.raes:
                c, j, score = raes(obs, p["slots"], q_ch, q_rt, mask, cfg.a, cfg.b, cfg.g, cfg.raes_order)
            else:
                c, j, score = q_ch.argmax(1), np.where(mask, q_rt, -np.inf).argmax(1), _z(q_ch)
            self.cool = np.maximum(self.cool - 1, 0)
            if cfg.migration_control:
                c_prop, j_prop = c.copy(), j.copy()
                c, j, moved = refine(obs, p["slots"], c, j, score, q_rt, self.cool, cfg.margin, cfg.k_frac,
                                     route_mask=mask, g=cfg.g)
                if explore:  # exploratory proposals are executed, otherwise refinement filters most of them
                    c[noisy], j[noisy] = c_prop[noisy], j_prop[noisy]
                    moved |= noisy
                # +1: the counter is decremented before refine, so a move blocks the next `cooldown` rounds
                self.cool[moved] = cfg.cooldown + 1
            a = c * K_SLOTS + j
        action = decode(obs, p["slots"], a)
        action.queue_preserve = cfg.queue_preserve
        self._pending = dict(p=p, h_prev=h_prev.cpu())
        self.last_action_idx = a
        return action

    def learn(self, obs, action_idx, rewards, next_obs, done):
        self.steps += 1
        p_next = self.prepare(next_obs)
        self._cache = (next_obs, p_next)
        item = dict(p=self._pending["p"], h_prev=self._pending["h_prev"], a=np.asarray(action_idx),
                    r=np.asarray(rewards, np.float32), p_next=p_next, h_next=self.h.cpu(), done=float(done),
                    alive=self._pending["p"]["alive"])
        if self.phase == "bc":
            self.bc_buf.append(item)
            if len(self.bc_buf) >= self.cfg.bc_steps:
                info = self.bc_fit()
                self.phase = "rl"
                return info
            return {}
        self._push(item)
        return self.update() if len(self.replay) >= self.cfg.batch else {}

    def _push(self, item):
        if len(self.replay) < self.cfg.replay_size:
            self.replay.append(item)
        else:
            self.replay[self.replay_pos] = item
        self.replay_pos = (self.replay_pos + 1) % self.cfg.replay_size

    def _td(self, batch):
        cfg, dev = self.cfg, self.device

        def st(key, dtype=torch.float32):
            return torch.as_tensor(np.stack([b[key] for b in batch]), device=dev, dtype=dtype)

        a, r, done, alive = st("a", torch.long), st("r"), st("done"), st("alive")
        mask_next = torch.as_tensor(np.stack([b["p_next"]["slot_mask"] for b in batch]), device=dev)
        h = torch.stack([b["h_prev"] for b in batch]).to(dev)
        h_next = torch.stack([b["h_next"] for b in batch]).to(dev)
        out = self._q([b["p"] for b in batch], h)
        j, c = a % K_SLOTS, a // K_SLOTS
        q_sa = out["chan_logits"].gather(-1, c[..., None])[..., 0] + out["route_logits"].gather(-1, j[..., None])[..., 0]
        with torch.no_grad():  # Double DQN per head: the online net picks, the target net scores
            nxt = [b["p_next"] for b in batch]
            on, tg = self._q(nxt, h_next), self._q(nxt, h_next, net=self.target)
            c2 = on["chan_logits"].argmax(-1)
            j2 = on["route_logits"].masked_fill(~mask_next, -1e9).argmax(-1)
            q_next = tg["chan_logits"].gather(-1, c2[..., None])[..., 0] + tg["route_logits"].gather(-1, j2[..., None])[..., 0]
            y = r + cfg.gamma * (1 - done)[:, None] * q_next
        td = torch.nn.functional.smooth_l1_loss(q_sa, y, reduction="none")
        return out, td, a, alive

    def _step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.target.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def update(self, last_obs=None):
        self.net.train()
        idx = self.rng.choice(len(self.replay), self.cfg.batch, replace=False)
        _, td, _, alive = self._td([self.replay[i] for i in idx])
        loss = (td * alive).sum() / alive.sum().clamp(min=1)
        self._step(loss)
        return {"loss": loss.item()}

    def bc_fit(self):
        """DQfD warm start: large-margin loss toward Ko's actions per head plus TD; demos then go to replay."""
        cfg, buf, dev = self.cfg, self.bc_buf, self.device
        self.target.load_state_dict(self.net.state_dict())
        self.net.train()
        stats = {}
        for _ in range(cfg.bc_epochs):
            order = self.rng.permutation(len(buf))
            tot, nb = 0.0, 0
            for start in range(0, len(buf), cfg.batch):
                batch = [buf[i] for i in order[start:start + cfg.batch]]
                out, td, a, alive = self._td(batch)
                j, c = a % K_SLOTS, a // K_SLOTS
                mask = torch.as_tensor(np.stack([b["p"]["slot_mask"] for b in batch]), device=dev)
                mask = mask.scatter(-1, j[..., None], True)
                one_c = torch.nn.functional.one_hot(c, N_DL_CH).float()
                one_j = torch.nn.functional.one_hot(j, K_SLOTS).float()
                lc = out["chan_logits"] + cfg.dqfd_margin * (1 - one_c)
                lr = (out["route_logits"] + cfg.dqfd_margin * (1 - one_j)).masked_fill(~mask, -1e9)
                margin = (lc.max(-1).values - (out["chan_logits"] * one_c).sum(-1)) \
                    + (lr.max(-1).values - (out["route_logits"] * one_j).sum(-1))
                loss = ((margin + td) * alive).sum() / alive.sum().clamp(min=1)
                self._step(loss)
                tot, nb = tot + loss.item(), nb + 1
            stats = {"bc_loss": tot / max(nb, 1)}
        self.target.load_state_dict(self.net.state_dict())
        for item in buf:
            self._push(item)
        self.bc_buf = []
        return stats

    def save(self, path):
        torch.save(dict(cfg=vars(self.cfg), seed=self.seed, net=self.net.state_dict(),
                        target=self.target.state_dict(), opt=self.opt.state_dict(), phase=self.phase,
                        steps=self.steps), path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu")
        ag = cls(seed=ck["seed"], device=device, **ck["cfg"])
        ag.net.load_state_dict(ck["net"])
        ag.target.load_state_dict(ck["target"])
        ag.opt.load_state_dict(ck["opt"])
        ag.phase, ag.steps = ck["phase"], ck["steps"]
        return ag
