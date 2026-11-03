"""EGATAgent: EGAT-MAPPO learner with the Plan 3 learner interface (act / learn / save / load)."""
import copy

import numpy as np
import torch

from mesh.baselines.ko import KoLocalCost
from mesh.model.egat import EGATConfig, EGATNet
from mesh.model.ko_prior import ko_action_idx, ko_channel_costs, ko_slot_costs
from mesh.rl.agents_io import (K_SLOTS, N_DL_CH, _valid_slots, agent_features, alive_mask, decode,
                               neighbor_slots, sequential_channels)
from mesh.sim.env import EDGE_FEATS, NODE_FEATS

_INPUT_KEYS = ("x", "e", "adj", "ko_slot", "ko_chan", "slot_mask")


class EGATAgent:
    config_cls = EGATConfig

    def __init__(self, seed=0, device=None, **cfg_overrides):
        self.cfg = self.config_cls(**cfg_overrides)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.net = EGATNet(self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.ko = KoLocalCost()
        self.steps = 0
        self.phase = "bc" if self.cfg.use_bc else "ppo"
        self.h = None
        self.last_action_idx = None
        self.fallback_rate = 0.0
        self._pending = None
        self.bc_buf, self.buf = [], []
        self.bc_net = None
        self.total_steps = 100_000

    # ---------- inputs ----------
    def progress_mask(self, obs, slots):
        valid = _valid_slots(obs, slots)
        if self.cfg.progress_mask:
            hop = obs.node_feat[:, NODE_FEATS.index("hop_to_gw")]
            valid &= hop[np.maximum(slots, 0)] < hop[:, None]  # strictly closer: routes are loop-free
        none = ~valid.any(axis=1)
        valid[none, 0] = True  # nothing usable (gateway, cut off, dead): slot 0, decoded to -1
        return valid

    def prepare(self, obs):
        slots = neighbor_slots(obs)
        o, m = agent_features(obs, slots)
        n = len(o)
        alive = alive_mask(obs)
        e = np.zeros((n, n, len(EDGE_FEATS)), dtype=np.float32)
        e[obs.edge_index[0], obs.edge_index[1]] = obs.edge_attr
        adj = np.zeros((n, n), dtype=bool)
        adj[obs.edge_index[0], obs.edge_index[1]] = True
        adj &= alive[:, None] & alive[None, :]
        adj |= np.eye(n, dtype=bool)
        return dict(x=np.concatenate([o, m], 1).astype(np.float32), e=e, adj=adj,
                    ko_slot=ko_slot_costs(obs, slots).astype(np.float32),
                    ko_chan=ko_channel_costs(obs).astype(np.float32),
                    slot_mask=self.progress_mask(obs, slots), slots=slots, alive=alive,
                    cur_chan=np.clip(obs.channels[:, 1] - 1, 0, N_DL_CH - 1),
                    is_chan_step=bool((not self.cfg.two_timescale) or obs.t % self.cfg.t_c == 0))

    def _tensors(self, items, h_prev, net=None):
        t = {k: torch.as_tensor(np.stack([it[k] for it in items]), device=self.device) for k in _INPUT_KEYS}
        return (net or self.net)(t["x"], t["e"], t["adj"], h_prev, t["ko_slot"], t["ko_chan"], t["slot_mask"])

    def _bc_tensors(self, items, h_prev):
        return self._tensors(items, h_prev, net=self.bc_net)

    def set_budget(self, steps):
        self.total_steps = steps

    # ---------- acting ----------
    def act(self, obs, explore=False):
        n = len(obs.node_feat)
        if obs.t == 0 or self.h is None or self.h.shape[0] != n:
            self.h = torch.zeros(n, self.cfg.hidden, device=self.device)
        p = self.prepare(obs)
        h_prev = self.h
        self.net.eval()
        with torch.no_grad():
            out = self._tensors([p], h_prev[None])
        self.h = out["h"][0]
        route = torch.distributions.Categorical(logits=out["route_logits"][0].cpu())
        chan = torch.distributions.Categorical(logits=out["chan_logits"][0].cpu())

        if self.phase == "bc" and explore:  # behave as Ko and record it for behaviour cloning
            a = ko_action_idx(obs, p["slots"], self.ko)
            self._pending = dict(p=p, h_prev=h_prev.cpu(), a=a)
            self.last_action_idx = a
            return decode(obs, p["slots"], a)

        if explore:
            j = torch.multinomial(route.probs, 1, generator=self.gen)[:, 0]
            c = torch.multinomial(chan.probs, 1, generator=self.gen)[:, 0]
        else:
            j = route.probs.argmax(-1)
            if self.cfg.seq_decode:
                c = torch.as_tensor(sequential_channels(obs, p["slots"], chan.logits.numpy(), j.numpy()))
            else:
                c = chan.probs.argmax(-1)
        if not p["is_chan_step"]:
            c = torch.as_tensor(p["cur_chan"])
        a = (c * K_SLOTS + j).numpy()

        self.fallback_rate = 0.0
        if not explore and self.cfg.use_fallback:
            # entropy-only: uses this node's own heads, so execution stays decentralised
            ent = route.entropy() + (chan.entropy() if p["is_chan_step"] else 0.0)
            unsure = ent.numpy() > self.cfg.fallback_entropy
            unsure &= p["alive"]
            if unsure.any():
                a = np.where(unsure, ko_action_idx(obs, p["slots"], self.ko), a)
            self.fallback_rate = float(unsure[p["alive"]].mean()) if p["alive"].any() else 0.0

        self._pending = dict(p=p, h_prev=h_prev.cpu(), j=j.numpy(), c=c.numpy(),
                             logp_route=route.log_prob(j).numpy(), logp_chan=chan.log_prob(c).numpy(),
                             value=out["value"][0].cpu().numpy(), is_chan_step=p["is_chan_step"])
        self.last_action_idx = a
        return decode(obs, p["slots"], a)

    # ---------- learning ----------
    def reward_lam(self, step, total):
        f = min(step / max(total, 1), 1.0)
        return self.cfg.lam_start + f * (self.cfg.lam_end - self.cfg.lam_start)

    def learn(self, obs, action_idx, rewards, next_obs, done):
        self.steps += 1
        info = {}
        if self.phase == "bc":
            self.bc_buf.append(dict(self._pending, rewards=np.asarray(rewards, np.float32), done=float(done)))
            if len(self.bc_buf) >= self.cfg.bc_steps:
                info = self.bc_fit()
                self.phase = "ppo"
            return info
        self.buf.append(dict(self._pending, rewards=np.asarray(rewards, np.float32), done=float(done),
                             alive=self._pending["p"]["alive"]))
        if len(self.buf) >= self.cfg.rollout:
            info = self.update(next_obs)
            self.buf = []
        return info

    def gae(self, rewards, values, dones, last_value):
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        adv = np.zeros_like(values, dtype=np.float64)
        running = np.zeros(values.shape[1])
        for t in reversed(range(len(rewards))):
            nonterm = 1.0 - dones[t]
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + g * next_v * nonterm - values[t]
            running = delta + g * lam * nonterm * running
            adv[t] = running
        return adv, adv + values

    def nstep_adv(self, rewards, values, dones, last_value, n_steps):
        """Semi-MDP credit: A_t = sum_{k<m} g^k r_{t+k} + g^m V_{t+m} - V_t, m <= n_steps, cut at done."""
        g, T = self.cfg.gamma, len(rewards)
        adv = np.zeros_like(values, dtype=np.float64)
        for t in range(T):
            ret, k, ended = np.zeros(values.shape[1]), 0, False
            while k < n_steps and t + k < T:
                ret += g ** k * rewards[t + k]
                k += 1
                if dones[t + k - 1]:
                    ended = True
                    break
            if not ended:
                ret += g ** k * (values[t + k] if t + k < T else last_value)
            adv[t] = ret - values[t]
        return adv

    def _last_value(self, last_obs):
        p = self.prepare(last_obs)
        self.net.eval()
        with torch.no_grad():
            return self._tensors([p], self.h[None])["value"][0].cpu().numpy()

    def _logp_entropy(self, out, j, c, is_chan):
        lr = torch.log_softmax(out["route_logits"], -1)
        lc = torch.log_softmax(out["chan_logits"], -1)
        logp = lr.gather(-1, j[..., None])[..., 0] + is_chan[:, None] * lc.gather(-1, c[..., None])[..., 0]
        ent = -(lr.exp() * lr).sum(-1) - is_chan[:, None] * (lc.exp() * lc).sum(-1)
        return logp, ent

    @staticmethod
    def _kl(logits_p, logits_q):
        lp, lq = torch.log_softmax(logits_p, -1), torch.log_softmax(logits_q, -1)
        return (lp.exp() * (lp - lq)).sum(-1)

    def update(self, last_obs):
        cfg, buf = self.cfg, self.buf
        rewards = np.stack([b["rewards"] for b in buf])
        values = np.stack([b["value"] for b in buf])
        dones = np.array([b["done"] for b in buf])
        alive = np.stack([b["alive"] for b in buf])
        last_v = np.zeros(values.shape[1]) if dones[-1] else self._last_value(last_obs)
        adv, ret = self.gae(rewards, values, dones, last_v)
        adv = (adv - adv[alive].mean()) / (adv[alive].std() + 1e-8)
        adv_ch = adv
        if cfg.semi_mdp:  # channel decisions are credited with the reward over the next t_c steps
            adv_ch = self.nstep_adv(rewards, values, dones, last_v, cfg.t_c)
            adv_ch = (adv_ch - adv_ch[alive].mean()) / (adv_ch[alive].std() + 1e-8)

        dev = self.device
        T = len(buf)
        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0)
        n_mb = 0
        self.net.train()
        for _ in range(cfg.epochs):
            order = np.random.default_rng(self.seed + self.steps + n_mb).permutation(T)
            for start in range(0, T, cfg.minibatch):
                idx = order[start:start + cfg.minibatch]
                items = [buf[i]["p"] for i in idx]
                h_prev = torch.stack([buf[i]["h_prev"] for i in idx]).to(dev)
                out = self._tensors(items, h_prev)
                j = torch.as_tensor(np.stack([buf[i]["j"] for i in idx]), device=dev).long()
                c = torch.as_tensor(np.stack([buf[i]["c"] for i in idx]), device=dev).long()
                is_chan = torch.as_tensor([float(buf[i]["is_chan_step"]) for i in idx], device=dev)
                old = torch.as_tensor(np.stack([buf[i]["logp_route"] + float(buf[i]["is_chan_step"]) * buf[i]["logp_chan"]
                                                for i in idx]), device=dev, dtype=torch.float32)
                a_t = torch.as_tensor(adv[idx], device=dev, dtype=torch.float32)
                r_t = torch.as_tensor(ret[idx], device=dev, dtype=torch.float32)
                m = torch.as_tensor(alive[idx], device=dev, dtype=torch.float32)
                logp, ent = self._logp_entropy(out, j, c, is_chan)
                if cfg.semi_mdp:  # separate clipped surrogates: route on GAE, channel on n-step advantage
                    lr_new = torch.log_softmax(out["route_logits"], -1).gather(-1, j[..., None])[..., 0]
                    lc_new = torch.log_softmax(out["chan_logits"], -1).gather(-1, c[..., None])[..., 0]
                    lr_old = torch.as_tensor(np.stack([buf[i]["logp_route"] for i in idx]), device=dev, dtype=torch.float32)
                    lc_old = torch.as_tensor(np.stack([buf[i]["logp_chan"] for i in idx]), device=dev, dtype=torch.float32)
                    ac_t = torch.as_tensor(adv_ch[idx], device=dev, dtype=torch.float32)
                    rr, rc = torch.exp(lr_new - lr_old), torch.exp(lc_new - lc_old)
                    pg = -torch.min(rr * a_t, rr.clamp(1 - cfg.clip, 1 + cfg.clip) * a_t)                         - is_chan[:, None] * torch.min(rc * ac_t, rc.clamp(1 - cfg.clip, 1 + cfg.clip) * ac_t)
                else:
                    ratio = torch.exp(logp - old)
                    pg = -torch.min(ratio * a_t, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_t)
                vl = (out["value"] - r_t) ** 2
                per = pg + cfg.value_coef * vl - cfg.entropy * ent
                if self.bc_net is not None:  # v1: stay near the BC policy (AlphaStar-style anchor)
                    with torch.no_grad():
                        ref = self._bc_tensors(items, h_prev)
                    kl = self._kl(out["route_logits"], ref["route_logits"])                         + is_chan[:, None] * self._kl(out["chan_logits"], ref["chan_logits"])
                    bc = cfg.bc_steps if cfg.use_bc else 0  # anneal from the end of BC
                    f = min(max(self.steps - bc, 0) / max(self.total_steps - bc, 1), 1.0)
                    per = per + (cfg.kl_start + f * (cfg.kl_end - cfg.kl_start)) * kl
                loss = (per * m).sum() / m.sum().clamp(min=1)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad)
                self.opt.step()
                k = m.sum().clamp(min=1)
                stats["policy_loss"] += ((pg * m).sum() / k).item()
                stats["value_loss"] += ((vl * m).sum() / k).item()
                stats["entropy"] += ((ent * m).sum() / k).item()
                stats["approx_kl"] += (((old - logp) * m).sum() / k).item()
                n_mb += 1
        return {key: val / max(n_mb, 1) for key, val in stats.items()}

    # ---------- behaviour cloning from Ko ----------
    def bc_fit(self):
        cfg, buf, dev = self.cfg, self.bc_buf, self.device
        T = len(buf)
        rewards = np.stack([b["rewards"] for b in buf])
        dones = np.array([b["done"] for b in buf])
        ret = np.zeros_like(rewards, dtype=np.float64)
        running = np.zeros(rewards.shape[1])
        for t in reversed(range(T)):  # Monte-Carlo returns of Ko's own trajectories
            running = rewards[t] + cfg.gamma * (1.0 - dones[t]) * running
            ret[t] = running
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.bc_lr)
        self.net.train()
        stats = {}
        for epoch in range(cfg.bc_epochs):
            order = np.random.default_rng(self.seed + epoch).permutation(T)
            hits = total = 0
            tot = dict(bc_loss=0.0, value_loss=0.0)
            for start in range(0, T, cfg.minibatch):
                idx = order[start:start + cfg.minibatch]
                items = [buf[i]["p"] for i in idx]
                out = self._tensors(items, torch.stack([buf[i]["h_prev"] for i in idx]).to(dev))
                a = np.stack([buf[i]["a"] for i in idx])
                j = torch.as_tensor(a % K_SLOTS, device=dev).long()
                c = torch.as_tensor(a // K_SLOTS, device=dev).long()
                is_chan = torch.as_tensor([float(it["is_chan_step"]) for it in items], device=dev)[:, None]
                alive = torch.as_tensor(np.stack([it["alive"] for it in items]), device=dev).float()
                # Ko may pick an equal-hop parent the progress mask forbids: no route target there
                allowed = torch.as_tensor(np.stack([it["slot_mask"] for it in items]), device=dev)
                allowed = allowed.gather(-1, j[..., None])[..., 0].float()
                ce_r = torch.nn.functional.cross_entropy(out["route_logits"].transpose(1, 2), j, reduction="none")
                ce_c = torch.nn.functional.cross_entropy(out["chan_logits"].transpose(1, 2), c, reduction="none")
                g = torch.as_tensor(ret[idx], device=dev, dtype=torch.float32)
                v_l = (out["value"] - g) ** 2  # critic warm start on Ko's returns
                per = allowed * ce_r + is_chan * ce_c + 0.5 * v_l
                loss = (per * alive).sum() / alive.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    ok = out["route_logits"].argmax(-1) == j
                    ok &= (is_chan == 0) | (out["chan_logits"].argmax(-1) == c)
                    hits += (ok.float() * alive).sum().item()
                    total += alive.sum().item()
                tot["bc_loss"] += ((allowed * ce_r + is_chan * ce_c) * alive).sum().item() / max(alive.sum().item(), 1)
                tot["value_loss"] += ((v_l * alive).sum() / alive.sum().clamp(min=1)).item()
            n_mb = int(np.ceil(T / cfg.minibatch))
            stats = dict(agreement=hits / max(total, 1), bc_loss=tot["bc_loss"] / n_mb, value_loss=tot["value_loss"] / n_mb)
            if stats["agreement"] >= 0.9:
                break
        self.bc_buf = []
        if cfg.use_kl_anchor:
            self.bc_net = copy.deepcopy(self.net).eval()
            self.bc_net.requires_grad_(False)
        if cfg.lr_after_bc is not None:
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr_after_bc
        return stats

    def agreement(self, obs):
        n = len(obs.node_feat)
        p = self.prepare(obs)
        self.net.eval()
        with torch.no_grad():
            out = self._tensors([p], torch.zeros(1, n, self.cfg.hidden, device=self.device))
        j = out["route_logits"][0].argmax(-1).cpu().numpy()
        ko = ko_action_idx(obs, p["slots"], KoLocalCost())
        # route agreement only: channel labels are interchangeable, so exact-label match with Ko
        # is not a meaningful imitation target (see agents_io.sequential_channels)
        ok = j == ko % K_SLOTS
        return float(ok[p["alive"]].mean())

    # ---------- checkpoints ----------
    def save(self, path):
        torch.save(dict(cfg=vars(self.cfg), seed=self.seed, net=self.net.state_dict(), opt=self.opt.state_dict(),
                        phase=self.phase, steps=self.steps,
                        fallback_rate=self.fallback_rate), path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu")
        ag = cls(seed=ck["seed"], device=device, **ck["cfg"])
        ag.net.load_state_dict(ck["net"])
        ag.opt.load_state_dict(ck["opt"])
        ag.phase, ag.steps = ck["phase"], ck["steps"]
        ag.fallback_rate = ck["fallback_rate"]
        return ag  # unfinished BC data is not saved: a BC-phase checkpoint resumes with an empty buffer


class EGATv1Agent(EGATAgent):
    """EGAT v1 (spec addendum 3): KL anchor to BC, fixed lambda = 0.1, semi-MDP channel credit, lr 1e-4 after BC."""

    def __init__(self, seed=0, device=None, **cfg_overrides):
        defaults = dict(use_kl_anchor=True, semi_mdp=True, lam_start=0.1, lam_end=0.1, lr_after_bc=1e-4)
        super().__init__(seed=seed, device=device, **{**defaults, **cfg_overrides})
