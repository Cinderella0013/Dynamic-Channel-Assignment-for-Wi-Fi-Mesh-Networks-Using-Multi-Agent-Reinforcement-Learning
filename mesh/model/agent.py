"""EGATAgent: EGAT-MAPPO learner with the Plan 3 learner interface (act / learn / save / load)."""
import numpy as np
import torch

from mesh.baselines.ko import KoLocalCost
from mesh.model.egat import EGATConfig, EGATNet
from mesh.model.ko_prior import ko_action_idx, ko_channel_costs, ko_slot_costs
from mesh.rl.agents_io import (K_SLOTS, N_DL_CH, _valid_slots, agent_features, alive_mask, decode,
                               neighbor_slots)
from mesh.sim.env import EDGE_FEATS, NODE_FEATS

_INPUT_KEYS = ("x", "e", "adj", "ko_slot", "ko_chan", "slot_mask")


class EGATAgent:
    def __init__(self, seed=0, device=None, **cfg_overrides):
        self.cfg = EGATConfig(**cfg_overrides)
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.net = EGATNet(self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.ko = KoLocalCost()
        self.steps = 0
        self.phase = "bc" if self.cfg.use_bc else "ppo"
        self.vko_ready = False  # V_Ko is only meaningful after it was fitted on Ko returns
        self.h = None
        self.last_action_idx = None
        self.fallback_rate = 0.0
        self._pending = None
        self.bc_buf, self.buf = [], []

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

    def _tensors(self, items, h_prev):
        t = {k: torch.as_tensor(np.stack([it[k] for it in items]), device=self.device) for k in _INPUT_KEYS}
        return self.net(t["x"], t["e"], t["adj"], h_prev, t["ko_slot"], t["ko_chan"], t["slot_mask"])

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
            j, c = route.probs.argmax(-1), chan.probs.argmax(-1)
        if not p["is_chan_step"]:
            c = torch.as_tensor(p["cur_chan"])
        a = (c * K_SLOTS + j).numpy()

        if not explore and self.cfg.use_fallback:
            ent = (route.entropy() + chan.entropy()).numpy()
            unsure = ent > self.cfg.fallback_entropy
            if self.vko_ready:
                v, vko = out["value"][0].cpu().numpy(), out["value_ko"][0].cpu().numpy()
                unsure |= v < vko - self.cfg.fallback_margin
            unsure &= p["alive"]
            if unsure.any():
                a = np.where(unsure, ko_action_idx(obs, p["slots"], self.ko), a)
            self.fallback_rate = float(unsure[p["alive"]].mean()) if p["alive"].any() else 0.0

        self._pending = dict(p=p, h_prev=h_prev.cpu(), j=j.numpy(), c=c.numpy(),
                             logp_route=route.log_prob(j).numpy(), logp_chan=chan.log_prob(c).numpy(),
                             value=out["value"][0].cpu().numpy(), is_chan_step=p["is_chan_step"])
        self.last_action_idx = a
        return decode(obs, p["slots"], a)
