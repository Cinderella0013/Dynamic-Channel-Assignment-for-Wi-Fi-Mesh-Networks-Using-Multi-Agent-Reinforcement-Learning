"""Yuan et al. 2022 GQN adaptation: MLP encoder + multi-head dot-product attention
over self and 3 neighbours (Eq. 11-12) + Q head, DQN loss + temporal relation KL (Eq. 13).
See docs/baselines-adaptation.md for departures from the paper.
"""
import numpy as np
import torch
from torch import nn

from mesh.rl.agents_io import (F_M, F_O, N_ACTIONS, action_mask, agent_features, alive_mask,
                               decode, neighbor_slots)

N_NBR = 3  # Yuan Table 2: number of neighbours


def _mlp(*sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class _AttnConv(nn.Module):
    def __init__(self, hidden, heads, tau):
        super().__init__()
        self.heads, self.d, self.tau = heads, hidden // heads, tau
        self.w = nn.Linear(hidden, heads * self.d, bias=False)

    def forward(self, h, nbr, valid):
        b, n, _ = h.shape
        wh = self.w(h).view(b, n, self.heads, self.d)
        whj = wh[torch.arange(b, device=h.device)[:, None, None], nbr]  # (B, n, K, H, d)
        logits = self.tau * (wh[:, :, None] * whj).sum(-1)              # (B, n, K, H)
        logits = logits.masked_fill(~valid[..., None], float("-inf"))
        att = torch.softmax(logits, dim=2)                               # over neighbours
        out = (att[..., None] * whj).sum(2).reshape(b, n, self.heads * self.d)
        return torch.relu(out), att.permute(0, 1, 3, 2)                  # att: (B, n, H, K)


class GQNNet(nn.Module):
    def __init__(self, f_o=F_O, f_m=F_M, hidden=64, heads=4, layers=2, tau=0.25):
        super().__init__()
        self.enc = nn.Sequential(_mlp(f_o, hidden, hidden), nn.ReLU())
        self.convs = nn.ModuleList(_AttnConv(hidden, heads, tau) for _ in range(layers))
        self.q = _mlp(f_o + f_m + hidden * (layers + 1), 128, 128, N_ACTIONS)

    def forward(self, o, m, nbr, valid):
        h = self.enc(o)
        feats, att = [o, m, h], None
        for conv in self.convs:
            h, att = conv(h, nbr, valid)
            feats.append(h)
        return self.q(torch.cat(feats, dim=-1)), att


class YuanAgent:
    def __init__(self, lr=1e-3, gamma=0.98, batch=10, buffer=10_000, soft_tau=0.01, kl_lambda=0.03,
                 eps_start=0.6, eps_end=0.05, eps_decay_steps=50_000, seed=0, device=None):
        self.hp = dict(lr=lr, gamma=gamma, batch=batch, buffer=buffer, soft_tau=soft_tau,
                       kl_lambda=kl_lambda, eps_start=eps_start, eps_end=eps_end,
                       eps_decay_steps=eps_decay_steps, seed=seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = GQNNet().to(self.device)
        self.target = GQNNet().to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.mem, self.mem_pos = [], 0
        self.steps = 0
        self.last_action_idx = None

    def _prep(self, obs):
        slots = neighbor_slots(obs)
        o, m = agent_features(obs, slots)
        n = len(o)
        alive = alive_mask(obs)
        nbr = np.concatenate([np.arange(n)[:, None], slots[:, :N_NBR]], axis=1)
        valid = np.concatenate([np.ones((n, 1), bool),
                                (slots[:, :N_NBR] >= 0) & alive[np.maximum(slots[:, :N_NBR], 0)]], axis=1)
        nbr = np.where(valid, nbr, np.arange(n)[:, None])
        return dict(o=o.astype(np.float32), m=m.astype(np.float32), nbr=nbr, valid=valid,
                    mask=action_mask(obs, slots), alive=alive), slots

    def _tensors(self, items, keys):
        return [torch.as_tensor(np.stack([it[k] for it in items]), device=self.device) for k in keys]

    def _q(self, net, items):
        o, m, nbr, valid = self._tensors(items, ("o", "m", "nbr", "valid"))
        return net(o, m, nbr.long(), valid)

    def _eps(self):
        f = min(self.steps / self.hp["eps_decay_steps"], 1.0)
        return self.hp["eps_start"] + f * (self.hp["eps_end"] - self.hp["eps_start"])

    def act(self, obs, explore=False):
        s, slots = self._prep(obs)
        with torch.no_grad():
            q = self._q(self.net, [s])[0][0].cpu().numpy()
        q = np.where(s["mask"], q, -np.inf)
        a = q.argmax(axis=1)
        if explore:
            eps = self._eps()
            for i in np.flatnonzero(self.rng.random(len(a)) < eps):
                a[i] = self.rng.choice(np.flatnonzero(s["mask"][i]))
            self.steps += 1
        self.last_action_idx = a
        return decode(obs, slots, a)

    def _target_values(self, rewards, nexts, done):
        with torch.no_grad():
            qn = self._q(self.target, nexts)[0]
            mask = self._tensors(nexts, ("mask",))[0]
            qmax = qn.masked_fill(~mask, float("-inf")).max(-1).values
        r = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)
        d = torch.as_tensor(np.asarray(done, dtype=np.float32), device=self.device)[:, None]
        return r + self.hp["gamma"] * (1 - d) * qmax

    def td_target(self, obs, rewards, next_obs, done):
        nxt, _ = self._prep(next_obs)
        return self._target_values([rewards], [nxt], [done])[0].cpu().numpy()

    def learn(self, obs, action_idx, rewards, next_obs, done):
        s, _ = self._prep(obs)
        nxt, _ = self._prep(next_obs)
        item = (s, np.asarray(action_idx), np.asarray(rewards, dtype=np.float32), nxt, float(done))
        if len(self.mem) < self.hp["buffer"]:
            self.mem.append(item)
        else:
            self.mem[self.mem_pos] = item
        self.mem_pos = (self.mem_pos + 1) % self.hp["buffer"]
        if len(self.mem) < self.hp["batch"]:
            return {}
        idx = self.rng.choice(len(self.mem), self.hp["batch"], replace=False)
        batch = [self.mem[i] for i in idx]
        states, nexts = [b[0] for b in batch], [b[3] for b in batch]
        y = self._target_values([b[2] for b in batch], nexts, [b[4] for b in batch])

        q, att = self._q(self.net, states)
        a = torch.as_tensor(np.stack([b[1] for b in batch]), device=self.device).long()
        q_sa = q.gather(-1, a[..., None])[..., 0]
        with torch.no_grad():
            att_next = self._q(self.net, nexts)[1]  # paper: target attention from the online net
        alive = self._tensors(states, ("alive",))[0].float()
        kl = (att * (torch.log(att + 1e-8) - torch.log(att_next + 1e-8))).sum(-1).mean(-1)
        per_agent = (y - q_sa) ** 2 + self.hp["kl_lambda"] * kl
        loss = (per_agent * alive).sum() / alive.sum().clamp(min=1)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.target.parameters()):
                tp.mul_(1 - self.hp["soft_tau"]).add_(self.hp["soft_tau"] * p)
        return {"loss": loss.item()}

    def save(self, path):
        torch.save(dict(hp=self.hp, net=self.net.state_dict(), target=self.target.state_dict(),
                        steps=self.steps), path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu")
        ag = cls(**ck["hp"], device=device)
        ag.net.load_state_dict(ck["net"])
        ag.target.load_state_dict(ck["target"])
        ag.steps = ck["steps"]
        return ag
