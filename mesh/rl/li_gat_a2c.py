"""Li et al. 2025 GAT-A2C adaptation: GAT over the conflict (interference) graph,
z-scored node inputs, actor/critic on [x || z], 1-step on-policy A2C.
See docs/baselines-adaptation.md for departures from the paper.
"""
import numpy as np
import torch
from torch import nn

from mesh.rl.agents_io import (F_M, F_O, N_ACTIONS, action_mask, agent_features, alive_mask,
                               decode, neighbor_slots)
from mesh.sim.env import edge_matrix


class _GATLayer(nn.Module):
    def __init__(self, f_in, heads, d, dropout):
        super().__init__()
        self.heads, self.d = heads, d
        self.w = nn.Linear(f_in, heads * d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, d) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, d) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        b, n, _ = x.shape
        wh = self.w(x).view(b, n, self.heads, self.d)
        e = (wh * self.a_src).sum(-1)[:, :, None] + (wh * self.a_dst).sum(-1)[:, None]  # (B, i, j, H)
        e = nn.functional.leaky_relu(e, 0.2).masked_fill(~adj[..., None], float("-inf"))
        att = self.drop(torch.softmax(e, dim=2))
        return torch.einsum("bijh,bjhd->bihd", att, wh)


class GATA2CNet(nn.Module):
    def __init__(self, f_in=F_O + F_M, heads=8, hidden=8, out=16, dropout=0.6):
        super().__init__()
        self.register_buffer("mu", torch.zeros(f_in))
        self.register_buffer("var", torch.ones(f_in))
        self.in_drop = nn.Dropout(dropout)
        self.gat1 = _GATLayer(f_in, heads, hidden, dropout)
        self.gat2 = _GATLayer(heads * hidden, 1, out, dropout)
        self.actor = nn.Sequential(nn.Linear(f_in + out, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(),
                                   nn.Linear(250, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(f_in + out, 500), nn.ReLU(), nn.Linear(500, 250), nn.ReLU(),
                                    nn.Linear(250, 1))

    def forward(self, x, adj, mask):
        if self.training:  # Li Eq. 10: z-score with training statistics (running estimate)
            flat = x.reshape(-1, x.shape[-1])
            self.mu.mul_(0.99).add_(0.01 * flat.mean(0).detach())
            self.var.mul_(0.99).add_(0.01 * flat.var(0, unbiased=False).detach())
        x = (x - self.mu) / (self.var.sqrt() + 1e-6)
        b, n, _ = x.shape
        z = nn.functional.elu(self.gat1(self.in_drop(x), adj).reshape(b, n, -1))
        z = self.gat2(z, adj).reshape(b, n, -1)
        h = torch.cat([x, z], dim=-1)
        logits = self.actor(h).masked_fill(~mask, -1e9)
        return logits, self.critic(h)[..., 0]


class LiAgent:
    def __init__(self, lr=5e-3, lr_min=1e-4, lr_decay=0.96, lr_decay_every=1000, gamma=0.98,
                 entropy=0.01, value_coef=0.5, seed=0, device=None):
        self.hp = dict(lr=lr, lr_min=lr_min, lr_decay=lr_decay, lr_decay_every=lr_decay_every,
                       gamma=gamma, entropy=entropy, value_coef=value_coef, seed=seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.net = GATA2CNet().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.updates = 0
        self.last_action_idx = None

    @staticmethod
    def adjacency(obs):
        alive = alive_mask(obs)
        adj = (edge_matrix(obs, "is_conflict") > 0) & alive[:, None] & alive[None, :]
        return adj | np.eye(len(alive), dtype=bool)

    def _prep(self, obs):
        slots = neighbor_slots(obs)
        o, m = agent_features(obs, slots)
        t = lambda a, dt: torch.as_tensor(a[None], dtype=dt, device=self.device)
        return (t(np.concatenate([o, m], 1), torch.float32), t(self.adjacency(obs), torch.bool),
                t(action_mask(obs, slots), torch.bool)), slots

    def act(self, obs, explore=False):
        s, slots = self._prep(obs)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(*s)[0][0].cpu()
        if explore:
            a = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.gen)[:, 0].numpy()
        else:
            a = logits.argmax(-1).numpy()
        self.last_action_idx = a
        return decode(obs, slots, a)

    def learn(self, obs, action_idx, rewards, next_obs, done):
        s, _ = self._prep(obs)
        nxt, _ = self._prep(next_obs)
        self.net.train()
        logits, v = self.net(*s)
        with torch.no_grad():
            v_next = self.net(*nxt)[1]
        r = torch.as_tensor(np.asarray(rewards, dtype=np.float32)[None], device=self.device)
        adv = r + self.hp["gamma"] * (1.0 - float(done)) * v_next - v
        logp = torch.log_softmax(logits, -1)
        a = torch.as_tensor(np.asarray(action_idx)[None], device=self.device).long()
        logp_a = logp.gather(-1, a[..., None])[..., 0]
        ent = -(logp.exp() * logp).sum(-1)
        alive = torch.as_tensor(alive_mask(obs)[None], dtype=torch.float32, device=self.device)
        per_agent = (-logp_a * adv.detach() + self.hp["value_coef"] * adv ** 2
                     - self.hp["entropy"] * ent)
        loss = (per_agent * alive).sum() / alive.sum().clamp(min=1)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.hp["lr_decay_every"] == 0:
            for g in self.opt.param_groups:
                g["lr"] = max(g["lr"] * self.hp["lr_decay"], self.hp["lr_min"])
        return {"loss": loss.item()}

    def save(self, path):
        torch.save(dict(hp=self.hp, net=self.net.state_dict(), updates=self.updates), path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu")
        ag = cls(**ck["hp"], device=device)
        ag.net.load_state_dict(ck["net"])
        ag.updates = ck["updates"]
        return ag
