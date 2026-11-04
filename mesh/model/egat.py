"""EGAT-MAPPO network: edge-aware GATv2 encoder + GRU + two-timescale heads + Ko residual prior
+ centralized critic (spec section 2)."""
import math
from dataclasses import dataclass

import torch
from torch import nn

from mesh.rl.agents_io import F_M, F_O, K_SLOTS, N_DL_CH
from mesh.sim.env import EDGE_FEATS


@dataclass
class EGATConfig:
    # components (ablation flags, spec 5.6)
    use_edge_feat: bool = True
    use_gru: bool = True
    two_timescale: bool = True
    t_c: int = 10
    progress_mask: bool = True
    use_bc: bool = True
    use_ko_prior: bool = True
    use_fallback: bool = True
    seq_decode: bool = True  # sequential conflict-aware greedy channels (shared with all learners)
    # network
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    # MAPPO
    lr: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    rollout: int = 1000
    minibatch: int = 50
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad: float = 0.5
    # behaviour cloning from Ko
    bc_steps: int = 10_000
    bc_epochs: int = 20
    bc_lr: float = 1e-3
    # reward mixing (spec section 3) and fallback
    lam_start: float = 0.5
    lam_end: float = 0.1
    fallback_entropy: float = 2.0
    # greedy decoding only: logit bonus for keeping the current channel / parent (switch hysteresis, 0 = off)
    switch_margin: float = 0.0
    # v1 (spec addendum 3): anchor PPO to the BC policy, n-step channel credit, lower lr after BC
    use_kl_anchor: bool = False
    kl_start: float = 0.1
    kl_end: float = 0.01
    semi_mdp: bool = False
    lr_after_bc: float | None = None


def _mlp(*sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class EdgeGATv2(nn.Module):
    """GATv2 with edge features: s_ij = a^T LeakyReLU(W_s h_i + W_t h_j + W_e e_ij)."""

    def __init__(self, d_in, d_edge, heads, d_head):
        super().__init__()
        self.heads, self.d = heads, d_head
        self.w_s = nn.Linear(d_in, heads * d_head, bias=False)
        self.w_t = nn.Linear(d_in, heads * d_head, bias=False)
        self.w_e = nn.Linear(d_edge, heads * d_head, bias=False) if d_edge else None
        self.a = nn.Parameter(torch.randn(heads, d_head) * 0.1)

    def forward(self, h, e, adj):
        b, n, _ = h.shape
        s = self.w_s(h).view(b, n, 1, self.heads, self.d)
        t = self.w_t(h).view(b, 1, n, self.heads, self.d)
        z = s + t
        if self.w_e is not None:
            z = z + self.w_e(e).view(b, n, n, self.heads, self.d)
        score = (nn.functional.leaky_relu(z, 0.2) * self.a).sum(-1)       # (B, i, j, H)
        att = torch.softmax(score.masked_fill(~adj[..., None], float("-inf")), dim=2)
        out = torch.einsum("bijh,bjhd->bihd", att, t.view(b, n, self.heads, self.d))
        return out.reshape(b, n, self.heads * self.d)


class EGATNet(nn.Module):
    def __init__(self, cfg, f_node=F_O + F_M, d_edge=len(EDGE_FEATS)):
        super().__init__()
        self.cfg = cfg
        hid = cfg.hidden
        d_e = d_edge if cfg.use_edge_feat else 0
        self.inp = nn.Linear(f_node, hid)
        self.gats = nn.ModuleList(EdgeGATv2(hid, d_e, cfg.heads, hid // cfg.heads) for _ in range(cfg.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hid) for _ in range(cfg.layers))
        self.gru = nn.GRUCell(hid, hid) if cfg.use_gru else None
        self.chan = _mlp(hid, 64, N_DL_CH)
        self.route = _mlp(hid, 64, K_SLOTS)
        beta0 = math.log(math.e - 1)  # softplus(beta0) = 1
        self.beta_c = nn.Parameter(torch.tensor(beta0))
        self.beta_r = nn.Parameter(torch.tensor(beta0))
        self.critic = _mlp(2 * hid, 128, 1)

    def forward(self, x, e, adj, h_prev, ko_slot, ko_chan, slot_mask):
        z = torch.relu(self.inp(x))
        for gat, norm in zip(self.gats, self.norms):
            z = torch.relu(norm(z + gat(z, e, adj)))
        if self.gru is not None:
            b, n, hid = z.shape
            z = self.gru(z.reshape(b * n, hid), h_prev.reshape(b * n, hid)).view(b, n, hid)
        chan = self.chan(z)
        route = self.route(z)
        if self.cfg.use_ko_prior:
            chan = chan - nn.functional.softplus(self.beta_c) * torch.nan_to_num(ko_chan, posinf=0.0)
            route = route - nn.functional.softplus(self.beta_r) * torch.nan_to_num(ko_slot, posinf=0.0)
        route = route.masked_fill(~slot_mask, -1e9)
        pooled = z.mean(dim=1, keepdim=True).expand_as(z)
        return dict(chan_logits=chan, route_logits=route,
                    value=self.critic(torch.cat([z, pooled], -1))[..., 0], h=z)
