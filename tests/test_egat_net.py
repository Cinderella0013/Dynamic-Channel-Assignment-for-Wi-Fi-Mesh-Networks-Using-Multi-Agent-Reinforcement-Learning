import torch
from mesh.model.egat import EGATConfig, EdgeGATv2, EGATNet


def _inputs(n=6, B=2, d_edge=7):
    x = torch.randn(B, n, 38)
    e = torch.randn(B, n, n, d_edge)
    adj = torch.eye(n, dtype=torch.bool).repeat(B, 1, 1)
    adj[:, 0, 1] = adj[:, 1, 0] = True
    h = torch.zeros(B, n, 64)
    ks = torch.zeros(B, n, 4)
    kc = torch.zeros(B, n, 6)
    sm = torch.ones(B, n, 4, dtype=torch.bool)
    sm[..., 3] = False
    return x, e, adj, h, ks, kc, sm


def test_shapes_and_route_mask():
    out = EGATNet(EGATConfig())(*_inputs())
    assert out["chan_logits"].shape == (2, 6, 6) and out["route_logits"].shape == (2, 6, 4)
    assert out["value"].shape == (2, 6) and out["h"].shape == (2, 6, 64)
    assert (out["route_logits"][..., 3] < -1e8).all()


def test_attention_respects_adjacency():
    layer = EdgeGATv2(8, 7, 2, 4)
    h = torch.randn(1, 4, 8)
    e = torch.randn(1, 4, 4, 7)
    adj = torch.eye(4, dtype=torch.bool)[None]
    out1 = layer(h, e, adj)
    h2 = h.clone()
    h2[0, 3] += 10.0
    assert torch.allclose(layer(h2, e, adj)[0, :3], out1[0, :3])


def test_edge_features_matter_only_when_enabled():
    torch.manual_seed(0)
    x, e, adj, h, ks, kc, sm = _inputs()
    on = EGATNet(EGATConfig(use_edge_feat=True)).eval()
    off = EGATNet(EGATConfig(use_edge_feat=False)).eval()
    assert not torch.allclose(on(x, e, adj, h, ks, kc, sm)["route_logits"],
                              on(x, e * 0, adj, h, ks, kc, sm)["route_logits"])
    assert torch.allclose(off(x, e, adj, h, ks, kc, sm)["route_logits"],
                          off(x, e * 0, adj, h, ks, kc, sm)["route_logits"])


def test_ko_prior_shifts_logits_toward_low_cost():
    x, e, adj, h, ks, kc, sm = _inputs()
    net = EGATNet(EGATConfig(use_ko_prior=True)).eval()
    base = net(x, e, adj, h, ks, kc, sm)["route_logits"]
    ks2 = ks.clone()
    ks2[..., 0] = -3.0
    assert (net(x, e, adj, h, ks2, kc, sm)["route_logits"][..., 0] > base[..., 0]).all()
