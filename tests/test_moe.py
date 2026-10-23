"""python tests/test_moe.py  — fast CPU checks, no data needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from gds_part_colab import GDSParT, GDSParTConfig, make_synthetic_batch
from moe import GDSParTMoE, GDSParTMoEConfig, upcycle_state_dict

SMALL = dict(embed_dim=32, n_heads=4, n_layers=3, ffn_dim=64, pairwise_embed_dim=16, max_particles=24,
             dropout=0.0)


def test_upcycle_matches_dense():
    torch.manual_seed(0)
    dense = GDSParT(GDSParTConfig(**SMALL)).eval()
    for p in dense.parameters():  # identity gates start at exactly 1; perturb so they matter
        p.data += 0.02 * torch.randn_like(p)
    moe = GDSParTMoE(GDSParTMoEConfig(n_experts=4, top_k=2, **SMALL)).eval()
    moe.load_state_dict(upcycle_state_dict(dense.state_dict(), moe, noise=0.0))
    f, u, m, _ = make_synthetic_batch(6, SMALL["max_particles"], dense.cfg)
    with torch.no_grad():
        assert torch.allclose(dense(f, u, m), moe(f, u, m), atol=1e-5)
    print("ok: upcycled MoE reproduces the dense model")


def test_padding_and_aux():
    torch.manual_seed(1)
    cfg = GDSParTMoEConfig(n_experts=4, top_k=2, moe_layers=[1, 2], **SMALL)
    model = GDSParTMoE(cfg).eval()
    f, u, m, _ = make_synthetic_batch(4, cfg.max_particles, cfg)
    m[:, -4:] = False
    with torch.no_grad():
        a = model(f, u, m)
        f2 = f.clone(); f2[:, -4:] = 99.0
        b = model(f2, u, m)
    assert torch.allclose(a, b, atol=1e-5), "padding leaked into the output"
    model.set_record(True)
    with torch.no_grad():
        model(f, u, m)
    for _, layer in model.moe_layers():
        assert layer.last["top"].shape[0] == int(m.sum()), "routing should cover real particles only"
    model.set_record(False)
    assert [i for i, _ in model.moe_layers()] == [1, 2]
    assert model.aux_loss.item() > 0
    print("ok: padding ignored, routing recorded per real particle, aux loss > 0")


def test_train_step_and_knockout():
    torch.manual_seed(2)
    cfg = GDSParTMoEConfig(n_experts=4, top_k=2, **SMALL)
    model = GDSParTMoE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    f, u, m, y = make_synthetic_batch(8, cfg.max_particles, cfg)
    loss = F.cross_entropy(model(f, u, m), y) + model.aux_loss
    loss.backward()
    routers = [layer.router.weight.grad for _, layer in model.moe_layers()]
    assert all(g is not None and g.abs().sum() > 0 for g in routers), "router got no gradient"
    opt.step()
    model.eval()
    model.disable_experts({0: [0, 1]})
    model.set_record(True)
    with torch.no_grad():
        model(f, u, m)
    top = model.moe_layers()[0][1].last["top"]
    assert not ((top == 0) | (top == 1)).any(), "disabled experts were still used"
    model.disable_experts(None); model.set_record(False)
    print("ok: training step updates routers; knockout removes experts")


if __name__ == "__main__":
    test_upcycle_matches_dense()
    test_padding_and_aux()
    test_train_step_and_knockout()
    print("ALL TESTS PASSED")
